"""Adapted from https://github.com/NVlabs/Fast-dLLM"""

import os
import sys

import torch
import numpy as np
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from dprm_guidance import OnlineDPRMEstimator, phase_tensor_from_step


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits - temperature * torch.log(-torch.log(noise + 1e-10) + 1e-10) 


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _resolve_remasking_scores(logits, x0, remasking):
    if remasking in {'low_confidence', 'dprm_soft_bon'}:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    if remasking == 'entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        token_confidence = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        return 1.0 - token_confidence
    if remasking == 'random':
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    raise NotImplementedError(remasking)


def _resolve_dprm_phase_tensor(
    estimator: OnlineDPRMEstimator | None,
    batch_size: int,
    device: torch.device,
    step_index: int | None = None,
    total_steps: int | None = None,
    explicit_phase: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if estimator is None:
        return None
    if explicit_phase is not None:
        return explicit_phase.to(device=device, dtype=torch.long)
    if step_index is None or total_steps is None:
        return phase_tensor_from_step(0, 1, estimator.num_phases, batch_size, device)
    return phase_tensor_from_step(step_index, total_steps, estimator.num_phases, batch_size, device)


def _append_transfer_trace(
    trace_steps,
    transfer_index: torch.Tensor,
    x0: torch.Tensor,
    prompt_length: int,
    absolute_offset: int,
    block_index: int,
    step_index: int,
    global_step: int,
) -> None:
    if trace_steps is None:
        return

    selected_by_row = []
    transfer_cpu = transfer_index.detach().cpu()
    x0_cpu = x0.detach().cpu()
    for row in range(transfer_cpu.shape[0]):
        local_cols = transfer_cpu[row].nonzero(as_tuple=False).view(-1).tolist()
        gen_positions = []
        token_ids = []
        for local_col in local_cols:
            absolute_col = int(absolute_offset + local_col)
            gen_pos = absolute_col - int(prompt_length)
            if gen_pos < 0:
                continue
            gen_positions.append(gen_pos)
            token_ids.append(int(x0_cpu[row, local_col].item()))
        selected_by_row.append(
            {
                "batch_index": int(row),
                "selected_positions": gen_positions,
                "selected_token_ids": token_ids,
                "newly_revealed_count": int(len(gen_positions)),
            }
        )

    trace_steps.append(
        {
            "block": int(block_index),
            "step": int(step_index),
            "global_step": int(global_step),
            "selected": selected_by_row,
        }
    )


@torch.no_grad()
def generate_llada(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0.,
                   remasking='low_confidence', mask_id=126336, **kwargs):
    """
    Default generation code of llada (https://github.com/ML-GSAI/LLaDA/blob/main/generate.py)
    """
    bs = prompt.shape[0]
    x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            dprm_phase = _resolve_dprm_phase_tensor(
                kwargs.get("dprm_estimator"),
                batch_size=x.shape[0],
                device=x.device,
                step_index=i,
                total_steps=steps,
            )
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, i],
                dprm_estimator=kwargs.get("dprm_estimator"),
                dprm_phase=dprm_phase,
                dprm_global_step=kwargs.get("dprm_global_step"),
                dprm_force_full=kwargs.get("dprm_force_full", False),
            )
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def generate_pd(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0.,
                remasking='low_confidence', mask_id=126336, threshold_pd=None, factor=None, **kwargs):
    """
    Block-wise confidence-aware parallel decoding
    This is slightly different from the LLaDA sampler in remasking
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            dprm_phase = _resolve_dprm_phase_tensor(
                kwargs.get("dprm_estimator"),
                batch_size=x.shape[0],
                device=x.device,
                step_index=i,
                total_steps=num_transfer_tokens.shape[1],
            )
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold_pd is None else None,
                    threshold_pd,
                    dprm_estimator=kwargs.get("dprm_estimator"),
                    dprm_phase=dprm_phase,
                    dprm_global_step=kwargs.get("dprm_global_step"),
                    dprm_force_full=kwargs.get("dprm_force_full", False),
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x


@torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                               remasking='low_confidence', mask_id=126336, threshold_pd=None, factor=None, **kwargs):
    """
    Block-wise confidence-aware parallel decoding with prefix KV cache
    """
    trace_steps = [] if kwargs.get("return_trace", False) else None
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        dprm_phase = _resolve_dprm_phase_tensor(
            kwargs.get("dprm_estimator"),
            batch_size=x.shape[0],
            device=x.device,
            step_index=0,
            total_steps=num_transfer_tokens.shape[1],
        )
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold_pd is None else None,
                threshold_pd,
                dprm_estimator=kwargs.get("dprm_estimator"),
                dprm_phase=dprm_phase,
                dprm_global_step=kwargs.get("dprm_global_step"),
                dprm_force_full=kwargs.get("dprm_force_full", False),
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        _append_transfer_trace(
            trace_steps,
            transfer_index,
            x0,
            prompt_length=prompt.shape[1],
            absolute_offset=0,
            block_index=num_block,
            step_index=0,
            global_step=num_block * steps,
        )
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            dprm_phase = _resolve_dprm_phase_tensor(
                kwargs.get("dprm_estimator"),
                batch_size=x.shape[0],
                device=x.device,
                step_index=i,
                total_steps=num_transfer_tokens.shape[1],
            )
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    num_transfer_tokens[:, i] if threshold_pd is None else None,
                    threshold_pd,
                    dprm_estimator=kwargs.get("dprm_estimator"),
                    dprm_phase=dprm_phase,
                    dprm_global_step=kwargs.get("dprm_global_step"),
                    dprm_force_full=kwargs.get("dprm_force_full", False),
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x[:, current_block_start:], None, factor)
            _append_transfer_trace(
                trace_steps,
                transfer_index,
                x0,
                prompt_length=prompt.shape[1],
                absolute_offset=current_block_start,
                block_index=num_block,
                step_index=i,
                global_step=num_block * steps + i,
            )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    if trace_steps is not None:
        return x, trace_steps
    return x


@torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                             remasking='low_confidence', mask_id=126336, threshold_pd=None, factor=None, **kwargs):
    """
    Block-wise confidence-aware parallel decoding with dual KV cache
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        dprm_phase = _resolve_dprm_phase_tensor(
            kwargs.get("dprm_estimator"),
            batch_size=x.shape[0],
            device=x.device,
            step_index=0,
            total_steps=num_transfer_tokens.shape[1],
        )
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold_pd is None else None,
                threshold_pd,
                dprm_estimator=kwargs.get("dprm_estimator"),
                dprm_phase=dprm_phase,
                dprm_global_step=kwargs.get("dprm_global_step"),
                dprm_force_full=kwargs.get("dprm_force_full", False),
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            dprm_phase = _resolve_dprm_phase_tensor(
                kwargs.get("dprm_estimator"),
                batch_size=x.shape[0],
                device=x.device,
                step_index=i,
                total_steps=num_transfer_tokens.shape[1],
            )
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    num_transfer_tokens[:, i] if threshold_pd is None else None,
                    threshold_pd,
                    dprm_estimator=kwargs.get("dprm_estimator"),
                    dprm_phase=dprm_phase,
                    dprm_global_step=kwargs.get("dprm_global_step"),
                    dprm_force_full=kwargs.get("dprm_force_full", False),
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold_pd=None,
    dprm_estimator: OnlineDPRMEstimator | None = None,
    dprm_phase: torch.Tensor | None = None,
    dprm_global_step: int | None = None,
    dprm_force_full: bool = False,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    x0_p = _resolve_remasking_scores(logits, x0, remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold_pd is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    if remasking == 'dprm_soft_bon':
        if dprm_estimator is None:
            raise ValueError("dprm_soft_bon remasking requires a DPRM estimator")
        phase = _resolve_dprm_phase_tensor(
            dprm_estimator,
            batch_size=x0.shape[0],
            device=x0.device,
            explicit_phase=dprm_phase,
        )
        transfer_index, _ = dprm_estimator.select_positions(
            probs=x0_p.float(),
            mask=mask_index.bool(),
            num_select=num_transfer_tokens.view(-1),
            phase=phase,
            global_step=int(dprm_global_step if dprm_global_step is not None else dprm_estimator.global_updates),
            force_full=dprm_force_full,
        )
        return x0, transfer_index
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold_pd is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold_pd:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking in {'dprm_soft_bon', 'entropy'}:
        raise NotImplementedError(f"{remasking} is not supported with dynamic factor remasking")
    x0_p = _resolve_remasking_scores(logits, x0, remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
