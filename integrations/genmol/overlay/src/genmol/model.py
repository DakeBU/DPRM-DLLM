# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info

try:
    from dprm import (
        DPRMConfig,
        HostDPRMBatch,
        OnlineDPRMController,
        scalarize_benefits,
    )
except Exception:
    DPRMConfig = None
    HostDPRMBatch = None
    OnlineDPRMController = None
    scalarize_benefits = None

class GenMol(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # set up tokenizer
        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        # set up backbone   
        self.backbone = BertForMaskedLM(BertConfig.from_dict(dict(self.config.model)))
        # set up mdlm
        if self.config.training.antithetic_sampling:
            time_distribution = AntitheticUniformTimeDistribution(sampling_eps = self.config.training.sampling_eps)
        else:
            time_distribution = UniformTimeDistribution()
        prior = DiscreteMaskedPrior(num_classes = self.tokenizer.vocab_size, mask_dim = self.mask_index)
        noise_schedule = LogLinearExpNoiseTransform()
        self.mdlm = MDLM(time_distribution=time_distribution,
                          prior_distribution=prior,
                          noise_schedule = noise_schedule)
        # set up ema
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        else:
            self.ema = None
        self.order_policy = self.config.training.get('order_policy', 'random_mdlm')
        self.order_num_phases = int(self.config.training.get('order_num_phases', 8))
        self.dprm_controller = self._build_dprm_controller()

    def _build_dprm_controller(self):
        if not str(self.order_policy).startswith('dprm'):
            return None
        if OnlineDPRMController is None:
            raise ImportError(
                'DPRM package is not importable. Install DPRM-DLLM with pip install -e.'
            )
        cfg = DPRMConfig(
            num_phases=int(self.config.training.get('dprm_num_phases', self.order_num_phases)),
            confidence_bins=int(self.config.training.get('dprm_confidence_bins', 16)),
            reward_temperature=float(self.config.training.get('dprm_reward_temperature', 1.0)),
            guidance_scale=float(self.config.training.get('dprm_guidance_scale', 1.0)),
            warmup_steps=int(self.config.training.get('dprm_warmup_steps', 500)),
            switch_steps=int(self.config.training.get('dprm_switch_steps', 2000)),
            ready_count=int(self.config.training.get('dprm_ready_count', 128)),
            sampled_soft_bon=bool(self.config.training.get('dprm_sampled_soft_bon', True)),
            candidate_multiplier=int(self.config.training.get('dprm_candidate_multiplier', 4)),
            min_candidates=int(self.config.training.get('dprm_min_candidates', 8)),
            max_candidates=int(self.config.training.get('dprm_max_candidates', 64)),
        )
        return OnlineDPRMController(cfg)

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        if self.dprm_controller is not None and 'dprm_state_dict' in checkpoint:
            self.dprm_controller.load_state_dict(checkpoint['dprm_state_dict'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None
        if self.dprm_controller is not None:
            checkpoint['dprm_state_dict'] = self.dprm_controller.state_dict()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            {'_target_': 'transformers.get_constant_schedule_with_warmup',
             'num_warmup_steps': 2500},
             optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        if self.dprm_controller is not None:
            self.dprm_controller.to(self.device)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters()))
        
    def forward(self, x, attention_mask=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.backbone(x, attention_mask)['logits']

    def _candidate_mask(self, input_ids, attention_mask):
        candidate = attention_mask.bool()
        candidate = candidate & (input_ids != self.mask_index)
        candidate = candidate & (input_ids != self.bos_index)
        candidate = candidate & (input_ids != self.eos_index)
        candidate = candidate & (input_ids != self.tokenizer.pad_token_id)
        return candidate

    def _topk_mask(self, scores, candidate_mask, k, largest=True):
        picked = torch.zeros_like(candidate_mask, dtype=torch.bool)
        masked_scores = scores.masked_fill(~candidate_mask, float('-inf') if largest else float('inf'))
        for row in range(candidate_mask.shape[0]):
            active = torch.nonzero(candidate_mask[row], as_tuple=False).squeeze(1)
            if active.numel() == 0:
                continue
            kk = min(int(k[row].item()), int(active.numel()))
            if kk <= 0:
                continue
            vals = masked_scores[row, active]
            chosen = active[torch.topk(vals, k=kk, largest=largest).indices]
            picked[row, chosen] = True
        return picked

    @torch.no_grad()
    def _ordering_scores(self, input_ids, attention_mask, candidate_mask, phase_ids, policy):
        probe = input_ids.clone()
        probe[candidate_mask] = self.mask_index
        logits = self.backbone(probe, attention_mask)["logits"]
        logprobs = self.mdlm._subs_parameterization(logits, probe)
        probs = F.softmax(logprobs, dim=-1)
        gt_conf = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1).clamp(1e-6, 1.0)
        sampled = torch.distributions.Categorical(probs=probs).sample()
        sample_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).clamp(1e-6, 1.0)
        confidence = torch.maximum(gt_conf, sample_conf)

        if policy in {'random_ordered', 'dprm_random'} and (
            policy == 'random_ordered' or self.global_step < int(self.config.training.get('dprm_warmup_steps', 500))
        ):
            score = torch.rand_like(confidence)
            return score.masked_fill(~candidate_mask, float('-inf')), confidence

        if policy in {'confidence', 'progressive'}:
            return torch.log(confidence).masked_fill(~candidate_mask, float('-inf')), confidence

        if policy in {'dprm', 'dprm_confidence', 'dprm_random'}:
            host = HostDPRMBatch(
                confidence=confidence,
                candidate_mask=candidate_mask,
                phase_ids=phase_ids,
                global_step=int(self.global_step),
            )
            summary = self.dprm_controller.summarize(host)
            return summary.score, confidence

        return torch.rand_like(confidence).masked_fill(~candidate_mask, float('-inf')), confidence

    def _terminal_dprm_reward(self, confidence, revealed, terminal_objectives):
        mode = str(self.config.training.get('dprm_reward_mode', 'selected_confidence'))
        if mode == 'selected_confidence':
            gt_conf = confidence.masked_fill(~revealed, 0.0)
            denom = revealed.sum(dim=1).clamp_min(1)
            return gt_conf.sum(dim=1) / denom
        if terminal_objectives is None:
            raise ValueError(f'{mode} requires terminal_objectives from the data collator')

        weights = torch.as_tensor(
            list(self.config.training.get('dprm_objective_weights', [0.55, 0.45])),
            dtype=terminal_objectives.dtype,
            device=terminal_objectives.device,
        )
        if mode == 'molecular_weighted_sum':
            method = 'weighted_sum'
        elif mode == 'molecular_tchebycheff':
            method = 'smooth_tchebycheff'
        else:
            raise ValueError(f'unknown dprm_reward_mode: {mode}')
        return scalarize_benefits(
            terminal_objectives,
            weights,
            method=method,
            temperature=float(self.config.training.get('dprm_tchebycheff_temperature', 0.05)),
            augmentation=float(self.config.training.get('dprm_tchebycheff_augmentation', 0.05)),
        )

    def _progressive_state(self, input_ids, attention_mask, t, terminal_objectives=None):
        policy = str(self.config.training.get('order_policy', 'random_mdlm'))
        xt_random = self.mdlm.forward_process(input_ids, t)
        candidate_mask = self._candidate_mask(input_ids, attention_mask)
        target_mask_count = ((xt_random == self.mask_index) & candidate_mask).sum(dim=1)
        xt = input_ids.clone()
        if not target_mask_count.any():
            return xt_random, None

        progress = 1.0 - (target_mask_count.float() / candidate_mask.sum(dim=1).clamp_min(1).float())
        phase_ids = torch.clamp((progress * self.order_num_phases).long(), 0, self.order_num_phases - 1)
        scores, confidence = self._ordering_scores(input_ids, attention_mask, candidate_mask, phase_ids, policy)
        mask_positions = self._topk_mask(scores, candidate_mask, target_mask_count, largest=False)
        xt[mask_positions] = self.mask_index

        if self.dprm_controller is not None and policy in {'dprm', 'dprm_confidence', 'dprm_random'}:
            revealed = candidate_mask & ~mask_positions
            reward = self._terminal_dprm_reward(
                confidence, revealed, terminal_objectives
            )
            host = HostDPRMBatch(
                confidence=confidence,
                candidate_mask=candidate_mask,
                phase_ids=phase_ids,
                global_step=int(self.global_step),
            )
            self.dprm_controller.observe(host, revealed, reward)
        return xt, mask_positions
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        terminal_objectives = batch.get('terminal_objectives')
        # sample time
        t = self.mdlm.sample_time(input_ids.shape[0])
        # forward process to add mask tokens
        if self.config.training.get('order_policy', 'random_mdlm') == 'random_mdlm':
            xt = self.mdlm.forward_process(input_ids, t)
        else:
            xt, _ = self._progressive_state(
                input_ids, attention_mask, t, terminal_objectives=terminal_objectives
            )
        # forward model pass
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(xt, attention_mask)["logits"]
        # compute loss
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()
        self.log(name='train_loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return loss
