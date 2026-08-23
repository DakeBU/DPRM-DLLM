#!/usr/bin/env python3
"""Precompute train-test-matched teacher-forced Omni T2I trajectories.

Each source example is rolled out three times with Omni's native entropy-penalty
sampler. The random, confidence, and DPRM trajectories use their deployed
orders. The DPRM trajectory uses the same frozen controller as inference. Selected
visual values are teacher-forced to the clean target after every action. No
CLIP model, terminal rollout, or test-time candidate selection is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dprm.omni_order import (
    OmniRankBucketDPRM,
    OmniBucketTableDPRM,
    OmniStageRankCodeDPRM,
    OmniStageRankSpatialDPRM,
    candidate_visual_indices,
    load_omni_order_controller,
    visual_candidate_mask,
)


MASK_ID = 151666
IMAGE_OFFSET = 168072
IMAGE_VOCAB = 8192
DEFAULT_POST_ACTION_CHECKPOINTS = (31, 63, 95, 127, 159, 191, 223)


def training_next_action_steps(checkpoints: set[int]) -> list[int]:
    """Map saved post-action canvases to the actions trained from those canvases."""
    return sorted(step + 1 for step in checkpoints)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenized_example(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        [messages[0]], tokenize=True, add_generation_prompt=True
    )
    complete = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    suffix = complete[len(prompt) :]
    visual = [token for token in suffix if IMAGE_OFFSET <= token < IMAGE_OFFSET + IMAGE_VOCAB]
    if len(suffix) != 260 or len(visual) != 256:
        raise ValueError(
            f"expected a 260-token T2I suffix with 256 visual codes; "
            f"found {len(suffix)} and {len(visual)}"
        )
    return prompt, suffix


def rollout_teacher_forced(
    *,
    model,
    tokenizer,
    prompt_ids: list[int],
    clean_suffix: list[int],
    policy: str,
    controller: OmniRankBucketDPRM
    | OmniStageRankSpatialDPRM
    | OmniStageRankCodeDPRM
    | OmniBucketTableDPRM
    | None,
    checkpoints: set[int],
) -> tuple[dict[int, list[int]], list[dict]]:
    """Roll out one exact policy and return post-action revealed canvases."""
    device = next(model.parameters()).device
    prompt_length = len(prompt_ids)
    clean = torch.tensor(clean_suffix, dtype=torch.long, device=device)
    visual_suffix = (clean >= IMAGE_OFFSET) & (clean < IMAGE_OFFSET + IMAGE_VOCAB)
    visual_positions = torch.where(visual_suffix)[0]
    states: dict[int, list[int]] = {}
    actions: list[dict] = []

    def teacher_force(step, x, logits):
        del logits
        suffix = x[0, prompt_length : prompt_length + clean.numel()]
        if step is None:
            suffix[:] = MASK_ID
            suffix[~visual_suffix] = clean[~visual_suffix]
            return x

        current = suffix[visual_suffix]
        revealed = current.ne(MASK_ID)
        current[revealed] = clean[visual_suffix][revealed]
        suffix[visual_suffix] = current
        step_int = int(step)
        if step_int in checkpoints:
            states[step_int] = torch.where(revealed)[0].detach().cpu().tolist()
        return x

    def dprm_score_hook(confidence: torch.Tensor, **context) -> torch.Tensor:
        if controller is None:
            return confidence
        visual = visual_candidate_mask(context["mask_index"], context["block_mask"]).to(
            confidence.device
        )
        visual_confidence = confidence.clone()
        visual_confidence[~visual] = -torch.inf
        if isinstance(controller, OmniStageRankCodeDPRM):
            visual_adjusted, _ = controller.score(
                visual_confidence,
                step=int(context["step"]),
                provisional_token_ids=context["x0"],
            )
        elif isinstance(controller, (OmniStageRankSpatialDPRM, OmniBucketTableDPRM)):
            visual_indices = candidate_visual_indices(
                context["mask_index"], context["block_mask"]
            ).to(confidence.device)
            visual_adjusted, _ = controller.score(
                visual_confidence,
                step=int(context["step"]),
                visual_indices=visual_indices,
            )
        else:
            visual_adjusted, _ = controller.score(
                visual_confidence, step=int(context["step"])
            )
        adjusted = confidence.clone()
        adjusted[visual] = visual_adjusted[visual]
        return adjusted

    def observer(confidence, transfer_index, number_transfer_tokens, **context):
        visual = visual_candidate_mask(context["mask_index"], context["block_mask"]).to(
            confidence.device
        )
        masked_positions = torch.where(context["mask_index"][0])[0]
        block_start = int(torch.where(context["block_mask"])[0][0].item())
        selected_positions = masked_positions[transfer_index]
        selected_visual = (
            selected_positions[(selected_positions - block_start >= 1) & (selected_positions - block_start <= 256)]
            - block_start
            - 1
        )
        actions.append(
            {
                "step": int(context["step"]),
                "selected_visual_indices": selected_visual.detach().cpu().tolist(),
                "number_transfer_tokens": int(number_transfer_tokens),
                "candidate_visual_count": int(visual.sum().item()),
            }
        )

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.generate(
        input_ids,
        temperature=0.0,
        top_p=0.9,
        steps=260,
        max_new_tokens=260,
        alg="entropy-penalty",
        cfg=0.0,
        tokenizer=tokenizer,
        max_position_penalty=2.0,
        repeat_penalty=1.2,
        output_text_only=False,
        task="T2I",
        order_policy=policy,
        dprm_warmup_steps=0,
        generation_tokens_hook_func=teacher_force,
        generation_order_score_hook_func=dprm_score_hook if controller else None,
        generation_order_observer_func=observer,
    )
    selected_visual_indices = [
        int(index)
        for action in actions
        for index in action.get("selected_visual_indices", [])
    ]
    if sorted(selected_visual_indices) != list(range(256)):
        raise RuntimeError(
            f"{policy} trajectory must select each visual index exactly once; "
            f"observed {len(selected_visual_indices)} actions over "
            f"{len(set(selected_visual_indices))} unique positions"
        )
    missing = sorted(checkpoints - states.keys())
    if missing:
        raise RuntimeError(f"trajectory did not capture checkpoints {missing}")
    return states, actions


def trajectory_rows(row: dict, policy: str, states: dict[int, list[int]]) -> list[dict]:
    output = []
    for step, revealed in sorted(states.items()):
        item = dict(row)
        item["dprm_revealed_visual_indices"] = revealed
        item["dprm_trajectory_policy"] = policy
        item["dprm_trajectory_step"] = int(step)
        item["dprm_next_action_step"] = int(step) + 1
        output.append(item)
    return output


def actions_by_step(actions: list[dict]) -> dict[int, dict]:
    """Index observer records by decode step and reject ambiguous traces."""
    indexed: dict[int, dict] = {}
    for action in actions:
        step = int(action["step"])
        if step in indexed:
            raise RuntimeError(f"duplicate order-observer record at step {step}")
        indexed[step] = action
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--confidence-output", type=Path, required=True)
    parser.add_argument("--dprm-output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=list(DEFAULT_POST_ACTION_CHECKPOINTS)
    )
    parser.add_argument("--forbidden-prompts", type=Path)
    args = parser.parse_args()

    checkpoints = set(args.checkpoints)
    if args.count <= 0 or not checkpoints or min(checkpoints) < 0 or max(checkpoints) >= 255:
        raise SystemExit("count must be positive and checkpoints must lie in [0, 254]")
    controller, controller_metadata = load_omni_order_controller(args.controller)
    next_action_steps = training_next_action_steps(checkpoints)
    reward_action_steps = sorted(
        int(step)
        for step in getattr(
            controller,
            "reward_action_steps",
            getattr(controller, "active_steps", ()),
        )
    )
    uncovered_reward_steps = sorted(set(reward_action_steps) - set(next_action_steps))
    if uncovered_reward_steps:
        raise SystemExit(
            "post-action checkpoints do not train all deployed DPRM reward actions; "
            f"missing next-action steps {uncovered_reward_steps}"
        )
    score_contract = controller_metadata.get("score_contract", {})
    if score_contract.get("base_order_score") != "negative_token_entropy":
        raise SystemExit("matched trajectories require the negative-token-entropy score contract")
    if score_contract.get("bucket_coordinate") not in {
        "exp_negative_token_entropy",
        "within_state_confidence_rank",
        "within_state_confidence_rank_and_provisional_code",
    }:
        raise SystemExit("matched trajectories require a declared bucket coordinate")
    if score_contract.get("position_selection_rule") != "single_path_top1_adjusted_order_score":
        raise SystemExit("matched trajectories require the single-path top-1 selection contract")
    deployment = controller_metadata.get("deployment_contract", {})
    if deployment.get("fixed_t2i_scaffold") is not True:
        raise SystemExit("matched trajectories require a fixed-T2I-scaffold controller")
    if deployment.get("ordered_visual_positions") != 256:
        raise SystemExit("matched trajectories require exactly 256 ordered visual positions")
    if deployment.get("complete_image_selection") is not False:
        raise SystemExit("matched trajectories cannot use completed-image selection")
    forbidden = set()
    if args.forbidden_prompts:
        forbidden = {
            line.strip()
            for line in args.forbidden_prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    final_outputs = (args.random_output, args.confidence_output, args.dprm_output)
    for path in (*final_outputs, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_outputs = tuple(
        path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in final_outputs
    )
    temporary_manifest = args.manifest_output.with_name(
        f".{args.manifest_output.name}.{os.getpid()}.tmp"
    )

    audit = []
    completed = 0
    rows_written = 0
    with (
        args.data_json.open(encoding="utf-8") as source,
        temporary_outputs[0].open("w", encoding="utf-8") as random_handle,
        temporary_outputs[1].open("w", encoding="utf-8") as confidence_handle,
        temporary_outputs[2].open("w", encoding="utf-8") as dprm_handle,
        torch.inference_mode(),
    ):
        for source_index, line in enumerate(source):
            if source_index < args.offset:
                continue
            if completed >= args.count:
                break
            row = json.loads(line)
            prompt_text = row["messages"][0]["content"].strip()
            if prompt_text in forbidden or "\n".join(prompt_text.split("\n")[1:]).strip() in forbidden:
                continue
            prompt_ids, clean_suffix = tokenized_example(row["messages"], tokenizer)
            stable_source_index = int(row.get("dprm_source_index", source_index))
            torch.manual_seed(20260813 + stable_source_index)
            random_states, random_actions = rollout_teacher_forced(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                clean_suffix=clean_suffix,
                policy="random",
                controller=None,
                checkpoints=checkpoints,
            )
            confidence_states, confidence_actions = rollout_teacher_forced(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                clean_suffix=clean_suffix,
                policy="progressive_confidence",
                controller=None,
                checkpoints=checkpoints,
            )
            dprm_states, dprm_actions = rollout_teacher_forced(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                clean_suffix=clean_suffix,
                policy="dprm_confidence_warmup",
                controller=controller,
                checkpoints=checkpoints,
            )
            random_rows = trajectory_rows(row, "random", random_states)
            confidence_rows = trajectory_rows(row, "progressive_confidence", confidence_states)
            dprm_rows = trajectory_rows(row, "dprm_confidence_warmup", dprm_states)
            for item in random_rows:
                random_handle.write(json.dumps(item) + "\n")
            for item in confidence_rows:
                confidence_handle.write(json.dumps(item) + "\n")
            for item in dprm_rows:
                dprm_handle.write(json.dumps(item) + "\n")
            confidence_actions_by_step = actions_by_step(confidence_actions)
            dprm_actions_by_step = actions_by_step(dprm_actions)
            paired_steps = sorted(
                set(confidence_actions_by_step) & set(dprm_actions_by_step)
            )
            differing_steps = [
                step
                for step in paired_steps
                if confidence_actions_by_step[step]["selected_visual_indices"]
                != dprm_actions_by_step[step]["selected_visual_indices"]
            ]
            audit.append(
                {
                    "source_index": stable_source_index,
                    "paired_action_steps": len(paired_steps),
                    "differing_action_steps": len(differing_steps),
                    "first_differing_step": differing_steps[0] if differing_steps else None,
                    "actions_differ": bool(differing_steps),
                    "confidence_revealed": {str(k): len(v) for k, v in confidence_states.items()},
                    "dprm_revealed": {str(k): len(v) for k, v in dprm_states.items()},
                }
            )
            completed += 1
            rows_written += len(checkpoints)
            print(f"completed {completed}/{args.count}", flush=True)
    if completed != args.count:
        raise RuntimeError(f"requested {args.count} samples, completed {completed}")
    manifest = {
        "format": "omni_matched_teacher_forced_trajectory_v2",
        "model_path": args.model_path,
        "source": str(args.data_json),
        "offset": args.offset,
        "source_examples": completed,
        "checkpoints": sorted(checkpoints),
        "post_action_checkpoints": sorted(checkpoints),
        "training_next_action_steps": next_action_steps,
        "controller_reward_action_steps": reward_action_steps,
        "reward_action_coverage_verified": True,
        "rows_per_policy": rows_written,
        "controller": str(args.controller),
        "controller_sha256": sha256_file(args.controller),
        "controller_metadata": controller_metadata,
        "confidence_output": str(args.confidence_output),
        "dprm_output": str(args.dprm_output),
        "random_output": str(args.random_output),
        "differing_active_action_fraction": sum(x["actions_differ"] for x in audit) / completed,
        "mean_differing_action_step_fraction": sum(
            x["differing_action_steps"] / max(x["paired_action_steps"], 1) for x in audit
        )
        / completed,
        "native_sampler": "entropy-penalty",
        "order_score": "negative token entropy plus frozen DPRM bucket value",
        "test_time_terminal_rollouts": 0,
        "forbidden_prompt_count": len(forbidden),
        "audit": audit,
    }
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for temporary, final in zip(temporary_outputs, final_outputs):
        temporary.replace(final)
    temporary_manifest.replace(args.manifest_output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "audit"}, indent=2))


if __name__ == "__main__":
    main()
