#!/usr/bin/env python3
"""Reject Omni checkpoints that do not share the frozen matched-training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ORDERS = ("confidence_matched", "dprm_matched")
SHARED_FIELDS = (
    "shared_initial_checkpoint",
    "shared_checkpoint_index_sha256",
    "controller",
    "controller_sha256",
    "trainer_sha256",
    "dataset_sha256",
    "order_code_sha256",
    "inference_hook",
    "inference_hook_sha256",
    "precomputed_trajectory_mode",
    "current_model_policy_refresh",
    "seed",
    "distributed_world_size",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "learning_rate",
    "warmup_ratio",
    "trainable_last_n_layers",
    "reveal_budget",
    "trajectory_stage_contract",
)


def inspect_trajectory_json(path: Path) -> dict:
    post_actions: set[int] = set()
    next_actions: set[int] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prefix = "tpami" if "tpami_trajectory_step" in row else "dprm"
            required = (
                f"{prefix}_trajectory_step",
                f"{prefix}_next_action_step",
                f"{prefix}_revealed_visual_indices",
            )
            missing = [key for key in required if key not in row]
            if missing:
                raise SystemExit(
                    f"missing trajectory fields in {path}: {', '.join(missing)}"
                )
            post = int(row[required[0]])
            next_action = int(row[required[1]])
            revealed = len(row[required[2]])
            if next_action != post + 1 or revealed != next_action:
                raise SystemExit(
                    f"trajectory/action alignment mismatch in {path}: "
                    f"post={post}, next={next_action}, revealed={revealed}"
                )
            post_actions.add(post)
            next_actions.add(next_action)
            rows += 1
    if rows == 0:
        raise SystemExit(f"empty matched trajectory JSONL: {path}")
    return {
        "post_action_checkpoints": sorted(post_actions),
        "training_next_action_steps": sorted(next_actions),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=("random_matched", "confidence_matched", "dprm_matched"),
        default=DEFAULT_ORDERS,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current_code = {
        "trainer_sha256": args.omni_root / "tools" / "trainer_v4_51_3.py",
        "dataset_sha256": args.omni_root
        / "omni_diffusion"
        / "data"
        / "dataset_qwen2.py",
        "order_code_sha256": args.release_root / "src" / "dprm" / "omni_order.py",
    }

    manifests: dict[str, dict] = {}
    evidence: dict[str, str] = {}
    orders = tuple(args.orders)
    if "confidence_matched" not in orders or "dprm_matched" not in orders:
        raise SystemExit("formal Omni audit requires confidence_matched and dprm_matched")
    if len(set(orders)) != len(orders):
        raise SystemExit("formal Omni audit received duplicate branches")
    for order in orders:
        path = args.train_root / order / "branch_manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing branch manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("order") != order:
            raise SystemExit(
                f"branch manifest order mismatch: {payload.get('order')!r} != {order!r}"
            )
        data_json = Path(payload.get("data_json", ""))
        if not data_json.is_file() or sha256(data_json) != payload.get("data_json_sha256"):
            raise SystemExit(f"trajectory JSONL hash mismatch for {order}: {data_json}")
        observed_stage_contract = inspect_trajectory_json(data_json)
        recorded_stage_contract = payload.get("trajectory_stage_contract", {})
        for key, value in observed_stage_contract.items():
            if recorded_stage_contract.get(key) != value:
                raise SystemExit(
                    f"trajectory stage contract mismatch for {order}/{key}: "
                    f"{recorded_stage_contract.get(key)!r} != {value!r}"
                )
        next_actions = observed_stage_contract["training_next_action_steps"]
        expected_loss_counts = [value + 1 for value in next_actions]
        if recorded_stage_contract.get("policy_input_visible_counts") != next_actions:
            raise SystemExit(f"policy-input visible counts are invalid for {order}")
        if recorded_stage_contract.get("loss_state_visible_counts") != expected_loss_counts:
            raise SystemExit(f"loss-state visible counts are invalid for {order}")
        if recorded_stage_contract.get("hybrid_transition_then_loss") is not True:
            raise SystemExit(f"hybrid transition/loss contract is missing for {order}")
        data_config = Path(payload.get("data_config", ""))
        if (
            not data_config.is_file()
            or sha256(data_config) != payload.get("data_config_sha256")
        ):
            raise SystemExit(f"trajectory config hash mismatch for {order}: {data_config}")
        checkpoint = args.train_root / order / f"checkpoint-{args.step}"
        trainer_state_path = checkpoint / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise SystemExit(f"missing trained checkpoint: {checkpoint}")
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        if int(trainer_state.get("global_step", -1)) != args.step:
            raise SystemExit(
                f"checkpoint step mismatch for {order}: "
                f"{trainer_state.get('global_step')!r} != {args.step}"
            )
        manifests[order] = payload
        evidence[order] = sha256(path)

    reference = manifests[orders[0]]
    for order in orders[1:]:
        for field in SHARED_FIELDS:
            if manifests[order].get(field) != reference.get(field):
                raise SystemExit(
                    f"training-contract mismatch for {field}: "
                    f"{orders[0]}={reference.get(field)!r}, "
                    f"{order}={manifests[order].get(field)!r}"
                )
    expected_controller_hash = sha256(args.controller)
    if reference.get("controller_sha256") != expected_controller_hash:
        raise SystemExit("evaluation controller differs from the training controller")
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    reward_actions = controller.get("config", {}).get("reward_action_steps", [])
    if not reward_actions:
        reward_actions = controller.get("config", {}).get("active_steps", [])
    if not reward_actions:
        reward_actions = controller.get("metadata", {}).get(
            "stagewise_order_contract", {}
        ).get("reward_action_steps", [])
    stage_contract = reference.get("trajectory_stage_contract", {})
    next_actions = set(stage_contract.get("training_next_action_steps", []))
    missing_reward_actions = sorted(set(map(int, reward_actions)) - next_actions)
    if missing_reward_actions:
        raise SystemExit(
            "training trajectories do not cover deployed DPRM reward actions: "
            f"{missing_reward_actions}"
        )
    if stage_contract.get("reward_action_coverage_verified") is not True:
        raise SystemExit("branch manifest did not verify DPRM reward-action coverage")
    for field, path in current_code.items():
        if not path.is_file() or sha256(path) != reference.get(field):
            raise SystemExit(f"formal Omni {field} changed after training: {path}")
    inference_hook = Path(reference.get("inference_hook", ""))
    if (
        not inference_hook.is_file()
        or sha256(inference_hook) != reference.get("inference_hook_sha256")
    ):
        raise SystemExit(
            f"formal Omni inference hook changed after training: {inference_hook}"
        )
    if reference.get("precomputed_trajectory_mode") != "hybrid":
        raise SystemExit("formal Omni evaluation requires hybrid policy refresh training")
    if reference.get("current_model_policy_refresh") is not True:
        raise SystemExit("formal Omni branches did not record current-model policy refresh")

    data_hashes = {manifests[order]["data_json_sha256"] for order in orders}
    if len(data_hashes) != len(orders):
        raise SystemExit("policy-specific Omni branches unexpectedly share trajectory JSONL")

    report = {
        "passed": True,
        "step": args.step,
        "orders": list(orders),
        "shared_contract": {field: reference.get(field) for field in SHARED_FIELDS},
        "policy_data_json_sha256": {
            order: manifests[order]["data_json_sha256"] for order in orders
        },
        "branch_manifest_sha256": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
