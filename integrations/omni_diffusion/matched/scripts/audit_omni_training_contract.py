#!/usr/bin/env python3
"""Reject Omni checkpoints that do not share the frozen matched-training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORDERS = ("random_matched", "confidence_matched", "dprm_matched")
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
)


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
    for order in ORDERS:
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

    reference = manifests[ORDERS[0]]
    for order in ORDERS[1:]:
        for field in SHARED_FIELDS:
            if manifests[order].get(field) != reference.get(field):
                raise SystemExit(
                    f"training-contract mismatch for {field}: "
                    f"{ORDERS[0]}={reference.get(field)!r}, "
                    f"{order}={manifests[order].get(field)!r}"
                )
    expected_controller_hash = sha256(args.controller)
    if reference.get("controller_sha256") != expected_controller_hash:
        raise SystemExit("evaluation controller differs from the training controller")
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

    data_hashes = {manifests[order]["data_json_sha256"] for order in ORDERS}
    if len(data_hashes) != len(ORDERS):
        raise SystemExit("policy-specific Omni branches unexpectedly share trajectory JSONL")

    report = {
        "passed": True,
        "step": args.step,
        "orders": list(ORDERS),
        "shared_contract": {field: reference.get(field) for field in SHARED_FIELDS},
        "policy_data_json_sha256": {
            order: manifests[order]["data_json_sha256"] for order in ORDERS
        },
        "branch_manifest_sha256": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
