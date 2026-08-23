#!/usr/bin/env python3
"""Check that released artifacts embed the paper's controller settings."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value


def fields(actual: dict[str, Any], expected: dict[str, Any], prefix: str) -> None:
    for key, value in expected.items():
        check(actual.get(key) == value, f"{prefix}: expected {key}={value!r}, got {actual.get(key)!r}")


def load_torch(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)


def same_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def read_tar_member(path: Path, member: str) -> str:
    return subprocess.check_output(
        ["tar", "--zstd", "-xOf", str(path), member], text=True
    )


def read_archive_member(path: Path, member: str) -> bytes:
    if path.suffix == ".zst":
        return subprocess.check_output(["tar", "--zstd", "-xOf", str(path), member])
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        check(handle is not None, f"missing {member} in {path}")
        return handle.read()


def archive_json(path: Path, member: str) -> dict[str, Any]:
    return json.loads(read_archive_member(path, member).decode("utf-8"))


def archive_jsonl(path: Path, member: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in read_archive_member(path, member).decode("utf-8").splitlines()
        if line.strip()
    ]


def audit_dmpo(root: Path) -> dict[str, Any]:
    task_sizes = {"math": 500, "countdown": 5120}
    result: dict[str, Any] = {}
    for task, expected in task_sizes.items():
        policies = {}
        for policy in ("confidence", "dprm"):
            archive = root / f"dmpo/records/{task}_{policy}_step5000.tar.gz"
            success = np.load(io.BytesIO(read_archive_member(archive, "success_matrix.npy")))
            progress = np.load(io.BytesIO(read_archive_member(archive, "sample_progress.npy")))
            metadata = archive_json(archive, "metadata.json")
            check(success.shape == (expected, 32), f"DMPO {task}/{policy} matrix mismatch")
            check(progress.shape == (32,), f"DMPO {task}/{policy} progress mismatch")
            check(np.all(progress == expected), f"DMPO {task}/{policy} is incomplete")
            fields(
                metadata,
                {
                    "num_examples": expected,
                    "ks": [1, 2, 4, 8, 16, 32],
                    "diffusion_steps": 128,
                    "temperature": 0.2,
                    "sampler": "pd_cache_prefix",
                },
                f"DMPO {task}/{policy}",
            )
            expected_remasking = "dprm_soft_bon" if policy == "dprm" else "low_confidence"
            check(
                metadata.get("remasking") == expected_remasking,
                f"DMPO {task}/{policy} remasking mismatch",
            )
            policies[policy] = {
                "shape": list(success.shape),
                "successes": int(success.sum()),
            }
        result[task] = policies
    checkpoint = root / "dmpo/countdown/dprm_checkpoint_step5000.tar.gz"
    check(checkpoint.is_file() and checkpoint.stat().st_size > 0, "DMPO primary checkpoint missing")
    source_map = json.loads((root / "dmpo/record_sources.json").read_text())
    check(set(source_map["sources"]) == set(task_sizes), "DMPO source-map task mismatch")
    for task in task_sizes:
        for policy in ("confidence", "dprm_confidence"):
            entry = source_map["sources"][task][policy]
            check(entry["checkpoint_binding_verified"] is False, "DMPO archive binding must remain explicit")
    return result


def audit_omni(root: Path) -> dict[str, Any]:
    target = root / "omni_diffusion"
    result = json.loads((target / "online_action_value_release.json").read_text())
    fields(
        result,
        {
            "prompt_count": 512,
            "action_step": 96,
            "confidence_rank_quantiles": "0.70 0.85 0.90 0.95",
            "guidance": 8.0,
            "candidate_paths": 5,
            "terminal_reward": "clip_cosine",
            "independent_check": "clip_b32_cosine",
        },
        "Omni release",
    )
    run_manifest = json.loads(
        (target / "records/confirmation512/run_manifest.json").read_text()
    )
    fields(
        run_manifest,
        {
            "prompt_count": 512,
            "action_step": 96,
            "confidence_rank_quantiles": "0.70 0.85 0.90 0.95",
            "candidate_rollouts_per_prompt": 5,
            "fixed_guidance": 8,
            "complete_candidate_path_selection": True,
            "human_selection": False,
        },
        "Omni confirmation",
    )
    deltas: dict[str, float] = {}
    for metric in ("clip_cosine", "clip_b32_cosine"):
        row = result["paired_deltas"][metric]
        check(float(row["ci95_low"]) > 0.0, f"Omni {metric} interval is not positive")
        deltas[metric] = float(row["mean_delta"])

    records = json.loads(
        (target / "records/confirmation512/records/two_encoder.json").read_text()
    )
    methods = ("confidence", "random", "step96_q0.70", "step96_q0.85", "step96_q0.90", "step96_q0.95")
    check(set(records) == set(methods), "Omni record methods mismatch")
    check(all(len(records[name]) == 512 for name in methods), "Omni record count mismatch")
    branches = methods[2:]
    for prompt_index in range(512):
        base = records["confidence"][prompt_index]
        signatures = []
        defaults = []
        for method in branches:
            row = records[method][prompt_index]
            check(row["prompt_id"] == base["prompt_id"], "Omni prompt pairing mismatch")
            check(row["seed"] == base["seed"], "Omni seed pairing mismatch")
            action = row["counterfactual_override"]["actions"][0]
            check(action["applied"] is True and action["step"] == 96, "Omni action mismatch")
            signatures.append(tuple(action["shared_provisional_visual_token_ids"]))
            defaults.append(action["default_candidate_index"])
        check(len(set(signatures)) == 1, "Omni branches do not share a provisional canvas")
        check(len(set(defaults)) == 1, "Omni branches do not share the native action")

    checkpoint = target / "checkpoint-1000/model.safetensors.index.json"
    check(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Omni checkpoint missing")
    check(len(list((target / "checkpoint-1000").glob("model-*.safetensors"))) == 4, "Omni shard count mismatch")
    mechanism = json.loads((target / "mechanism_cases/manifest.json").read_text())
    check(len(mechanism.get("cases", [])) == 2, "Omni mechanism cases missing")
    return {"selected_step": 1000, "prompts": 512, "deltas": deltas}


def audit_puma(root: Path) -> dict[str, Any]:
    ckpt = load_torch(root / "puma/dprm_confidence/ema_step_2000000.pt", mmap=True)
    training = plain(ckpt["config"])["training"]
    fields(
        training["dprm"],
        {
            "num_bins": 16,
            "reward_beta": 1.0,
            "warmup_steps": 2000,
            "switch_steps": 60000,
            "ready_count": 128,
            "sampled_shortlist": True,
            "min_candidates": 8,
            "max_candidates": 64,
        },
        "PUMA",
    )
    check(ckpt["global_step"] == 2_000_000, "PUMA step mismatch")
    check(training["order_policy"] == "dprm_soft_bon", "PUMA policy mismatch")
    shape = tuple(ckpt["dprm_order_state"]["counts"].shape)
    check(shape == (39, 16, 1), "PUMA table mismatch")
    return {"step": 2_000_000, "table_shape": shape}


def audit_prism(root: Path) -> dict[str, Any]:
    result = {}
    for name in ("confidence", "dprm"):
        path = root / f"prism/gsm8k/{name}_res.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        check(len(rows) == 1319, f"Prism {name} row count mismatch")
        check(all(row.get("svf_calls") == 29 for row in rows), f"Prism {name} SVF mismatch")
        check(all(len(row.get("all_trajectories", [])) == 4 for row in rows), f"Prism {name} survivor mismatch")
        result[name] = {"questions": 1319, "mean_nfe": sum(row["nfe"] for row in rows) / 1319}
    return result


def audit_dplm(root: Path) -> dict[str, Any]:
    rewards = {"dprm_tchebycheff": "aar_structure_tchebycheff"}
    for name, reward in rewards.items():
        ckpt = load_torch(root / f"dplm/{name}/last.ckpt", mmap=True)
        order = plain(ckpt["hyper_parameters"]["model"])["order"]
        check(ckpt["global_step"] == 5000, f"DPLM {name} step mismatch")
        fields(
            order,
            {
                "train_order_policy": "progressive_dprm",
                "decode_order_policy": "dprm_soft_bon",
                "num_phases": 8,
                "confidence_bins": 16,
                "reward": reward,
                "reward_temperature": 8.0,
                "guidance_scale": 1.0,
                "warmup_steps": 500,
                "switch_steps": 2000,
                "ready_count": 128,
                "sampled_soft_bon": True,
                "min_candidates": 8,
                "max_candidates": 32,
            },
            f"DPLM {name}",
        )
        if name == "dprm_tchebycheff":
            fields(
                order,
                {
                    "reward_aa_weight": 0.5,
                    "reward_structure_weight": 0.5,
                    "reward_tchebycheff_temperature": 0.05,
                    "reward_tchebycheff_augmentation": 0.05,
                },
                "DPLM Tchebycheff",
            )
        for stream in ("aa", "struct"):
            key = f"model.{stream}_order_controller.counts"
            check(tuple(ckpt["state_dict"][key].shape) == (8, 16, 1), f"DPLM {stream} table mismatch")
    return {"step": 5000, "rewards": rewards}


def audit_dcm(root: Path) -> dict[str, Any]:
    weights = {"recovery": [0.9, 0.075, 0.025]}
    for name, preference in weights.items():
        record = load_torch(root / f"dcm/controllers/{name}.pt")
        check(bool(record.get("model_state_dict")), f"DCM {name} model is missing")
        reward = record["dprm_reward_config"]
        fields(
            reward,
            {
                "objective_weights": preference,
                "tchebycheff_temperature": 0.05,
                "tchebycheff_augmentation": 0.05,
                "aux_mode": "predicted_zero",
                "model_frozen": True,
            },
            f"DCM {name}",
        )
        state = record["dprm_state_dict"]
        check(tuple(state["counts"].shape) == (4, 16, 2), f"DCM {name} table mismatch")
        check(state["cfg"]["ready_count"] == 64, f"DCM {name} readiness mismatch")
    return {"selected_checkpoint": "recovery", "preferences": weights}


def audit_genmol(root: Path) -> dict[str, Any]:
    weights = {"balanced": [0.55, 0.45]}
    for name, preference in weights.items():
        ckpt = load_torch(root / f"genmol/preferences/{name}_5000.ckpt", mmap=True)
        training = plain(ckpt["hyper_parameters"]["config"])["training"]
        check(ckpt["global_step"] == 5000, f"GenMol {name} step mismatch")
        fields(
            training,
            {
                "order_policy": "dprm_random",
                "dprm_num_phases": 8,
                "dprm_confidence_bins": 16,
                "dprm_aux_mode": "molecular_token_class",
                "dprm_reward_mode": "molecular_tchebycheff",
                "dprm_objective_weights": preference,
                "dprm_reward_temperature": 1.0,
                "dprm_guidance_scale": 1.0,
                "dprm_warmup_steps": 500,
                "dprm_switch_steps": 2000,
                "dprm_ready_count": 128,
                "dprm_sampled_soft_bon": True,
                "dprm_min_candidates": 8,
                "dprm_max_candidates": 64,
            },
            f"GenMol {name}",
        )
        check(tuple(ckpt["dprm_state_dict"]["counts"].shape) == (8, 16, 8), f"GenMol {name} table mismatch")
    return {"step": 5000, "preferences": weights}


def audit_sdpo(root: Path) -> dict[str, Any]:
    for name in ("dprm_random",):
        state = load_torch(root / f"sdpo/{name}/model.pt")
        check(tuple(state["dprm_count"].shape) == (1, 10), f"SDPO {name} phase/bin mismatch")
        check(tuple(state["dprm_sum"].shape) == (1, 10), f"SDPO {name} sum mismatch")
    return {"phase_confidence_shape": [1, 10]}


def audit_llada_v(root: Path) -> dict[str, Any]:
    controller = json.loads((root / "llada_v/controllers/p1_b8_pos4.json").read_text(encoding="utf-8"))
    expected = {
        "num_phases": 1,
        "confidence_bins": 8,
        "aux_bins": 24,
        "aux_mode": "format_eot_position",
        "position_bins": 4,
        "format_bins": 3,
        "reward_temperature": 1.0,
        "warmup_steps": 0,
        "switch_steps": 4,
        "ready_count": 4,
        "sampled_soft_bon": False,
    }
    fields(controller["cfg"], expected, "LLaDA-V")
    shape = (len(controller["counts"]), len(controller["counts"][0]), len(controller["counts"][0][0]))
    check(shape == (1, 8, 24), "LLaDA-V table mismatch")
    fields(
        controller["metadata"],
        {
            "reward_count": 128,
            "used_trace_rows": 512,
            "nonempty_buckets": 54,
            "total_buckets": 192,
            "max_docs_per_task": 128,
        },
        "LLaDA-V fitting split",
    )
    records = root / "llada_v/records/realworldqa_split_records.tar.zst"
    summary = json.loads(read_tar_member(records, "reduction/public_summary.json"))[
        "llada_v"
    ]["realworldqa"]
    fields(
        summary,
        {
            "n": 509,
            "confidence": 0.47347740667976423,
            "dprm_confidence": 0.48919449901768175,
            "wins": 19,
            "losses": 11,
        },
        "LLaDA-V RealWorldQA",
    )
    numeric = summary["by_prompt_format"]["numeric"]
    fields(
        numeric,
        {
            "n": 78,
            "confidence": 0.32051282051282054,
            "dprm_confidence": 0.41025641025641024,
            "wins": 7,
            "losses": 0,
        },
        "LLaDA-V RealWorldQA numeric",
    )
    paired_ids = []
    for member in (
        "confidence/samples_realworldqa.jsonl",
        "dprm/samples_realworldqa.jsonl",
    ):
        ids = {
            int(json.loads(line)["doc_id"])
            for line in read_tar_member(records, member).splitlines()
            if line.strip()
        }
        check(ids == set(range(765)), f"LLaDA-V split record mismatch: {member}")
        paired_ids.append(ids)
    check(paired_ids[0] == paired_ids[1], "LLaDA-V records are not paired")
    diagnostic = root / "llada_v/diagnostics/ai2d_preregistered_confirmation.tar.gz"
    ai2d_diagnostic = None
    if diagnostic.exists():
        frozen = archive_json(diagnostic, "frozen_controller.json")
        selected_table = archive_json(diagnostic, "selected_table.json")
        check(frozen.get("label") == "p1_g8", "LLaDA-V AI2D selected controller mismatch")
        check(frozen.get("guidance") == 8, "LLaDA-V AI2D guidance mismatch")
        fields(
            selected_table["cfg"],
            {
                "num_phases": 1,
                "confidence_bins": 8,
                "aux_mode": "eot_position",
                "position_bins": 4,
                "ready_count": 4,
                "warmup_steps": 0,
                "switch_steps": 4,
            },
            "LLaDA-V AI2D table",
        )
        confirmation = archive_json(diagnostic, "confirmation_audit.json")
        label = confirmation.get("selected")
        selected = confirmation.get("candidates", {}).get(label, {})
        check(selected.get("active_controller") is True, "LLaDA-V AI2D controller inactive")
        check(
            selected.get("positive_point_delta") is False,
            "LLaDA-V AI2D diagnostic unexpectedly changed promotion status",
        )
        check(selected.get("documents") == 244, "LLaDA-V AI2D confirmation size mismatch")
        for name in (
            "confidence_samples.jsonl",
            "dprm_samples.jsonl",
        ):
            rows = archive_jsonl(diagnostic, name)
            check(len(rows) == 500, f"LLaDA-V {name} row count mismatch")
            check(
                {int(row["doc_id"]) for row in rows} == set(range(500)),
                f"LLaDA-V {name} ids mismatch",
            )
        ai2d_diagnostic = {
            "documents": selected["documents"],
            "confidence": selected["baseline_accuracy"],
            "dprm": selected["accuracy"],
            "paired_delta": selected["paired_delta"],
            "promoted": False,
        }
    return {
        "table_shape": shape,
        "config": expected,
        "realworldqa": {
            "documents": summary["n"],
            "confidence": summary["confidence"],
            "dprm": summary["dprm_confidence"],
            "numeric_delta": numeric["paired_delta"],
        },
        "ai2d_diagnostic": ai2d_diagnostic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("reproducibility/release_artifacts.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["hosts"]
    audits = {
        "puma": audit_puma,
        "dmpo": audit_dmpo,
        "prism": audit_prism,
        "dplm": audit_dplm,
        "dcm": audit_dcm,
        "genmol": audit_genmol,
        "sdpo": audit_sdpo,
        "omni_diffusion": audit_omni,
        "llada_v": audit_llada_v,
    }
    result = {
        host: audit(args.artifact_root)
        for host, audit in audits.items()
        if manifest[host]["status"] == "complete"
    }
    result["pending_hosts"] = [host for host, entry in manifest.items() if entry["status"] != "complete"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
