#!/usr/bin/env python3
"""Reduce matched DMPO success matrices into release and LaTeX summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TASKS = {
    "gsm8k": {"label": "GSM8K", "hard_label": "GSM8K Hard", "hard_level": 3},
    "math": {"label": "MATH", "hard_label": "MATH Hard", "hard_level": 4},
    "countdown": {
        "label": "Countdown",
        "hard_label": "Countdown Hard",
        "hard_level": 3,
    },
}
KS = (1, 2, 4, 8, 16, 32)


def per_example_mean_passk(success: np.ndarray) -> np.ndarray:
    if success.ndim != 2 or success.shape[1] < max(KS):
        raise ValueError(f"invalid success matrix shape: {success.shape}")
    success = success.astype(bool, copy=False)
    return np.stack([success[:, :k].any(axis=1) for k in KS], axis=1).mean(axis=1)


def paired_interval(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    if baseline.shape != method.shape:
        raise ValueError("paired statistics require equal shapes")
    delta = method - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 250):
        width = min(250, iterations - start)
        indices = rng.integers(0, delta.size, size=(width, delta.size))
        samples[start : start + width] = delta[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


def load_policy(
    root: Path,
    task: str,
    policy: str,
    source_map: dict[str, dict[str, dict[str, object]]] | None,
    artifact_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if source_map is None:
        directory = root / "evaluations" / task / f"{policy}_step5000"
    else:
        directory = Path(str(source_map[task][policy]["path"]))
        if not directory.is_absolute():
            if artifact_root is None:
                raise ValueError("relative record paths require --artifact-root")
            directory = artifact_root / directory
    success = np.load(directory / "success_matrix.npy")
    levels = np.load(directory / "levels.npy")
    if success.shape[0] != levels.shape[0]:
        raise ValueError(f"{task}/{policy}: levels and success rows differ")
    return success, levels


def fmt(value: float) -> str:
    return f"{100.0 * value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--paper-results-csv", type=Path)
    parser.add_argument("--experiment-registry", type=Path)
    parser.add_argument(
        "--record-source-map",
        type=Path,
        help="optional package_release.py-compatible map of retained record roots",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="root used to resolve relative paths in --record-source-map",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    source_map = None
    if args.record_source_map:
        source_payload = json.loads(args.record_source_map.read_text(encoding="utf-8"))
        if source_payload.get("schema_version") != 1:
            raise ValueError("record source map must use schema_version 1")
        source_map = source_payload.get("sources")
        if not isinstance(source_map, dict):
            raise ValueError("record source map must contain an object named sources")
    selected_tasks = [
        task
        for task in TASKS
        if source_map is None
        or all(policy in source_map.get(task, {}) for policy in ("confidence", "dprm_confidence"))
    ]
    if not selected_tasks:
        raise ValueError("no paired DMPO tasks are available")

    payload: dict[str, object] = {
        "endpoint_step": 5000,
        "ks": list(KS),
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_seed": args.seed,
        "tasks": {},
    }
    csv_rows: list[dict[str, object]] = []
    row_values: dict[str, list[str]] = {"confidence": [], "dprm_confidence": []}

    for task_index, task in enumerate(selected_tasks):
        spec = TASKS[task]
        confidence, levels = load_policy(
            args.repro_root, task, "confidence", source_map, args.artifact_root
        )
        dprm, dprm_levels = load_policy(
            args.repro_root, task, "dprm_confidence", source_map, args.artifact_root
        )
        if not np.array_equal(levels, dprm_levels):
            raise ValueError(f"{task}: confidence and DPRM level arrays differ")
        conf_stat = per_example_mean_passk(confidence)
        dprm_stat = per_example_mean_passk(dprm)

        subsets = {
            "all": np.ones(levels.shape[0], dtype=bool),
            "hard": levels == int(spec["hard_level"]),
        }
        task_payload: dict[str, object] = {}
        for subset_index, (subset, mask) in enumerate(subsets.items()):
            if not mask.any():
                raise ValueError(
                    f"{task}: no examples with hard level {spec['hard_level']}"
                )
            delta, low, high = paired_interval(
                conf_stat[mask],
                dprm_stat[mask],
                iterations=args.bootstrap_iters,
                seed=args.seed + 10 * task_index + subset_index,
            )
            summary = {
                "n": int(mask.sum()),
                "confidence": float(conf_stat[mask].mean()),
                "dprm": float(dprm_stat[mask].mean()),
                "paired_delta": delta,
                "ci95_low": low,
                "ci95_high": high,
            }
            task_payload[subset] = summary
            task_label = spec["label"] if subset == "all" else spec["hard_label"]
            for method, variant, value in (
                ("Progressive DMPO", "confidence", summary["confidence"]),
                ("DMPO-DPRM", "dprm_confidence", summary["dprm"]),
            ):
                csv_rows.append(
                    {
                        "host": "DMPO",
                        "task": task_label,
                        "method": method,
                        "variant": variant,
                        "metric": "mean_pass_at_k",
                        "direction": "higher",
                        "value": f"{100.0 * float(value):.6f}",
                        "ci95_low": "",
                        "ci95_high": "",
                        "n": int(mask.sum()),
                        "protocol": "matched_step5000",
                    }
                )
            csv_rows.append(
                {
                    "host": "DMPO",
                    "task": task_label,
                    "method": "DMPO-DPRM",
                    "variant": "dprm_confidence",
                    "metric": "paired_delta",
                    "direction": "higher",
                    "value": f"{100.0 * delta:.6f}",
                    "ci95_low": f"{100.0 * low:.6f}",
                    "ci95_high": f"{100.0 * high:.6f}",
                    "n": int(mask.sum()),
                    "protocol": "matched_step5000",
                }
            )
        payload["tasks"][task] = task_payload
        if source_map is not None:
            task_payload["record_provenance"] = {
                policy: source_map[task][policy] for policy in ("confidence", "dprm_confidence")
            }
        row_values["confidence"].extend(
            [fmt(task_payload["all"]["confidence"]), fmt(task_payload["hard"]["confidence"])]
        )
        row_values["dprm_confidence"].extend(
            [fmt(task_payload["all"]["dprm"]), fmt(task_payload["hard"]["dprm"])]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.latex_output:
        dprm_cells = " & ".join(
            "\\bestcell{" + value + "}" for value in row_values["dprm_confidence"]
        )
        latex = (
            "\\newcommand{\\dmpomatchedrows}{%\n"
            f"Progressive DMPO & {' & '.join(row_values['confidence'])} \\\\\n"
            f"DMPO-DPRM & {dprm_cells} \\\\%\n"
            "}\n"
        )
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(latex, encoding="utf-8")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "host",
            "task",
            "method",
            "variant",
            "metric",
            "direction",
            "value",
            "ci95_low",
            "ci95_high",
            "n",
            "protocol",
        ]
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    if args.paper_results_csv:
        with args.paper_results_csv.open(newline="", encoding="utf-8") as handle:
            retained = [
                row for row in csv.DictReader(handle) if row.get("host") != "DMPO"
            ]
        temporary = args.paper_results_csv.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(retained + csv_rows)
        temporary.replace(args.paper_results_csv)

    if args.experiment_registry:
        registry = json.loads(args.experiment_registry.read_text(encoding="utf-8"))
        experiment = next(
            item for item in registry["experiments"] if item["id"] == "dmpo"
        )
        for variant in experiment["variants"]:
            if variant["id"] in {"confidence", "dprm_confidence"}:
                variant["status"] = "reported"
        temporary = args.experiment_registry.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.experiment_registry)


if __name__ == "__main__":
    main()
