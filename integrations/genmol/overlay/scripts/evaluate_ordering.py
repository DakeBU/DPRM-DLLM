# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Additional evaluation harness for ordering ablations. This file keeps the
# original GenMol model and task metrics intact, but records raw outputs and
# bootstrap intervals for method comparisons.

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from time import time

sys.path.append(os.path.realpath("."))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, QED

from genmol.sampler import Sampler

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def smiles_distance(smiles: str, reference_df: pd.DataFrame) -> float:
    if len(reference_df) == 0:
        return np.nan
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    if "MOL" not in reference_df:
        reference_df["MOL"] = reference_df["smiles"].apply(Chem.MolFromSmiles)
    if "FPS" not in reference_df:
        mols = [m for m in reference_df["MOL"].tolist() if m is not None]
        reference_df = reference_df.iloc[: len(mols)].copy()
        reference_df["FPS"] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    fps_list = reference_df["FPS"].tolist()
    if not fps_list:
        return np.nan
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    return float(np.mean(DataStructs.BulkTanimotoSimilarity(fps, fps_list, returnDistance=True)))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def morgan_diversity(smiles_list: list[str]) -> float:
    unique_smiles = sorted(set([s for s in smiles_list if s]))
    if len(unique_smiles) <= 1:
        return 0.0
    fps = []
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False))
    if len(fps) <= 1:
        return 0.0
    distances = []
    for idx, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[idx + 1 :])
        distances.extend([1.0 - sim for sim in sims])
    return float(np.mean(distances)) if distances else 0.0


def score_samples(samples: list[str]) -> tuple[list[float], list[float]]:
    qeds, sas = [], []
    for smi in samples:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            qeds.append(np.nan)
            sas.append(np.inf)
        else:
            qeds.append(float(QED.qed(mol)))
            sas.append(float(sascorer.calculateScore(mol)))
    return qeds, sas


def summarize_denovo(samples: list[str], num_attempts: int) -> tuple[dict, pd.DataFrame]:
    qeds, sas = score_samples(samples)
    raw = pd.DataFrame({"smiles": samples, "qed": qeds, "sa": sas})
    raw["is_quality"] = (raw["qed"] >= 0.6) & (raw["sa"] <= 4)
    unique = raw.drop_duplicates("smiles") if len(raw) else raw
    diversity = morgan_diversity(unique["smiles"].tolist()) if len(unique) > 1 else 0.0
    summary = {
        "validity": safe_div(len(raw), num_attempts),
        "uniqueness": safe_div(len(unique), len(raw)),
        "quality": safe_div(int(raw["is_quality"].sum()) if len(raw) else 0, num_attempts),
        "diversity": diversity,
        "num_attempts": int(num_attempts),
        "num_valid": int(len(raw)),
        "num_unique": int(len(unique)),
    }
    raw.insert(0, "attempt_rank", np.arange(len(raw)))
    return summary, raw


def bootstrap_denovo(raw: pd.DataFrame, num_attempts: int, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    smiles = raw["smiles"].to_numpy() if len(raw) else np.array([])
    is_quality = raw["is_quality"].to_numpy(dtype=bool) if len(raw) else np.array([], dtype=bool)
    n_valid = len(raw)
    # A resampled set necessarily contains duplicate draws even when the original
    # generated set is entirely unique. Recomputing set cardinality inside each
    # bootstrap sample therefore biases uniqueness toward 1-exp(-1). Treat the
    # original first-occurrence events as the per-sample novelty observations;
    # their mean is exactly num_unique / num_valid.
    novelty = (~pd.Series(smiles).duplicated()).to_numpy(dtype=bool) if n_valid else np.array([], dtype=bool)
    unique_smiles = sorted(set(smiles.tolist())) if n_valid else []
    unique_index = {s: i for i, s in enumerate(unique_smiles)}
    raw_unique_ids = np.array([unique_index[s] for s in smiles], dtype=int) if n_valid else np.array([], dtype=int)
    distance_matrix = None
    if unique_smiles:
        fps = []
        for smi in unique_smiles:
            mol = Chem.MolFromSmiles(smi)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False))
        distance_matrix = np.zeros((len(fps), len(fps)), dtype=np.float32)
        for i, fp in enumerate(fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
            distance_matrix[i, :] = 1.0 - np.asarray(sims, dtype=np.float32)

    def boot_diversity(unique_ids: np.ndarray) -> float:
        if distance_matrix is None or len(unique_ids) <= 1:
            return 0.0
        sub = distance_matrix[np.ix_(unique_ids, unique_ids)]
        tri = sub[np.triu_indices_from(sub, k=1)]
        return float(np.mean(tri)) if len(tri) else 0.0

    for _ in range(n_boot):
        valid_count = rng.binomial(num_attempts, safe_div(n_valid, num_attempts))
        if n_valid and valid_count:
            idx = rng.integers(0, n_valid, size=valid_count)
            uniqueness = float(novelty[idx].mean())
            quality_count = int(is_quality[idx].sum())
            diversity = boot_diversity(np.unique(raw_unique_ids[idx]))
        else:
            uniqueness = 0.0
            quality_count = 0
            diversity = 0.0
        records.append(
            {
                "validity": safe_div(valid_count, num_attempts),
                "uniqueness": uniqueness,
                "quality": safe_div(quality_count, num_attempts),
                "diversity": diversity,
            }
        )
    result = summarize_bootstrap(pd.DataFrame(records))
    unique_count = len(unique_smiles)
    point_values = {
        "validity": safe_div(n_valid, num_attempts),
        "uniqueness": safe_div(unique_count, n_valid),
        "quality": safe_div(int(is_quality.sum()), num_attempts),
        "diversity": morgan_diversity(unique_smiles) if unique_count > 1 else 0.0,
    }
    for metric, point in point_values.items():
        result[metric]["mean"] = float(point)
    return result


def summarize_bootstrap(samples: pd.DataFrame) -> dict:
    out = {}
    for col in samples.columns:
        values = samples[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            out[col] = {"mean": np.nan, "lo": np.nan, "hi": np.nan}
        else:
            out[col] = {
                "mean": float(np.mean(values)),
                "lo": float(np.percentile(values, 2.5)),
                "hi": float(np.percentile(values, 97.5)),
            }
    return out


def run_denovo(sampler: Sampler, config: dict, out_dir: Path, seed: int, n_boot: int) -> dict:
    print(f"[eval] denovo start: num_samples={config['num_samples']} out_dir={out_dir}", flush=True)
    start = time()
    samples = sampler.de_novo_generation(
        config["num_samples"],
        softmax_temp=config["softmax_temp"],
        randomness=config["randomness"],
        min_add_len=config["min_add_len"],
    )
    elapsed = time() - start
    summary, raw = summarize_denovo(samples, config["num_samples"])
    summary["time_sec"] = float(elapsed)
    raw.to_csv(out_dir / "denovo_raw.csv", index=False)
    boot = bootstrap_denovo(raw, config["num_samples"], n_boot, seed + 17)
    print(
        f"[eval] denovo done: valid={len(raw)}/{config['num_samples']} "
        f"unique={raw['smiles'].nunique() if len(raw) else 0} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {"summary": summary, "bootstrap": boot}


def task_sampler(sampler: Sampler, task_key: str, task_config: dict):
    if task_key == "linker_design":
        return lambda fragment, num_samples: sampler.fragment_linking(fragment, num_samples, **task_config["params"])
    if task_key == "linker_design_onestep":
        return lambda fragment, num_samples: sampler.fragment_linking_onestep(fragment, num_samples, **task_config["params"])
    return lambda fragment, num_samples: sampler.fragment_completion(fragment, num_samples, **task_config["params"])


def summarize_fragment_task(task: str, rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    metrics = ["validity", "uniqueness", "quality", "diversity", "distance"]
    return {metric: float(df[metric].mean()) for metric in metrics}


def bootstrap_fragment_rows(raw: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    metrics = ["validity", "uniqueness", "quality", "diversity", "distance"]
    out = {}
    for task, group in raw.groupby("task"):
        values = group[metrics].to_numpy(dtype=float)
        records = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(group), size=len(group))
            records.append(np.nanmean(values[idx], axis=0))
        out[task] = summarize_bootstrap(pd.DataFrame(records, columns=metrics))
    return out


def run_fragment(sampler: Sampler, config: dict, out_dir: Path, seed: int, n_boot: int) -> dict:
    print(
        f"[eval] fragment start: num_samples={config['num_samples']} "
        f"max_examples={config.get('max_examples')} out_dir={out_dir}",
        flush=True,
    )
    data = pd.read_csv(ROOT / "data/fragments.csv")
    tasks = [
        ("linker_design_onestep", "linker_design", config["linker_design_onestep"]),
        ("linker_design", "linker_design", config["linker_design"]),
        ("motif_extension", "motif_extension", config["motif_extension"]),
        ("scaffold_decoration", "scaffold_decoration", config["scaffold_decoration"]),
        ("superstructure_generation", "superstructure_generation", config["superstructure_generation"]),
    ]

    all_rows = []
    all_samples = []
    start = time()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        data = data.iloc[: int(max_examples)].copy()
    skip_examples = set(config.get("skip_examples", []))
    if skip_examples:
        data = data.drop(index=[idx for idx in skip_examples if idx in data.index]).copy()

    for task_name, fragment_column, params in tasks:
        print(f"[eval] fragment task start: {task_name} examples={len(data)}", flush=True)
        fn = task_sampler(sampler, task_name, {"params": params})
        for example_idx, row in data.reset_index(drop=True).iterrows():
            original = row["smiles"]
            fragment = row[fragment_column]
            print(
                f"[eval] fragment example: task={task_name} local_idx={example_idx} "
                f"name={row.get('name', str(example_idx))}",
                flush=True,
            )
            samples = []
            remaining = int(config["num_samples"])
            chunk_size = int(config.get("chunk_size") or remaining)
            chunk_size = max(1, min(chunk_size, remaining))
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                chunk_idx = (int(config["num_samples"]) - remaining) // chunk_size
                print(
                    f"[eval] fragment chunk: task={task_name} local_idx={example_idx} "
                    f"chunk={chunk_idx} attempts={this_chunk}/{config['num_samples']}",
                    flush=True,
                )
                samples.extend(fn(fragment, this_chunk))
                remaining -= this_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            qeds, sas = score_samples(samples)
            df = pd.DataFrame({"smiles": samples, "qed": qeds, "sa": sas})
            df["is_quality"] = (df["qed"] >= 0.6) & (df["sa"] <= 4)
            unique = df.drop_duplicates("smiles") if len(df) else df
            diversity = morgan_diversity(unique["smiles"].tolist()) if len(unique) > 1 else 0.0
            distance = smiles_distance(original, unique.copy()) if len(unique) else np.nan
            metric_row = {
                "task": task_name,
                "example_idx": int(example_idx),
                "name": row.get("name", str(example_idx)),
                "num_attempts": int(config["num_samples"]),
                "num_valid": int(len(df)),
                "num_unique": int(len(unique)),
                "validity": safe_div(len(df), config["num_samples"]),
                "uniqueness": safe_div(len(unique), len(df)),
                "quality": safe_div(int(df["is_quality"].sum()) if len(df) else 0, config["num_samples"]),
                "diversity": diversity,
                "distance": distance,
            }
            all_rows.append(metric_row)
            for sample_idx, sample in enumerate(samples):
                all_samples.append(
                    {
                        "task": task_name,
                        "example_idx": int(example_idx),
                        "sample_idx": int(sample_idx),
                        "smiles": sample,
                        "qed": qeds[sample_idx],
                        "sa": sas[sample_idx],
                    }
                )
            pd.DataFrame(all_rows).to_csv(out_dir / "fragment_rows.partial.csv", index=False)
            pd.DataFrame(all_samples).to_csv(out_dir / "fragment_samples.partial.csv", index=False)
        print(f"[eval] fragment task done: {task_name}", flush=True)
    elapsed = time() - start
    raw = pd.DataFrame(all_rows)
    samples_df = pd.DataFrame(all_samples)
    raw.to_csv(out_dir / "fragment_rows.csv", index=False)
    samples_df.to_csv(out_dir / "fragment_samples.csv", index=False)
    summaries = {task: summarize_fragment_task(task, rows.to_dict("records")) for task, rows in raw.groupby("task")}
    print(f"[eval] fragment done: rows={len(raw)} samples={len(samples_df)} elapsed={elapsed:.1f}s", flush=True)
    return {"summary": summaries, "bootstrap": bootstrap_fragment_rows(raw, n_boot, seed + 31), "time_sec": float(elapsed)}


def write_summary(path: Path, payload: dict) -> None:
    if path.suffix != ".json":
        path = path / "summary.json"
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def extract_controller_diagnostics(checkpoint_path: str, out_dir: Path) -> dict:
    """Export the learned bucket table without constructing the model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("dprm_state_dict")
    if not state:
        return {"available": False}

    cfg = dict(state.get("cfg", {}))
    counts = torch.as_tensor(state["counts"], dtype=torch.float64)
    exp_sums = torch.as_tensor(state["exp_reward_sums"], dtype=torch.float64)
    beta = float(cfg.get("reward_temperature", 1.0))
    ready_count = int(cfg.get("ready_count", 1))
    nonzero = counts > 0
    ready = counts >= ready_count
    safe_mean = torch.where(nonzero, exp_sums / counts.clamp_min(1.0), torch.ones_like(exp_sums))
    values = torch.log(safe_mean.clamp_min(1e-12)) / max(beta, 1e-12)

    rows = []
    for phase in range(counts.shape[0]):
        for conf_bin in range(counts.shape[1]):
            for aux_bin in range(counts.shape[2]):
                rows.append(
                    {
                        "phase": phase,
                        "confidence_bin": conf_bin,
                        "aux_bin": aux_bin,
                        "count": float(counts[phase, conf_bin, aux_bin]),
                        "ready": bool(ready[phase, conf_bin, aux_bin]),
                        "log_moment_reward": float(values[phase, conf_bin, aux_bin]),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "dprm_bucket_table.csv", index=False)

    active_values = values[nonzero]
    phase_counts = counts.sum(dim=(1, 2))
    total_count = float(counts.sum())
    phase_mass = phase_counts / max(total_count, 1.0)
    return {
        "available": True,
        "global_step": int(checkpoint.get("global_step", -1)),
        "num_buckets": int(counts.numel()),
        "nonzero_buckets": int(nonzero.sum()),
        "ready_buckets": int(ready.sum()),
        "nonzero_bucket_rate": float(nonzero.double().mean()),
        "ready_bucket_rate": float(ready.double().mean()),
        "total_selected_events": total_count,
        "count_min_nonzero": float(counts[nonzero].min()) if nonzero.any() else 0.0,
        "count_median_nonzero": float(counts[nonzero].median()) if nonzero.any() else 0.0,
        "reward_min_active": float(active_values.min()) if active_values.numel() else 0.0,
        "reward_mean_active": float(active_values.mean()) if active_values.numel() else 0.0,
        "reward_max_active": float(active_values.max()) if active_values.numel() else 0.0,
        "phase_selected_mass": [float(value) for value in phase_mass],
        "config": cfg,
    }


def plot_method_summary(comparison_path: Path) -> None:
    with comparison_path.open("r") as handle:
        data = json.load(handle)
    out_dir = comparison_path.parent

    denovo_metrics = ["validity", "uniqueness", "quality", "diversity"]
    rows = []
    for method, payload in data["methods"].items():
        if "denovo" not in payload:
            continue
        for metric in denovo_metrics:
            stat = payload["denovo"]["bootstrap"][metric]
            rows.append({"method": method, "metric": metric, **stat})
    if rows:
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, len(denovo_metrics), figsize=(4 * len(denovo_metrics), 4), constrained_layout=True)
        for ax, metric in zip(axes, denovo_metrics):
            sub = df[df["metric"] == metric]
            x = np.arange(len(sub))
            y = sub["mean"].to_numpy()
            yerr = np.vstack([y - sub["lo"].to_numpy(), sub["hi"].to_numpy() - y])
            yerr = np.nan_to_num(np.maximum(yerr, 0.0), nan=0.0)
            ax.bar(x, y, yerr=yerr, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["method"], rotation=35, ha="right")
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.3)
        fig.savefig(out_dir / "denovo_bootstrap_metrics.png", dpi=220)
        plt.close(fig)

    frag_metrics = ["validity", "quality", "diversity", "distance"]
    frag_rows = []
    for method, payload in data["methods"].items():
        if "fragment" not in payload:
            continue
        for task, task_payload in payload["fragment"]["bootstrap"].items():
            for metric in frag_metrics:
                stat = task_payload[metric]
                frag_rows.append({"method": method, "task": task, "metric": metric, **stat})
    if frag_rows:
        df = pd.DataFrame(frag_rows)
        for metric in frag_metrics:
            subm = df[df["metric"] == metric]
            tasks = sorted(subm["task"].unique())
            methods = list(data["methods"].keys())
            fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tasks) * len(methods)), 4.8), constrained_layout=True)
            width = 0.8 / max(len(methods), 1)
            base = np.arange(len(tasks))
            for j, method in enumerate(methods):
                sub = subm[subm["method"] == method].set_index("task").reindex(tasks)
                y = sub["mean"].to_numpy()
                yerr = np.vstack([y - sub["lo"].to_numpy(), sub["hi"].to_numpy() - y])
                yerr = np.nan_to_num(np.maximum(yerr, 0.0), nan=0.0)
                ax.bar(base + (j - (len(methods) - 1) / 2) * width, y, width=width, yerr=yerr, capsize=3, label=method)
            ax.set_xticks(base)
            ax.set_xticklabels(tasks, rotation=25, ha="right")
            ax.set_title(f"fragment {metric}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            fig.savefig(out_dir / f"fragment_{metric}_bootstrap.png", dpi=220)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="METHOD=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["denovo", "fragment"], choices=["denovo", "fragment"])
    parser.add_argument("--denovo-config", default=str(ROOT / "scripts/exps/denovo/hparams_v2.yaml"))
    parser.add_argument("--fragment-config", default=str(ROOT / "scripts/exps/frag/hparams_v2.yaml"))
    parser.add_argument("--denovo-num", type=int, default=None)
    parser.add_argument("--fragment-num", type=int, default=None)
    parser.add_argument(
        "--fragment-chunk-size",
        type=int,
        default=None,
        help="Generate each fragment example in chunks while preserving total attempts.",
    )
    parser.add_argument("--fragment-max-examples", type=int, default=None)
    parser.add_argument(
        "--fragment-skip-examples",
        nargs="*",
        type=int,
        default=[],
        help="Original row indices in data/fragments.csv to skip for every fragment task.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument(
        "--dprm-guidance-scale",
        type=float,
        default=None,
        help="Decode-only override for a loaded checkpoint's DPRM guidance scale.",
    )
    parser.add_argument(
        "--dprm-ready-count",
        type=int,
        default=None,
        help="Decode-only override for a loaded checkpoint's DPRM readiness threshold.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denovo_config = load_yaml(Path(args.denovo_config))
    fragment_config = load_yaml(Path(args.fragment_config))
    if args.denovo_num is not None:
        denovo_config["num_samples"] = args.denovo_num
    if args.fragment_num is not None:
        fragment_config["num_samples"] = args.fragment_num
    if args.fragment_chunk_size is not None:
        fragment_config["chunk_size"] = args.fragment_chunk_size
    if args.fragment_max_examples is not None:
        fragment_config["max_examples"] = args.fragment_max_examples
    if args.fragment_skip_examples:
        fragment_config["skip_examples"] = args.fragment_skip_examples
    methods = {}

    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError(f"checkpoint spec must be METHOD=PATH, got {spec}")
        method, ckpt = spec.split("=", 1)
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        summary_path = method_dir / "summary.json"
        if summary_path.exists():
            print(f"[eval] method skip: {method} existing summary={summary_path}", flush=True)
            with summary_path.open("r") as f:
                methods[method] = json.load(f)
            continue

        print(f"[eval] method start: {method} checkpoint={ckpt}", flush=True)
        sampler = Sampler(ckpt)
        controller = getattr(sampler.model, "dprm_controller", None)
        if controller is not None and args.dprm_guidance_scale is not None:
            controller.cfg.guidance_scale = float(args.dprm_guidance_scale)
        if controller is not None and args.dprm_ready_count is not None:
            controller.cfg.ready_count = int(args.dprm_ready_count)
        controller_diagnostics = extract_controller_diagnostics(ckpt, method_dir)
        if controller is not None:
            controller_diagnostics["decode_effective_config"] = dict(vars(controller.cfg))
        payload = {
            "checkpoint": ckpt,
            "controller_diagnostics": controller_diagnostics,
            "decode_overrides": {
                "dprm_guidance_scale": args.dprm_guidance_scale,
                "dprm_ready_count": args.dprm_ready_count,
            },
        }
        if "denovo" in args.tasks:
            payload["denovo"] = run_denovo(sampler, denovo_config, method_dir, args.seed, args.bootstrap)
        if "fragment" in args.tasks:
            payload["fragment"] = run_fragment(sampler, fragment_config, method_dir, args.seed, args.bootstrap)
        payload["decode_ordering_diagnostics"] = sampler.ordering_diagnostics()
        write_summary(method_dir, payload)
        print(f"[eval] method summary written: {method_dir / 'summary.json'}", flush=True)
        methods[method] = payload
        del sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = {"methods": methods, "bootstrap_resamples": args.bootstrap, "seed": args.seed}
    comparison_path = out_dir / "comparison_summary.json"
    write_summary(comparison_path, comparison)
    plot_method_summary(comparison_path)


if __name__ == "__main__":
    main()
