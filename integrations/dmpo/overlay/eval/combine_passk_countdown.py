#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


LEVEL_SPECS = [
    ("overall", "Countdown pass@K: {vanilla_label} vs {progressive_label}", None),
    ("trivial_2", "Trivial (2) Countdown pass@K: {vanilla_label} vs {progressive_label}", 0),
    ("easy_3", "Easy (3) Countdown pass@K: {vanilla_label} vs {progressive_label}", 1),
    ("medium_4", "Medium (4) Countdown pass@K: {vanilla_label} vs {progressive_label}", 2),
    ("hard_5", "Hard (5) Countdown pass@K: {vanilla_label} vs {progressive_label}", 3),
    ("ood_6", "OOD (6) Countdown pass@K: {vanilla_label} vs {progressive_label}", 4),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_state_dir", type=str, default="")
    parser.add_argument("--vanilla_state_dir", type=str, required=True)
    parser.add_argument("--progressive_state_dir", type=str, required=True)
    parser.add_argument("--extra_state_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_label", type=str, default="Base Model")
    parser.add_argument("--vanilla_label", type=str, default="DMPO")
    parser.add_argument("--progressive_label", type=str, default="Progressive DMPO")
    parser.add_argument("--extra_label", type=str, default="DMPO-DPRM")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_state(state_dir):
    with open(os.path.join(state_dir, "metadata.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    success_matrix = np.load(os.path.join(state_dir, "success_matrix.npy"))
    levels = np.load(os.path.join(state_dir, "levels.npy"))
    return metadata, success_matrix, levels


def verify_pair_compatible(meta_a, levels_a, meta_b, levels_b):
    for key in ["ks", "selected_indices", "num_examples", "dataset_jsonl"]:
        if meta_a.get(key) != meta_b.get(key):
            raise RuntimeError(f"Incompatible cached results: mismatch on '{key}'.")
    if levels_a.shape != levels_b.shape or not np.array_equal(levels_a, levels_b):
        raise RuntimeError("Incompatible cached results: level arrays differ.")


def compute_curve(success_matrix, mask, ks):
    subset = success_matrix if mask is None else success_matrix[mask]
    if subset.size == 0:
        raise RuntimeError("Split mask selected zero examples.")
    curve = {}
    for k in ks:
        curve[str(k)] = float(subset[:, :k].any(axis=1).mean())
    return curve


def save_curves_json_csv(ks, curves, meta, output_json_path, output_csv_path, label_order):
    payload = {"ks": ks}
    payload.update(curves)
    payload.update(meta)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write("k," + ",".join(label_order) + "\n")
        for k in ks:
            row = [str(k)]
            for label in label_order:
                row.append(str(payload[label][str(k)]))
            f.write(",".join(row) + "\n")

    return payload


def plot_passk_curves(ks, curves, output_png, title, label_order):
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    x = np.arange(len(ks), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    marker_map = {
        "Base Model": "^",
        "DMPO": "o",
        "Progressive DMPO": "s",
        "DMPO-DPRM": "D",
    }
    color_map = {
        "Base Model": "#2CA02C",
        "DMPO": "#1F77B4",
        "Progressive DMPO": "#D62728",
        "DMPO-DPRM": "#9467BD",
    }

    for label in label_order:
        y = np.array([curves[label][str(k)] for k in ks], dtype=np.float32)
        ax.plot(
            x,
            y,
            marker=marker_map.get(label, "o"),
            linewidth=3.2,
            markersize=9,
            label=label,
            color=color_map.get(label),
        )

    ax.set_xlabel("K", fontsize=22, fontweight="bold")
    ax.set_ylabel("pass@K", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.tick_params(axis="both", which="major", labelsize=18, width=2, length=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    legend = ax.legend(fontsize=18, frameon=True, loc="lower right")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    base_meta = base_success = base_levels = None
    if args.base_state_dir:
        base_meta, base_success, base_levels = load_state(args.base_state_dir)

    vanilla_meta, vanilla_success, vanilla_levels = load_state(args.vanilla_state_dir)
    progressive_meta, progressive_success, progressive_levels = load_state(args.progressive_state_dir)
    extra_meta = extra_success = extra_levels = None
    if args.extra_state_dir:
        extra_meta, extra_success, extra_levels = load_state(args.extra_state_dir)
    verify_pair_compatible(vanilla_meta, vanilla_levels, progressive_meta, progressive_levels)
    if base_meta is not None:
        verify_pair_compatible(base_meta, base_levels, vanilla_meta, vanilla_levels)
    if extra_meta is not None:
        verify_pair_compatible(extra_meta, extra_levels, vanilla_meta, vanilla_levels)

    ks = [int(k) for k in vanilla_meta["ks"]]
    label_order = [args.vanilla_label, args.progressive_label]
    if base_meta is not None:
        label_order = [args.base_label] + label_order
    if extra_meta is not None:
        label_order = label_order + [args.extra_label]

    for split_name, title_template, level in LEVEL_SPECS:
        mask = None if level is None else (vanilla_levels == level)
        curves = {
            args.vanilla_label: compute_curve(vanilla_success, mask, ks),
            args.progressive_label: compute_curve(progressive_success, mask, ks),
        }
        if base_meta is not None:
            curves[args.base_label] = compute_curve(base_success, mask, ks)
        if extra_meta is not None:
            curves[args.extra_label] = compute_curve(extra_success, mask, ks)

        meta = {
            "base_model_path": vanilla_meta["base_model_path"],
            "dataset_jsonl": vanilla_meta.get("dataset_jsonl"),
            "base_checkpoint": None if base_meta is None else (base_meta.get("checkpoint") or None),
            "vanilla_checkpoint": vanilla_meta["checkpoint"],
            "progressive_checkpoint": progressive_meta["checkpoint"],
            "extra_checkpoint": None if extra_meta is None else extra_meta["checkpoint"],
            "sampler": vanilla_meta["sampler"],
            "use_fast_sampler": vanilla_meta["use_fast_sampler"],
            "temperature": vanilla_meta["temperature"],
            "gen_length": vanilla_meta["gen_length"],
            "diffusion_steps": vanilla_meta["diffusion_steps"],
            "seed": vanilla_meta["seed"],
            "num_examples": int(vanilla_success.shape[0] if mask is None else mask.sum()),
            "levels": [0, 1, 2, 3, 4] if level is None else [int(level)],
        }

        prefix = f"passk_countdown_{split_name}_random_vs_progressive"
        json_path = os.path.join(args.output_dir, f"{prefix}.json")
        csv_path = os.path.join(args.output_dir, f"{prefix}.csv")
        png_path = os.path.join(args.output_dir, f"{prefix}.png")

        result = save_curves_json_csv(ks, curves, meta, json_path, csv_path, label_order)
        if base_meta is not None and extra_meta is not None:
            title = title_template.replace(": {vanilla_label} vs {progressive_label}", "")
            title = (
                f"{title}: {args.base_label} vs {args.vanilla_label} vs "
                f"{args.progressive_label} vs {args.extra_label}"
            )
        elif base_meta is not None:
            title = title_template.replace(": {vanilla_label} vs {progressive_label}", "")
            title = f"{title}: {args.base_label} vs {args.vanilla_label} vs {args.progressive_label}"
        elif extra_meta is not None:
            title = title_template.replace(": {vanilla_label} vs {progressive_label}", "")
            title = f"{title}: {args.vanilla_label} vs {args.progressive_label} vs {args.extra_label}"
        else:
            title = title_template.format(
                vanilla_label=args.vanilla_label,
                progressive_label=args.progressive_label,
            )
        plot_passk_curves(ks, curves, png_path, title, label_order)

        print("Saved:")
        print(json_path)
        print(csv_path)
        print(png_path)
        print("Results:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
