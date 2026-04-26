#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_X_KEYS = ["train/global_step", "global_step", "step", "_step"]
DEFAULT_COLORS = [
    "#E6863B",
    "#D45087",
    "#7A1CAC",
    "#2A2A9B",
    "#3D8B5A",
    "#C33C54",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch reward histories from WandB, stitch resumed runs, and plot paper-style step-reward curves."
    )
    parser.add_argument("--entity", required=True, help="WandB entity/team slug")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series spec in the form label=run_id1,run_id2,... . Repeat for multiple curves.",
    )
    parser.add_argument("--metric", default="train/reward", help="Mean metric key to plot")
    parser.add_argument(
        "--std-metric",
        default="train/reward_std",
        help="Std metric key used for the shaded band. Use empty string to disable the band.",
    )
    parser.add_argument(
        "--x-keys",
        nargs="+",
        default=DEFAULT_X_KEYS,
        help="Candidate x-axis keys to search in each history row",
    )
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--output-dir", required=True, help="Directory for png/csv/pdf outputs")
    parser.add_argument("--min-step", type=float, default=None, help="Optional lower bound for step filtering")
    parser.add_argument("--max-step", type=float, default=None, help="Optional upper bound for step filtering")
    parser.add_argument(
        "--window",
        type=int,
        default=151,
        help="Centered moving-average window in points. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--band-scale",
        type=float,
        default=1.0,
        help="Shaded band width multiplier. 1.0 means mean +/- std.",
    )
    parser.add_argument("--band-alpha", type=float, default=0.18, help="Alpha for the shaded band")
    parser.add_argument("--linewidth", type=float, default=2.4, help="Line width")
    parser.add_argument("--dpi", type=int, default=240, help="PNG dpi")
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF copy")
    parser.add_argument("--legend-loc", default="lower right", help="Legend location")
    return parser.parse_args()


def parse_series_specs(series_specs: list[str]) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    for spec in series_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --series '{spec}'. Expected label=run_id1,run_id2,...")
        label, run_ids_raw = spec.split("=", 1)
        run_ids = [run_id.strip() for run_id in run_ids_raw.split(",") if run_id.strip()]
        if not label or not run_ids:
            raise ValueError(f"Invalid --series '{spec}'. Expected non-empty label and run ids.")
        parsed.append((label.strip(), run_ids))
    return parsed


def to_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def choose_x(row: dict, x_keys: list[str]):
    for key in x_keys:
        value = to_float(row.get(key))
        if value is not None:
            return value, key
    return None, None


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def fetch_series(
    api: wandb.Api,
    entity: str,
    project: str,
    label: str,
    run_ids: list[str],
    metric: str,
    std_metric: str | None,
    x_keys: list[str],
) -> list[tuple[float, float, float | None]]:
    rows_by_step: dict[float, dict[str, float | None]] = {}
    chosen_x_keys: set[str] = set()
    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        found_metric = False
        # Do not pass `keys=` here. WandB can drop rows unless every requested key
        # is present in the same row, which breaks metrics like reward/reward_std.
        for row in run.scan_history(page_size=1000):
            x_value, chosen_x_key = choose_x(row, x_keys)
            y_value = to_float(row.get(metric))
            y_std = to_float(row.get(std_metric)) if std_metric else None

            if x_value is None:
                continue

            chosen_x_keys.add(chosen_x_key)
            slot = rows_by_step.setdefault(x_value, {"mean": None, "std": None})
            if y_value is not None:
                slot["mean"] = y_value
                found_metric = True
            if y_std is not None:
                slot["std"] = y_std

        if not found_metric:
            raise RuntimeError(
                f"No usable rows found for series '{label}' run '{run_id}' with metric '{metric}' "
                f"and x-key candidates {x_keys}."
            )

    stitched = [
        (step, slot["mean"], slot["std"])
        for step, slot in sorted(rows_by_step.items())
        if slot["mean"] is not None
    ]

    print(
        f"[{label}] stitched {len(stitched)} points from runs {run_ids}; "
        f"x-keys used: {sorted(chosen_x_keys)}"
    )
    if stitched:
        print(f"[{label}] step range: {stitched[0][0]} -> {stitched[-1][0]}")
    return stitched


def filter_points(
    points: list[tuple[float, float, float | None]],
    min_step: float | None,
    max_step: float | None,
) -> list[tuple[float, float, float | None]]:
    filtered: list[tuple[float, float, float | None]] = []
    for step, value, value_std in points:
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        filtered.append((step, value, value_std))
    return filtered


def write_csv(
    output_dir: Path,
    label: str,
    steps: np.ndarray,
    mean_raw: np.ndarray,
    std_raw: np.ndarray | None,
    mean_smoothed: np.ndarray,
    std_smoothed: np.ndarray | None,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> None:
    csv_path = output_dir / f"{label}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "reward",
                "reward_std",
                "reward_smoothed",
                "reward_std_smoothed",
                "lower",
                "upper",
            ]
        )
        for idx, step in enumerate(steps):
            writer.writerow(
                [
                    float(step),
                    float(mean_raw[idx]),
                    "" if std_raw is None else float(std_raw[idx]),
                    float(mean_smoothed[idx]),
                    "" if std_smoothed is None else float(std_smoothed[idx]),
                    "" if lower is None else float(lower[idx]),
                    "" if upper is None else float(upper[idx]),
                ]
            )


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.labelsize": 15,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
        }
    )


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be >= 1.")
    if args.band_scale < 0:
        raise ValueError("--band-scale must be >= 0.")

    std_metric = args.std_metric.strip() if args.std_metric else None
    if std_metric == "":
        std_metric = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_specs = parse_series_specs(args.series)
    api = wandb.Api(timeout=60)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    for color, (label, run_ids) in zip(DEFAULT_COLORS, series_specs):
        points = fetch_series(api, args.entity, args.project, label, run_ids, args.metric, std_metric, args.x_keys)
        points = filter_points(points, args.min_step, args.max_step)
        if not points:
            raise RuntimeError(f"Series '{label}' has no points after step filtering.")

        steps = np.array([step for step, _, _ in points], dtype=np.float64)
        mean_raw = np.array([value for _, value, _ in points], dtype=np.float64)
        mean_smoothed = centered_moving_average(mean_raw, args.window)

        std_raw = None
        std_smoothed = None
        lower = None
        upper = None
        if std_metric:
            std_values = [0.0 if value_std is None else value_std for _, _, value_std in points]
            std_raw = np.array(std_values, dtype=np.float64)
            std_smoothed = centered_moving_average(std_raw, args.window)
            lower = mean_smoothed - args.band_scale * std_smoothed
            upper = mean_smoothed + args.band_scale * std_smoothed
            ax.fill_between(steps, lower, upper, color=color, alpha=args.band_alpha, linewidth=0)

        write_csv(output_dir, label, steps, mean_raw, std_raw, mean_smoothed, std_smoothed, lower, upper)
        ax.plot(steps, mean_smoothed, label=label, color=color, linewidth=args.linewidth)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title(args.title or "Reward vs Training Step")
    ax.legend(frameon=False, loc=args.legend_loc)
    fig.tight_layout()

    png_path = output_dir / "step_reward.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {png_path}")

    if args.pdf:
        pdf_path = output_dir / "step_reward.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    main()
