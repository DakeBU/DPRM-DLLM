#!/usr/bin/env python3
"""Render the multi-case PUMA interpretation gallery for the project page."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from analyze_reveal_order import order_metrics


INK = "#17191c"
MUTED = "#5c636c"
LINE = "#cbd2d9"
BLUE = "#3b6fa8"
GREEN = "#238b57"
RED = "#c83e36"
ORANGE = "#d56a0c"
SOFT = "#f5f7f8"
BLUE_BG = "#f2f6fb"
GREEN_BG = "#f2f8f4"


CASE_SPECS = {
    159: {
        "title": "Pond visibility",
        "prompt": "11 visible; 6 appear; 2 hide. Gold: 15.",
        "insight": "Broader context preserves the opposite signs of the two events.",
        "confidence": [
            "tadpoles = 11",
            "tadpoles = tadpoles - 6",
            "tadpoles = tadpoles - 2",
            "result = tadpoles",
        ],
        "dprm": [
            "tadpoles = 11",
            "tadpoles += 6",
            "tadpoles -= 2",
            "result = tadpoles",
        ],
    },
    284: {
        "title": "Two-week story total",
        "prompt": "First-week total 120; second week doubles it. Gold: 360.",
        "insight": "Backfill keeps the two week totals distinct instead of doubling twice.",
        "confidence": [
            "total_stories = 20 + 40 + 60",
            "total_stories *= 2 * 2",
            "result = total_stories",
        ],
        "dprm": [
            "stories_first_week = 20 + 40 + 60",
            "stories_second_week = stories_first_week * 2",
            "result = stories_first_week + stories_second_week",
        ],
    },
    1209: {
        "title": "Dehumidifier rates",
        "prompt": "Rates are 1, 2x low, and 2x medium. Gold: 29 liters.",
        "insight": "The widened trajectory resolves the rate hierarchy before aggregation.",
        "confidence": [
            "low_setting_rate = 1",
            "medium_setting_rate = 20 * 2",
            "high_setting_rate = 20 * 2",
            "total_water = low_setting_water +",
            "    medium_setting_water + high_setting_water",
        ],
        "dprm": [
            "low_setting = 1",
            "medium_setting = 2 * low_setting",
            "high_setting = 2 * medium_setting",
            "total_water = (low_setting * 3) +",
            "    (medium_setting * 3) + (high_setting * 5)",
        ],
    },
    246: {
        "title": "Movie-makeup discount",
        "prompt": "250 x 6 x 4 x 5, then 10% off. Gold: 27000.",
        "insight": "Later numeric commitment retains the terminal discount operation.",
        "confidence": [
            "hourly_rate = 250",
            "hours_per_day = 6",
            "days_per_week = 4",
            "weeks_to_finish = 5",
            "total_hours = hours_per_day * days_per_week * weeks_to_finish",
            "total_cost = total_hours * hourly_rate",
            "result = total_cost",
        ],
        "dprm": [
            "total_cost = 250 * 6 * 4 * 5 * 0.9",
            "result = total_cost",
        ],
    },
    908: {
        "title": "Marbles remaining",
        "prompt": "Start with 30; give away one fifth and then 10. Gold: 14.",
        "insight": "The reveal path conditions the fraction on the original total.",
        "confidence": [
            "total = 30",
            "total = total / 5",
            "result = total - 10",
        ],
        "dprm": [
            "marbles = 30",
            "marbles = marbles - marbles / 5 - 10",
            "result = marbles",
        ],
    },
    50: {
        "title": "Egg revenue per week",
        "prompt": "252 eggs/day at 2 dollars per dozen for 7 days. Gold: 294.",
        "insight": "Delayed commitment retains the dozen conversion before pricing.",
        "confidence": [
            "eggs_per_day = 252",
            "price_per_dozen = 2",
            "days_per_week = 7",
            "eggs_per_week = eggs_per_day * days_per_week",
            "revenue = eggs_per_week * price_per_dozen",
            "result = revenue",
        ],
        "dprm": [
            "eggs_per_day = 252",
            "price_per_dozen = 2",
            "days_per_week = 7",
            "eggs_per_week = eggs_per_day * days_per_week",
            "dozens_per_week = eggs_per_week / 12",
            "result = price_per_dozen * dozens_per_week",
        ],
    },
}

PAGES = ((159, 284, 1209), (246, 908, 50))


def load_cases(trace_dir: Path) -> dict[int, dict[str, dict]]:
    selected = set(CASE_SPECS)
    records: dict[int, dict[str, dict]] = {index: {} for index in selected}
    for path in sorted(trace_dir.glob("puma_*trace*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                index = int(row["index"])
                if index not in selected:
                    continue
                method = str(row.get("confidence"))
                if method in {"top_k", "dprm_soft_bon"}:
                    records[index][method] = row
    missing = {
        index: sorted({"top_k", "dprm_soft_bon"} - set(rows))
        for index, rows in records.items()
        if set(rows) != {"top_k", "dprm_soft_bon"}
    }
    if missing:
        raise ValueError(f"missing paired traces: {missing}")
    return records


def rounded_box(ax, x: float, y: float, width: float, height: float, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.009,rounding_size=0.012",
            transform=ax.transAxes,
            linewidth=1.3,
            edgecolor=LINE,
            facecolor=color,
        )
    )


def result_text(row: dict) -> str:
    value = row.get("predicted_value")
    return "no answer" if value is None else str(value)


def draw_code_panel(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    result: str,
    lines: list[str],
    accent: str,
    background: str,
    correct: bool,
) -> None:
    rounded_box(ax, x, y, width, height, background)
    ax.text(x + 0.018, y + height - 0.030, label, transform=ax.transAxes, fontsize=13.5, color=accent, va="top")
    ax.text(
        x + width - 0.018,
        y + height - 0.030,
        f"{result}  {'CORRECT' if correct else 'WRONG'}",
        transform=ax.transAxes,
        fontsize=12.5,
        color=GREEN if correct else RED,
        ha="right",
        va="top",
    )
    line_y = y + height - 0.067
    line_gap = min(0.022, (height - 0.079) / max(len(lines) - 1, 1))
    base_size = 9.2 if len(lines) > 5 else 10.8
    for line in lines:
        ax.text(
            x + 0.025,
            line_y,
            line,
            transform=ax.transAxes,
            fontsize=min(base_size, 9.0 if len(line) > 75 else (10.2 if len(line) > 53 else 10.8)),
            fontfamily="DejaVu Sans Mono",
            fontweight="bold" if correct else "normal",
            color=accent if correct else INK,
            va="top",
        )
        line_y -= line_gap


def draw_case(ax, index: int, rows: dict[str, dict], y: float, height: float) -> None:
    confidence = rows["top_k"]
    dprm = rows["dprm_soft_bon"]
    if bool(confidence.get("correct")) or not bool(dprm.get("correct")):
        raise ValueError(f"case {index} is not a DPRM-only win")

    spec = CASE_SPECS[index]
    for method, excerpt_key in (("top_k", "confidence"), ("dprm_soft_bon", "dprm")):
        code = str(rows[method].get("code") or "")
        missing_lines = [line for line in spec[excerpt_key] if line.strip() not in code]
        if missing_lines:
            raise ValueError(f"case {index} {method} excerpt is not present in the saved code: {missing_lines}")
    rounded_box(ax, 0.012, y, 0.976, height, "#ffffff")
    top = y + height
    ax.text(0.030, top - 0.026, f"Case {index}  |  {spec['title']}", transform=ax.transAxes, fontsize=15.2, color=INK, va="top")
    ax.text(0.030, top - 0.063, spec["prompt"], transform=ax.transAxes, fontsize=11.3, color=MUTED, va="top")

    excluded = {"<|endoftext|>"}
    conf_metrics = order_metrics(confidence, excluded)
    dprm_metrics = order_metrics(dprm, excluded)
    metric_line = (
        f"mean span {conf_metrics['same_step_span']:.2f} -> {dprm_metrics['same_step_span']:.2f}    "
        f"backfill {100 * conf_metrics['backfill_step_rate']:.1f}% -> {100 * dprm_metrics['backfill_step_rate']:.1f}%    "
        f"adjacency {100 * conf_metrics['same_step_adjacency']:.1f}% -> {100 * dprm_metrics['same_step_adjacency']:.1f}%"
    )
    ax.text(0.970, top - 0.026, metric_line, transform=ax.transAxes, fontsize=10.5, color=GREEN, ha="right", va="top")
    ax.text(0.970, top - 0.063, spec["insight"], transform=ax.transAxes, fontsize=10.8, color=ORANGE, ha="right", va="top")

    panel_y = y + 0.020
    panel_height = height - 0.102
    draw_code_panel(
        ax,
        0.030,
        panel_y,
        0.455,
        panel_height,
        "Confidence order",
        result_text(confidence),
        spec["confidence"],
        BLUE,
        BLUE_BG,
        False,
    )
    draw_code_panel(
        ax,
        0.515,
        panel_y,
        0.455,
        panel_height,
        "DPRM order",
        result_text(dprm),
        spec["dprm"],
        GREEN,
        GREEN_BG,
        True,
    )


def render_page(output: Path, indices: tuple[int, ...], cases: dict[int, dict[str, dict]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    fig_height = 4.05 + 2.72 * len(indices)
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=160, facecolor=SOFT)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.018, 0.976, "How reward tilt changes a reasoning trajectory", transform=ax.transAxes, fontsize=24, color=INK, va="top")
    ax.text(
        0.018,
        0.937,
        r"$g_i=\log\psi_i+\eta_i\,\beta\widehat R_{\phi,b_i}$: a useful reward bucket can promote a still-masked position over confidence.",
        transform=ax.transAxes,
        fontsize=14.3,
        color=GREEN,
        va="top",
    )
    ax.text(
        0.018,
        0.904,
        "Pipeline-level interpretation at the paired 2.00M/unmasking-3 endpoint; token values still come from each host denoiser.",
        transform=ax.transAxes,
        fontsize=11.5,
        color=MUTED,
        va="top",
    )

    row_gap = 0.015
    first_top = 0.865
    bottom = 0.035
    row_height = (first_top - bottom - row_gap * (len(indices) - 1)) / len(indices)
    for row, index in enumerate(indices):
        y = first_top - row * (row_height + row_gap) - row_height
        draw_case(ax, index, cases[index], y, row_height)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", pad_inches=0.06, facecolor=SOFT)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        if output.suffix.lower() == ".webp":
            image.save(output, format="WEBP", quality=92, method=6)
        else:
            image.save(output, format="PNG", optimize=True)


def parse_pages(value: str | None) -> tuple[tuple[int, ...], ...]:
    if not value:
        return PAGES
    pages = tuple(
        tuple(int(item.strip()) for item in page.split(",") if item.strip())
        for page in value.split(";")
        if page.strip()
    )
    if not pages or any(not page for page in pages):
        raise ValueError("--pages must contain semicolon-separated, non-empty case-id lists")
    unknown = sorted({index for page in pages for index in page} - set(CASE_SPECS))
    if unknown:
        raise ValueError(f"unknown case ids in --pages: {unknown}")
    return pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", help="Semicolon-separated pages of comma-separated case ids")
    parser.add_argument("--output-prefix", default="puma_reasoning_gallery")
    parser.add_argument("--format", choices=("webp", "png"), default="webp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.trace_dir)
    for page_number, indices in enumerate(parse_pages(args.pages), start=1):
        output = args.output_dir / f"{args.output_prefix}_{page_number}.{args.format}"
        render_page(output, indices, cases)
        print(output)


if __name__ == "__main__":
    main()
