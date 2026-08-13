#!/usr/bin/env python3
"""Package Omni-Diffusion formal eval outputs for human visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ORDER_LABELS = {
    "random": "Random",
    "progressive_confidence": "Omni default",
    "omni_default": "Omni default",
    "dprm_direct": "DPRM-direct",
    "dprm_confidence_warmup": "DPRM",
    "dprm_random_warmup": "DPRM-random",
}


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def short_text(text: str, max_len: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def get_font(size: int, *, bold: bool = True) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for path in (
        f"/usr/share/fonts/truetype/dejavu/{names[0]}",
        f"/usr/share/fonts/truetype/liberation2/{names[1]}",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def mean_clip(records: list[dict[str, Any]]) -> float | None:
    vals = [float(rec["clip_cosine"]) for rec in records if rec.get("clip_cosine") is not None]
    return statistics.fmean(vals) if vals else None


def by_prompt(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(rec["prompt_id"]): rec
        for rec in records
        if rec.get("has_image") and Path(str(rec.get("image_path", ""))).is_file()
    }


def paste_thumbnail(sheet: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = box
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((w, h))
    x = x0 + (w - image.width) // 2
    y = y0 + (h - image.height) // 2
    sheet.paste(image, (x, y))


def make_grid(
    order_maps: dict[str, dict[str, dict[str, Any]]],
    orders: list[str],
    prompt_ids: list[str],
    output: Path,
    title: str,
    *,
    thumb: int = 190,
) -> None:
    if not prompt_ids:
        return
    title_h = 44
    header_h = 32
    prompt_h = 48
    left_w = 300
    width = left_w + len(orders) * thumb
    prompts = {
        prompt_id: next(
            (
                str(order_maps[order][prompt_id].get("prompt", ""))
                for order in orders
                if prompt_id in order_maps[order]
            ),
            "",
        )
        for prompt_id in prompt_ids
    }
    wrapped = {
        prompt_id: textwrap.wrap(" ".join(prompt.split()), width=34) or [""]
        for prompt_id, prompt in prompts.items()
    }
    row_heights = [
        max(thumb + prompt_h, 42 + 16 * len(wrapped[prompt_id]))
        for prompt_id in prompt_ids
    ]
    height = title_h + header_h + sum(row_heights)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    f_title = get_font(20)
    f_header = get_font(15)
    f_small = get_font(12, bold=False)

    draw.text((12, 10), title, fill=(20, 20, 20), font=f_title)
    y = title_h
    for col, order in enumerate(orders):
        x = left_w + col * thumb
        draw.rectangle((x, y, x + thumb - 1, y + header_h - 1), fill=(232, 242, 255))
        draw.text((x + 8, y + 8), ORDER_LABELS.get(order, order), fill=(20, 50, 90), font=f_header)
    y += header_h

    y0 = y
    for row, prompt_id in enumerate(prompt_ids):
        draw.text((8, y0 + 8), prompt_id, fill=(20, 20, 20), font=f_header)
        draw.multiline_text(
            (8, y0 + 34), "\n".join(wrapped[prompt_id]), fill=(55, 55, 55), font=f_small, spacing=3
        )
        for col, order in enumerate(orders):
            rec = order_maps[order][prompt_id]
            x = left_w + col * thumb
            paste_thumbnail(sheet, Path(str(rec["image_path"])), (x, y0, thumb, thumb))
            score = rec.get("clip_cosine")
            score_text = f"CLIP {float(score):.3f}" if score is not None else "CLIP --"
            draw.text((x + 8, y0 + thumb + 8), score_text, fill=(30, 30, 30), font=f_small)
        y0 += row_heights[row]

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def blind_assignment(prompt_id: str, orders: list[str], seed: int) -> dict[str, str]:
    """Return a deterministic per-prompt column-to-order permutation."""
    shuffled = sorted(
        orders,
        key=lambda order: hashlib.sha256(
            f"{seed}:{prompt_id}:{order}".encode("utf-8")
        ).digest(),
    )
    labels = [chr(ord("A") + idx) for idx in range(len(shuffled))]
    return dict(zip(labels, shuffled))


def make_blind_grid(
    order_maps: dict[str, dict[str, dict[str, Any]]],
    prompt_ids: list[str],
    assignments: dict[str, dict[str, str]],
    output: Path,
    title: str,
    *,
    thumb: int = 190,
) -> None:
    if not prompt_ids:
        return
    labels = sorted(next(iter(assignments.values())))
    title_h = 44
    header_h = 32
    prompt_h = 48
    left_w = 300
    width = left_w + len(labels) * thumb
    prompts = {
        prompt_id: str(
            order_maps[assignments[prompt_id][labels[0]]][prompt_id].get("prompt", "")
        )
        for prompt_id in prompt_ids
    }
    wrapped = {
        prompt_id: textwrap.wrap(" ".join(prompt.split()), width=34) or [""]
        for prompt_id, prompt in prompts.items()
    }
    row_heights = [
        max(thumb + prompt_h, 42 + 16 * len(wrapped[prompt_id]))
        for prompt_id in prompt_ids
    ]
    height = title_h + header_h + sum(row_heights)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    f_title = get_font(20)
    f_header = get_font(15)
    f_small = get_font(12, bold=False)

    draw.text((12, 10), title, fill=(20, 20, 20), font=f_title)
    y = title_h
    for col, label in enumerate(labels):
        x = left_w + col * thumb
        draw.rectangle((x, y, x + thumb - 1, y + header_h - 1), fill=(242, 242, 242))
        draw.text((x + 8, y + 8), label, fill=(20, 20, 20), font=f_header)
    y += header_h

    y0 = y
    for row, prompt_id in enumerate(prompt_ids):
        assignment = assignments[prompt_id]
        draw.text((8, y0 + 8), prompt_id, fill=(20, 20, 20), font=f_header)
        draw.multiline_text(
            (8, y0 + 34), "\n".join(wrapped[prompt_id]), fill=(55, 55, 55), font=f_small, spacing=3
        )
        for col, label in enumerate(labels):
            order = assignment[label]
            rec = order_maps[order][prompt_id]
            x = left_w + col * thumb
            paste_thumbnail(sheet, Path(str(rec["image_path"])), (x, y0, thumb, thumb))
            draw.text((x + 8, y0 + thumb + 8), label, fill=(30, 30, 30), font=f_small)
        y0 += row_heights[row]

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def write_blind_audit(
    order_maps: dict[str, dict[str, dict[str, Any]]],
    orders: list[str],
    prompt_ids: list[str],
    out_dir: Path,
    *,
    seed: int,
    prompts_per_sheet: int,
) -> tuple[list[Path], Path, Path]:
    assignments = {
        prompt_id: blind_assignment(prompt_id, orders, seed) for prompt_id in prompt_ids
    }
    labels = [chr(ord("A") + idx) for idx in range(len(orders))]
    sheets: list[Path] = []
    for sheet_idx, start in enumerate(range(0, len(prompt_ids), prompts_per_sheet)):
        sheet_ids = prompt_ids[start : start + prompts_per_sheet]
        path = out_dir / "blind_sheets" / f"blind_sheet_{sheet_idx:02d}.png"
        make_blind_grid(
            order_maps,
            sheet_ids,
            assignments,
            path,
            f"Omni-Diffusion blinded audit {sheet_idx + 1}",
        )
        sheets.append(path)

    key_path = out_dir / "blind_key.tsv"
    with key_path.open("w", encoding="utf-8") as handle:
        handle.write("prompt_id\t" + "\t".join(f"{label}_order" for label in labels) + "\n")
        for prompt_id in prompt_ids:
            handle.write(
                prompt_id
                + "\t"
                + "\t".join(assignments[prompt_id][label] for label in labels)
                + "\n"
            )

    ratings_path = out_dir / "human_rating_template.tsv"
    rating_fields = ["prompt_id", "sheet"]
    for label in labels:
        rating_fields.extend(
            [
                f"{label}_semantic_alignment_1to5",
                f"{label}_visual_coherence_1to5",
                f"{label}_artifact_free_1to5",
            ]
        )
    rating_fields.extend(["preferred_column", "notes"])
    with ratings_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(rating_fields) + "\n")
        for idx, prompt_id in enumerate(prompt_ids):
            sheet_idx = idx // prompts_per_sheet
            handle.write(f"{prompt_id}\tblind_sheet_{sheet_idx:02d}.png\t")
            handle.write("\t" * (len(rating_fields) - 3))
            handle.write("\n")

    return sheets, key_path, ratings_path


def write_scores(
    records: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    orders: list[str],
    out_path: Path,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    clip_summary = summary.get("clip", {}).get("orders", {}) if isinstance(summary, dict) else {}
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("order\tcount\timage_count\tmean_clip_cosine\tmedian_clip_cosine\n")
        for order in orders:
            order_records = records.get(order, [])
            order_clip = clip_summary.get(order, {})
            mean = order_clip.get("mean_clip_cosine")
            median = order_clip.get("median_clip_cosine")
            if mean is None:
                mean = mean_clip(order_records)
            if median is None:
                vals = [float(rec["clip_cosine"]) for rec in order_records if rec.get("clip_cosine") is not None]
                median = statistics.median(vals) if vals else None
            image_count = sum(1 for rec in order_records if rec.get("has_image"))
            count = len(order_records)
            scores[order] = {
                "count": count,
                "image_count": image_count,
                "mean_clip_cosine": mean,
                "median_clip_cosine": median,
            }
            handle.write(
                f"{order}\t{count}\t{image_count}\t"
                f"{mean if mean is not None else 'NA'}\t{median if median is not None else 'NA'}\n"
            )
    return scores


def write_index(
    out_path: Path,
    *,
    records_path: Path,
    summary_path: Path,
    scores_path: Path,
    scores: dict[str, dict[str, Any]],
    common_count: int,
    first_sheet: Path,
    paired_sheet: Path,
    blind_sheets: list[Path],
    blind_key_path: Path,
    ratings_path: Path,
) -> None:
    lines = [
        "# Omni Formal Visual Audit",
        "",
        f"Records: `{records_path}`",
        f"Summary: `{summary_path}`",
        f"Common complete prompts: `{common_count}`",
        "",
        "## Scores",
        "",
        "| Order | Images | Mean CLIP | Median CLIP |",
        "|---|---:|---:|---:|",
    ]
    for order, vals in scores.items():
        mean = vals.get("mean_clip_cosine")
        median = vals.get("median_clip_cosine")
        lines.append(
            f"| `{order}` | {vals.get('image_count', 0)} | "
            f"{fmt_float(mean)} | {fmt_float(median)} |"
        )
    lines += [
        "",
        "## Image Sheets",
        "",
        f"- Scores TSV: `{scores_path}`",
        f"- Paired contact sheet from formal summary: `{paired_sheet}`",
        f"- First matched examples: `{first_sheet}`",
    ]
    lines += [
        "",
        "## Blinded Human Audit",
        "",
        f"- Blinded sheets: `{blind_sheets[0].parent if blind_sheets else 'NA'}`",
        f"- Rating template: `{ratings_path}`",
        f"- Reveal key (keep hidden until ratings are complete): `{blind_key_path}`",
        "- Rate semantic alignment, visual coherence, and artifact freedom from 1 (poor) to 5 (strong).",
        "",
        "Use the blinded sheets for the formal human-readability screen; CLIP is only a ranking aid.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", default=None)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--blind-seed", type=int, default=20260713)
    args = parser.parse_args()

    records = load_json(args.records)
    summary = load_json(args.summary)
    if not isinstance(records, dict):
        raise SystemExit(f"records must be an order->records mapping: {args.records}")
    orders = args.orders or list(records)
    orders = [order for order in orders if order in records]
    if not orders:
        raise SystemExit("no requested orders found in records")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    order_maps = {order: by_prompt(records[order]) for order in orders}
    common = sorted(set.intersection(*(set(mapping) for mapping in order_maps.values()))) if order_maps else []

    scores_path = args.out_dir / "formal_visual_audit_scores.tsv"
    scores = write_scores(records, summary, orders, scores_path)

    first_ids = common[: args.num_examples]
    first_sheet = args.out_dir / "omni_formal_first_examples.png"
    make_grid(order_maps, orders, first_ids, first_sheet, "Omni-Diffusion formal eval: first matched examples")

    paired_sheet = args.summary.parent / "paired_contact_sheet.png"
    blind_sheets, blind_key_path, ratings_path = write_blind_audit(
        order_maps,
        orders,
        common,
        args.out_dir,
        seed=args.blind_seed,
        prompts_per_sheet=args.num_examples,
    )
    write_index(
        args.out_dir / "visual_audit_index.md",
        records_path=args.records,
        summary_path=args.summary,
        scores_path=scores_path,
        scores=scores,
        common_count=len(common),
        first_sheet=first_sheet,
        paired_sheet=paired_sheet,
        blind_sheets=blind_sheets,
        blind_key_path=blind_key_path,
        ratings_path=ratings_path,
    )
    selection_manifest = args.out_dir / "selection_manifest.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "selection_policy": "first lexicographic matched prompt ids before score inspection",
                "main_figure_prompt_id": first_ids[0] if first_ids else None,
                "supplement_fixed_prompt_ids": first_ids[1:],
                "first_prompt_ids": first_ids,
                "blind_prompt_ids": common,
                "orders": orders,
                "outcome_ranked_selection": False,
                "clip_used_for_selection": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "orders": orders,
                "common_prompt_count": len(common),
                "first_examples": len(first_ids),
                "blind_sheets": len(blind_sheets),
                "blind_key": str(blind_key_path),
                "ratings_template": str(ratings_path),
                "selection_manifest": str(selection_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
