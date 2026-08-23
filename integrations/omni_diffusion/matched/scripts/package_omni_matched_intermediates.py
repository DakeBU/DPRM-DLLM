#!/usr/bin/env python3
"""Build fixed-index intermediate-canvas sheets for matched Omni policies."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABELS = {
    "random": "Random",
    "progressive_confidence": "Omni default",
    "dprm_confidence_warmup": "DPRM",
}

COLORS = {
    "random": (90, 90, 90),
    "progressive_confidence": (42, 96, 170),
    "dprm_confidence_warmup": (31, 139, 76),
}
DISPLAY_STEPS = (0, 32, 64, 128, 192, 255)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def read_trace(path: str) -> list[dict]:
    trace_path = Path(path)
    if not trace_path.is_file():
        return []
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_policy_divergence(rows_by_order: dict[str, dict]) -> dict | None:
    dprm = read_trace(
        str(rows_by_order["dprm_confidence_warmup"].get("order_trace_path", ""))
    )
    committed: set[int] = set()
    for dprm_row in dprm:
        selected = dprm_row.get("selected_visual_indices", [])
        default_candidate = dprm_row.get("confidence_default_candidate_index")
        remaining = [idx for idx in range(256) if idx not in committed]
        if (
            selected
            and isinstance(default_candidate, int)
            and 0 <= default_candidate < len(remaining)
        ):
            default_visual = remaining[default_candidate]
            selected_visual = int(selected[0])
            if selected_visual != default_visual:
                selected_base = dprm_row.get("selected_base_order_scores", [None])[0]
                selected_value = dprm_row.get("selected_dprm_values", [None])[0]
                selected_adjusted = dprm_row.get(
                    "selected_adjusted_order_scores", [None]
                )[0]
                return {
                    "step": int(dprm_row["step"]),
                    "confidence_default": divmod(default_visual, 16),
                    "dprm": divmod(selected_visual, 16),
                    "base_order_score": {
                        "confidence_default": dprm_row.get(
                            "confidence_default_base_order_score"
                        ),
                        "dprm": selected_base,
                    },
                    "gated_process_value": {
                        "confidence_default": dprm_row.get(
                            "confidence_default_dprm_value"
                        ),
                        "dprm": selected_value,
                    },
                    "adjusted_order_score": {
                        "confidence_default": dprm_row.get(
                            "confidence_default_adjusted_order_score"
                        ),
                        "dprm": selected_adjusted,
                    },
                }
        committed.update(int(idx) for idx in selected)
    return None


def frame_step(path: str) -> int:
    return int(Path(path).stem.removeprefix("step_"))


def score_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-count", type=int, default=4)
    parser.add_argument("--prompt-ids", nargs="*", default=())
    args = parser.parse_args()
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    orders = [order for order in LABELS if order in payload]
    maps = {
        order: {str(row["prompt_id"]): row for row in payload[order]} for order in orders
    }
    common = sorted(set.intersection(*(set(rows) for rows in maps.values())))
    requested = [str(prompt_id) for prompt_id in args.prompt_ids]
    fixed = requested if requested else common[: args.fixed_count]
    missing = [prompt_id for prompt_id in fixed if prompt_id not in common]
    if missing:
        raise ValueError(f"fixed prompt ids missing from records: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection": (
            "prompt ids fixed from concrete prompt text before generation"
            if requested
            else "first fixed prompt ids before score inspection"
        ),
        "paper_role": "supplementary fixed-index intermediate canvases",
        "main_text_figure_prompt_id": None,
        "supplement_fixed_prompt_ids": fixed,
        "prompts": [],
    }

    for prompt_id in fixed:
        rows = [maps[order][prompt_id] for order in orders]
        rows_by_order = dict(zip(orders, rows))
        histories = []
        for row in rows:
            by_step = {
                frame_step(path): path for path in row.get("history_frame_paths", [])
            }
            histories.append([by_step[step] for step in DISPLAY_STEPS if step in by_step])
        width_frames = min((len(paths) for paths in histories), default=0)
        if width_frames == 0:
            continue
        divergence = first_policy_divergence(rows_by_order)
        if divergence:
            saved_steps = [frame_step(path) for path in histories[-1][:width_frames]]
            divergence["marked_frame_step"] = max(
                (step for step in saved_steps if step <= divergence["step"]),
                default=min(saved_steps),
            )
        thumb = 210
        left = 175
        prompt = " ".join(str(rows[0].get("prompt", "")).split())
        prompt_lines = textwrap.wrap(prompt, width=108) or [""]
        top = 76 + 26 * len(prompt_lines) + (58 if divergence else 0)
        sheet = Image.new("RGB", (left + width_frames * thumb, top + len(orders) * thumb), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{prompt_id} (fixed before scoring)", fill=(20, 20, 20), font=font(21, True))
        for line_idx, line in enumerate(prompt_lines):
            draw.text((8, 38 + 26 * line_idx), line, fill=(20, 20, 20), font=font(18))
        if divergence:
            conf_rc = divergence["confidence_default"]
            dprm_rc = divergence["dprm"]
            info_y = 46 + 26 * len(prompt_lines)
            draw.text(
                (8, info_y),
                f"Same-model intervention at action {divergence['step']}: "
                f"local confidence would select {conf_rc}; DPRM selects {dprm_rc}. "
                "Blue/green boxes mark the two 16x16 grid cells.",
                fill=(35, 35, 35),
                font=font(17, True),
            )
            base = divergence["base_order_score"]
            value = divergence["gated_process_value"]
            draw.text(
                (8, info_y + 20),
                "Native scores (confidence / DPRM): "
                f"{score_text(base['confidence_default'])} / "
                f"{score_text(base['dprm'])}; "
                "gated process values: "
                f"{score_text(value['confidence_default'])} / "
                f"{score_text(value['dprm'])}.",
                fill=(35, 35, 35),
                font=font(16),
            )
        for col in range(width_frames):
            frame_name = Path(histories[0][col]).stem.replace("step_", "step ")
            draw.text((left + col * thumb + 8, top - 24), frame_name, fill=(40, 40, 40), font=font(16))
        for row_idx, (order, paths) in enumerate(zip(orders, histories)):
            y = top + row_idx * thumb
            draw.text((8, y + 88), LABELS[order], fill=COLORS[order], font=font(18, True))
            for col, image_path in enumerate(paths[:width_frames]):
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb, thumb))
                x = left + col * thumb + (thumb - image.width) // 2
                image_y = y + (thumb - image.height) // 2
                sheet.paste(image, (x, image_y))
                if (
                    divergence
                    and order == "dprm_confidence_warmup"
                    and frame_step(image_path) == divergence["marked_frame_step"]
                ):
                    for key, color in (
                        ("confidence_default", COLORS["progressive_confidence"]),
                        ("dprm", COLORS["dprm_confidence_warmup"]),
                    ):
                        grid_row, grid_col = divergence[key]
                        x0 = x + round(grid_col * image.width / 16)
                        y0 = image_y + round(grid_row * image.height / 16)
                        x1 = x + round((grid_col + 1) * image.width / 16)
                        y1 = image_y + round((grid_row + 1) * image.height / 16)
                        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        output = args.output_dir / f"{prompt_id}_intermediate_canvases.png"
        sheet.save(output)
        manifest["prompts"].append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "first_policy_divergence": divergence,
                "sheet": str(output),
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
