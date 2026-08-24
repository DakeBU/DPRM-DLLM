#!/usr/bin/env python3
"""Render one self-contained four-method comparison image per Omni prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


METHODS = ["confidence", "step96_q0.70", "step96_q0.85", "step96_q0.90", "step96_q0.95"]
SALT = "uniform_gallery_v1"
LABELS = ("Random", "Omni default", "Uniform action", "DPRM")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["prompt"]), int(row["seed"])


def uniform_method(prompt: str, seed: int) -> str:
    digest = hashlib.sha256(f"{prompt}|{seed}|{SALT}".encode()).hexdigest()
    return METHODS[int(digest, 16) % len(METHODS)]


def resolve(item: dict[str, Any], root: Path) -> Path:
    path = Path(item["image_path"])
    if path.is_file():
        return path.resolve()
    for marker in ("baseline", "random", "branches"):
        if marker in path.parts:
            candidate = root.joinpath(*path.parts[path.parts.index(marker) :])
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(path)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def slug(text: str, limit: int = 56) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (value[:limit].rstrip("-") or "prompt")


def render_composite(
    output: Path,
    index: int,
    prompt: str,
    seed: int,
    items: list[dict[str, Any]],
    root: Path,
) -> None:
    panel = 384
    gap = 18
    margin = 24
    title_height = 104
    label_height = 46
    metric_height = 42
    width = margin * 2 + panel * 4 + gap * 3
    height = title_height + label_height + panel + metric_height + margin
    canvas = Image.new("RGB", (width, height), "#f4f5f6")
    draw = ImageDraw.Draw(canvas)

    title = f"{index:03d}  |  seed {seed}  |  {prompt}"
    title_lines = textwrap.wrap(title, width=105, max_lines=2, placeholder="...")
    draw.multiline_text(
        (margin, 18),
        "\n".join(title_lines),
        fill="#15181b",
        font=font(27, bold=True),
        spacing=5,
    )

    for column, (label, item) in enumerate(zip(LABELS, items)):
        x = margin + column * (panel + gap)
        y = title_height
        draw.rounded_rectangle(
            (x, y, x + panel, y + label_height - 6),
            radius=5,
            fill="#ffffff",
            outline="#c6cbd0",
            width=2,
        )
        draw.text((x + 12, y + 7), label, fill="#1d2328", font=font(23, bold=True))
        source = Image.open(resolve(item, root)).convert("RGB")
        source = source.resize((panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(source, (x, y + label_height))
        draw.rectangle(
            (x, y + label_height, x + panel - 1, y + label_height + panel - 1),
            outline="#aeb5bb",
            width=2,
        )
        metrics = (
            f"CLIP-L/14 {float(item['clip_cosine']):.3f}   "
            f"CLIP-B/32 {float(item['clip_b32_cosine']):.3f}"
        )
        draw.text(
            (x + 7, y + label_height + panel + 8),
            metrics,
            fill="#4f5860",
            font=font(18),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "WEBP", quality=92, method=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = load(args.summary)["records"]
    scored = load(args.scored_records)
    lookup = {name: {record_key(row): row for row in rows} for name, rows in scored.items()}
    image_dir = args.output_dir / "comparisons"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for index, chosen in enumerate(selected):
        prompt, seed = record_key(chosen)
        confidence = lookup["confidence"][(prompt, seed)]
        random = lookup["random"][(prompt, seed)]
        uniform_name = uniform_method(prompt, seed)
        uniform = lookup[uniform_name][(prompt, seed)]
        selected_name = str(chosen["selected_method"])
        dprm = lookup[selected_name][(prompt, seed)]
        dual_win = (
            float(dprm["clip_cosine"]) > float(confidence["clip_cosine"])
            and float(dprm["clip_b32_cosine"]) > float(confidence["clip_b32_cosine"])
        )
        filename = f"{index:03d}_{slug(prompt)}.webp"
        render_composite(
            image_dir / filename,
            index,
            prompt,
            seed,
            [random, confidence, uniform, dprm],
            args.confirmation_root,
        )
        manifest.append(
            {
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "image": f"comparisons/{filename}",
                "uniform_method": uniform_name,
                "dprm_method": selected_name,
                "dual_clip_win": dual_win,
                "confidence_clip_l14": float(confidence["clip_cosine"]),
                "dprm_clip_l14": float(dprm["clip_cosine"]),
                "confidence_clip_b32": float(confidence["clip_b32_cosine"]),
                "dprm_clip_b32": float(dprm["clip_b32_cosine"]),
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    cards = "".join(
        f'<article data-prompt="{html.escape(row["prompt"].lower())}" '
        f'data-win="{str(row["dual_clip_win"]).lower()}">'
        f'<a href="{html.escape(row["image"])}"><img loading="lazy" '
        f'src="{html.escape(row["image"])}" alt="{html.escape(row["prompt"])}"></a>'
        f"</article>"
        for row in manifest
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Omni four-method comparisons</title><style>
*{{box-sizing:border-box}}body{{margin:0;font:14px Arial,sans-serif;color:#17191c;background:#eef1f3}}
.bar{{position:sticky;top:0;z-index:2;display:flex;gap:16px;align-items:center;padding:14px 22px;background:#fff;border-bottom:1px solid #c8ced3}}
input[type=search]{{width:min(560px,55vw);padding:9px;border:1px solid #aeb5bb}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(680px,1fr));gap:16px;padding:18px}}
article{{background:#fff;border:1px solid #c8ced3;padding:8px}}article[hidden]{{display:none}}
img{{display:block;width:100%;height:auto}}</style></head><body>
<div class="bar"><b>Omni four-method comparisons</b>
<input id="q" type="search" placeholder="Filter prompt or record index">
<label><input id="wins" type="checkbox"> dual-CLIP wins only</label>
<span id="count">{len(manifest)} shown</span></div><main>{cards}</main><script>
const cards=[...document.querySelectorAll('article')],q=document.querySelector('#q'),wins=document.querySelector('#wins'),count=document.querySelector('#count');
function filter(){{const s=q.value.trim().toLowerCase();let n=0;cards.forEach((c,i)=>{{const ok=(!s||c.dataset.prompt.includes(s)||String(i).includes(s))&&(!wins.checked||c.dataset.win==='true');c.hidden=!ok;if(ok)n++}});count.textContent=String(n)+' shown'}}q.oninput=filter;wins.onchange=filter;
</script></body></html>"""
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {len(manifest)} composites to {args.output_dir}")


if __name__ == "__main__":
    main()
