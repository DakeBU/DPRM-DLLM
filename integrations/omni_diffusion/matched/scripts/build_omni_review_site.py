#!/usr/bin/env python3
"""Build a local, lazy-loaded review page for all frozen Omni comparisons."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


METHODS = ["confidence", "step96_q0.70", "step96_q0.85", "step96_q0.90", "step96_q0.95"]
SALT = "uniform_gallery_v1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, Any]) -> tuple[str, int]:
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


def image_cell(label: str, item: dict[str, Any], root: Path) -> str:
    path = resolve(item, root).as_uri()
    return (
        f'<figure><h3>{html.escape(label)}</h3><img loading="lazy" src="{path}">'
        f'<figcaption>L/14 {float(item["clip_cosine"]):.3f} &nbsp; '
        f'B/32 {float(item["clip_b32_cosine"]):.3f}</figcaption></figure>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = load(args.summary)["records"]
    scored = load(args.scored_records)
    lookup = {name: {key(row): row for row in rows} for name, rows in scored.items()}
    cards = []
    for index, chosen in enumerate(selected):
        prompt, seed = key(chosen)
        confidence = lookup["confidence"][(prompt, seed)]
        random = lookup["random"][(prompt, seed)]
        uniform_name = uniform_method(prompt, seed)
        uniform = lookup[uniform_name][(prompt, seed)]
        dprm = lookup[str(chosen["selected_method"])][(prompt, seed)]
        dual_win = (
            float(dprm["clip_cosine"]) > float(confidence["clip_cosine"])
            and float(dprm["clip_b32_cosine"]) > float(confidence["clip_b32_cosine"])
        )
        images = "".join(
            (
                image_cell("Random", random, args.confirmation_root),
                image_cell("Omni default", confidence, args.confirmation_root),
                image_cell("Uniform order", uniform, args.confirmation_root),
                image_cell("DPRM", dprm, args.confirmation_root),
            )
        )
        cards.append(
            f'<article data-prompt="{html.escape(prompt.lower())}" data-win="{str(dual_win).lower()}">'
            f'<header><b>Record {index}</b><span>seed {seed}</span>'
            f'<span>{"dual-CLIP win" if dual_win else "all records"}</span></header>'
            f'<p>{html.escape(prompt)}</p><div class="grid">{images}</div></article>'
        )

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Omni confirmation review</title><style>
*{{box-sizing:border-box}}body{{margin:0;font:14px Arial,sans-serif;color:#17191c;background:#f3f5f6}}
.bar{{position:sticky;top:0;z-index:2;display:flex;gap:16px;align-items:center;padding:14px 24px;background:#fff;border-bottom:1px solid #ccd1d5}}
input[type=search]{{width:min(520px,55vw);padding:9px;border:1px solid #aeb5bb}}main{{padding:24px}}
article{{max-width:1500px;margin:0 auto 28px;padding:18px;background:#fff;border:1px solid #ccd1d5}}
article header{{display:flex;gap:16px;color:#626970;font-size:12px}}article p{{font-size:16px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:10px;overflow-x:auto}}
figure{{margin:0;min-width:240px}}h3{{margin:0 0 7px;font-size:13px}}img{{display:block;width:100%;aspect-ratio:1;object-fit:cover;border:1px solid #c5cbd0}}
figcaption{{padding-top:5px;color:#596168;font-size:11px}}article[hidden]{{display:none}}
</style></head><body><div class="bar"><b>Omni 512-prompt review</b>
<input id="q" type="search" placeholder="Filter prompt or record index">
<label><input id="wins" type="checkbox"> dual-CLIP wins only</label><span id="count">512 shown</span></div>
<main>{''.join(cards)}</main><script>
const cards=[...document.querySelectorAll('article')],q=document.querySelector('#q'),wins=document.querySelector('#wins'),count=document.querySelector('#count');
function filter(){{const s=q.value.trim().toLowerCase();let n=0;cards.forEach((c,i)=>{{const ok=(!s||c.dataset.prompt.includes(s)||String(i).includes(s))&&(!wins.checked||c.dataset.win==='true');c.hidden=!ok;if(ok)n++}});count.textContent=`${{n}} shown`}}q.oninput=filter;wins.onchange=filter;
</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
