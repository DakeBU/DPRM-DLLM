import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/omni_diffusion/matched/scripts/render_omni_qualitative_pairs.py"


def test_renders_fixed_two_by_two_endpoint_sheet(tmp_path: Path) -> None:
    payload = {}
    for order, color in (("progressive_confidence", "blue"), ("dprm_confidence_warmup", "green")):
        payload[order] = []
        for prompt_id in ("beach_three_children", "boy_flute_kittens"):
            image = tmp_path / f"{order}_{prompt_id}.png"
            Image.new("RGB", (128, 128), color).save(image)
            payload[order].append({"prompt_id": prompt_id, "prompt": prompt_id, "image_path": str(image)})
    records = tmp_path / "records.json"
    output = tmp_path / "sheet.png"
    records.write_text(json.dumps(payload))
    subprocess.run(
        [
            sys.executable, str(SCRIPT), "--records", str(records), "--output", str(output),
            "--prompt-ids", "beach_three_children", "boy_flute_kittens",
        ],
        check=True,
    )
    assert Image.open(output).size == (898, 908)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert [row["prompt_id"] for row in manifest["rows"]] == [
        "beach_three_children", "boy_flute_kittens"
    ]
