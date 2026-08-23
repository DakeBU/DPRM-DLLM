import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/build_omni_prompt_split.py"
)


def test_prompt_split_is_disjoint_and_deterministic(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_text("\n".join(f"prompt {index}" for index in range(12)) + "\n")
    dev = tmp_path / "dev.txt"
    confirm = tmp_path / "confirm.txt"
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--development-output",
        str(dev),
        "--confirmation-output",
        str(confirm),
        "--manifest-output",
        str(manifest),
        "--development-count",
        "4",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = manifest.read_text()
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert manifest.read_text() == first
    dev_prompts = set(dev.read_text().splitlines())
    confirm_prompts = set(confirm.read_text().splitlines())
    assert len(dev_prompts) == 4
    assert len(confirm_prompts) == 8
    assert not dev_prompts & confirm_prompts
    assert json.loads(first)["intersection_count"] == 0
