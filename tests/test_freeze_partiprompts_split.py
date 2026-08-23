import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/omni_diffusion/matched/scripts/freeze_partiprompts_split.py"


def test_freeze_partiprompts_split_is_disjoint_and_deterministic(tmp_path: Path):
    source = tmp_path / "PartiPrompts.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Prompt", "Category", "Challenge"], delimiter="\t"
        )
        writer.writeheader()
        for index in range(12):
            writer.writerow(
                {"Prompt": f"prompt {index}", "Category": "objects", "Challenge": "basic"}
            )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "split"
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--output-root",
        str(output),
        "--source-url",
        "https://example.test/PartiPrompts.tsv",
        "--expected-sha256",
        source_sha,
        "--development-count",
        "4",
        "--confirmation-count",
        "6",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first_manifest = (output / "manifest.json").read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "manifest.json").read_bytes() == first_manifest

    development = {
        json.loads(line)["prompt"]
        for line in (output / "development.jsonl").read_text().splitlines()
    }
    confirmation = {
        json.loads(line)["prompt"]
        for line in (output / "confirmation.jsonl").read_text().splitlines()
    }
    assert len(development) == 4
    assert len(confirmation) == 6
    assert development.isdisjoint(confirmation)
