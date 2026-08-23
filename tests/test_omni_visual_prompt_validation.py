from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "integrations"
    / "omni_diffusion"
    / "matched"
    / "scripts"
    / "validate_omni_visual_prompts.py"
)


def test_visual_prompt_validator_checks_text_and_split(tmp_path: Path) -> None:
    prompts = [f"concrete prompt {index}" for index in range(8)]
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(
            json.dumps({"messages": [{"content": f"header\n{text}"}]}) + "\n"
            for text in prompts
        ),
        encoding="utf-8",
    )
    selected = [2, 3, 4, 5]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "selection_rule": "before generation",
                "prompts": [
                    {
                        "prompt_id": index,
                        "text": prompts[index],
                        "sha256": hashlib.sha256(prompts[index].encode()).hexdigest(),
                    }
                    for index in selected
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable, str(VALIDATOR), "--data", str(data),
            "--manifest", str(manifest), "--eval-offset", "2",
            "--eval-count", "4", "--expected-prompt-ids",
            *[str(index) for index in selected], "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
