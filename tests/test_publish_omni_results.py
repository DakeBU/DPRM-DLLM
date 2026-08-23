import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/omni_diffusion/matched/scripts/publish_omni_results.py"


def row(baseline: str, method: str, base: float, value: float) -> dict:
    return {
        "baseline": baseline,
        "method": method,
        "matched_prompts": 512,
        "baseline_mean": base,
        "method_mean": value,
        "mean_delta": value - base,
        "ci95_low": 0.001,
        "ci95_high": 0.005,
    }


def test_publishes_only_promoted_confirmation(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion.json"
    paired = tmp_path / "paired.json"
    results = tmp_path / "results.csv"
    registry = tmp_path / "registry.json"
    summary = tmp_path / "summary.json"
    promotion.write_text(json.dumps({"passed": True}))
    paired.write_text(
        json.dumps(
            {
                "comparisons_by_metric": {
                    metric: [row("progressive_confidence", "dprm_confidence_warmup", 0.25, 0.255)]
                    for metric in ("clip_cosine", "clip_b32_cosine")
                }
            }
        )
    )
    results.write_text(
        ",".join(("host", "task", "method", "variant", "metric", "direction", "value", "ci95_low", "ci95_high", "n", "protocol")) + "\n",
        encoding="utf-8",
    )
    registry.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "id": "omni_diffusion",
                        "variants": [
                            {"id": name, "status": "formal_pending"}
                            for name in ("random", "confidence", "dprm_confidence", "dprm_random")
                        ],
                    }
                ]
            }
        )
    )
    summary.write_text(json.dumps({"llada_v": {}}))
    subprocess.run(
        [
            sys.executable, str(SCRIPT), "--promotion", str(promotion),
            "--paired", str(paired), "--paper-results", str(results),
            "--registry", str(registry), "--multimodal-summary", str(summary),
        ],
        check=True,
    )
    rows = list(csv.DictReader(results.open()))
    assert len(rows) == 6
    assert {row["protocol"] for row in rows} == {"untouched_confirmation"}
    statuses = {
        row["id"]: row["status"]
        for row in json.loads(registry.read_text())["experiments"][0]["variants"]
    }
    assert statuses["confidence"] == "reported"
    assert statuses["dprm_confidence"] == "reported"
    assert statuses["random"] == "implemented_control"
    assert statuses["dprm_random"] == "implemented_control"
