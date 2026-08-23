import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/select_omni_dual_clip_controller.py"
)
SPEC = importlib.util.spec_from_file_location("select_omni_dual_clip_controller", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_intervention_stats_counts_only_direct_overrides(tmp_path):
    prompt = tmp_path / "candidate" / "prompt_0001"
    prompt.mkdir(parents=True)
    trace = prompt / "sample_order_trace.jsonl"
    rows = [
        {"selected_candidate_indices": [2], "confidence_default_candidate_index": 1},
        {"selected_candidate_indices": [3], "confidence_default_candidate_index": 3},
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    stats = MODULE.intervention_stats(tmp_path, "candidate")

    assert stats["traced_prompts"] == 1.0
    assert stats["mean_direct_overrides"] == 1.0
    assert stats["prompt_fraction_with_override"] == 1.0
