import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "integrations" / "llada_v" / "scripts"
OVERLAY_ROOT = REPO_ROOT / "integrations" / "llada_v" / "overlay"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(OVERLAY_ROOT))

from dprm_generation import _candidate_aux_bins, apply_dprm_scores  # noqa: E402
from build_dprm_table import normalize_number  # noqa: E402


def test_format_eot_position_aux_bins() -> None:
    cfg = {
        "aux_mode": "format_eot_position",
        "position_bins": 2,
        "format_bins": 3,
    }
    predicted = torch.tensor([[10, 99, 11, 12]])
    actual = _candidate_aux_bins(
        seq_len=4,
        prompt_length=0,
        gen_length=4,
        table_cfg=cfg,
        table_aux_bins=12,
        device=torch.device("cpu"),
        position_offset=0,
        predicted_token_ids=predicted,
        eot_token_ids=[99],
        context_bin=2,
    )
    assert actual.tolist() == [[8, 10, 9, 9]]


def test_reward_value_can_overcome_confidence_margin(tmp_path: Path) -> None:
    counts = torch.ones((1, 2, 4), dtype=torch.float64)
    exp_reward_sums = torch.ones_like(counts)
    # Choice-format, second-position non-EOT has value log(exp(1)) = 1;
    # the competing EOT cell has value zero.
    exp_reward_sums[0, 1, 1] = torch.exp(torch.tensor(1.0, dtype=torch.float64))
    table = {
        "cfg": {
            "aux_mode": "format_eot_position",
            "position_bins": 2,
            "format_bins": 1,
            "reward_temperature": 1.0,
            "ready_count": 1,
        },
        "counts": counts.tolist(),
        "exp_reward_sums": exp_reward_sums.tolist(),
    }
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")

    base = torch.tensor([[0.95, 0.80]], dtype=torch.float64)
    predicted = torch.tensor([[99, 10]])
    diagnostics = {}
    scores = apply_dprm_scores(
        remasking="dprm_confidence_warmup",
        base_score=base,
        bucket_confidence=base,
        dprm_table=str(table_path),
        step_index=0,
        total_steps=2,
        prompt_length=0,
        gen_length=2,
        dprm_guidance_scale=1.0,
        dprm_ready_count=1,
        dprm_force_full=True,
        predicted_token_ids=predicted,
        eot_token_ids=[99],
        dprm_context_bin=0,
        diagnostics=diagnostics,
    )

    assert int(torch.argmax(base, dim=1).item()) == 0
    assert int(torch.argmax(scores, dim=1).item()) == 1
    assert diagnostics["gate"].tolist() == [[1.0, 1.0]]


def test_zero_word_number_normalization_is_symmetric() -> None:
    for text in ("no pedestrians", "None", "zero cars"):
        assert normalize_number(text) == "0"
    assert normalize_number("not visible") is None
