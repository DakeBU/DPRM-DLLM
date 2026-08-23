from pathlib import Path
import importlib.util


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/analyze_omni_paired_results.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_omni_paired_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_default_comparisons_are_preserved():
    assert MODULE.parse_comparisons(None) == MODULE.COMPARISONS


def test_custom_comparisons():
    assert MODULE.parse_comparisons(["a:b", "c:d"]) == (("a", "b"), ("c", "d"))


def test_invalid_comparison():
    try:
        MODULE.parse_comparisons(["missing_separator"])
    except ValueError as exc:
        assert "BASELINE:METHOD" in str(exc)
    else:
        raise AssertionError("invalid comparison was accepted")
