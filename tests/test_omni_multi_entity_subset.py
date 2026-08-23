import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/analyze_omni_multi_entity_subset.py"
)
SPEC = importlib.util.spec_from_file_location("omni_multi_entity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
RULE = json.loads(
    (Path(__file__).parents[1] / "reproducibility/omni_multi_entity_prompt_rule.json").read_text()
)


def test_multi_entity_rule_uses_declared_terms():
    assert MODULE.classify_prompt("A boy plays music for an audience of kittens.", RULE)
    assert MODULE.classify_prompt("Three red cubes sit on a table.", RULE)
    assert not MODULE.classify_prompt("A single beetle crosses the desert.", RULE)


def test_term_matching_does_not_use_substrings():
    assert not MODULE.classify_prompt("A woman wears a bowtie.", RULE)
