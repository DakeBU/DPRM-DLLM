#!/usr/bin/env python3
"""Validate the public DPRM release manifest and canonical results."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reproducibility" / "experiments.json"
RESULTS = ROOT / "results" / "paper_results.csv"
EXPECTED_HOSTS = {
    "PUMA",
    "DMPO",
    "Prism",
    "DPLM-2 Bit",
    "DCM",
    "GenMol V2",
    "SDPO",
    "Omni-Diffusion",
    "LLaDA-V",
}
TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".json", ".md", ".py", ".sh", ".tex",
    ".toml", ".txt", ".yaml", ".yml",
}
FORBIDDEN = (
    "/home/" + "nitanda_sub",
    "/var/tmp/" + "nitanda_sub",
    "T" + "PAMI_",
    "t" + "pami_",
    "DPRM_" + "DPRM",
    "ICML " + "2026",
    "<" + "PATH",
    "/path" + "/to/",
    "your-entity-" + "here",
)
REQUIRED_REPRO_FILES = (
    "integrations/dmpo/overlay/DMPO/dmpo_train_compat.py",
    "integrations/dmpo/overlay/transformers_compat.py",
    "integrations/dmpo/overlay/fast_samplers/fast_dllm/generate.py",
    "integrations/dmpo/scripts/package_release.py",
    "integrations/dcm/scripts/run_preference_sweep.sh",
    "integrations/dcm/scripts/run_terminal_calibration.sh",
    "integrations/genmol/scripts/run_preference_sweep.sh",
    "integrations/genmol/overlay/src/genmol/utils/utils_data.py",
    "reproducibility/scientific_preference_sweeps.json",
    "scripts/sync_scientific_results.py",
    "integrations/omni_diffusion/matched/run_pipeline.sh",
    "integrations/omni_diffusion/matched/package_release.sh",
    "integrations/omni_diffusion/matched/fit_paper_controller.sh",
    "integrations/omni_diffusion/matched/develop_public_base_fallback_controller.sh",
    "integrations/omni_diffusion/matched/develop_controller.sh",
    "integrations/omni_diffusion/matched/scripts/orchestrate_matched_pipeline.sh",
    "integrations/omni_diffusion/matched/scripts/train_matched_branches.sh",
    "integrations/omni_diffusion/matched/scripts/select_and_confirm_matched_endpoints.sh",
    "integrations/omni_diffusion/matched/scripts/select_omni_training_endpoint.py",
    "integrations/omni_diffusion/matched/scripts/freeze_partiprompts_split.py",
    "integrations/omni_diffusion/matched/scripts/evaluate_matched_branches.sh",
    "integrations/omni_diffusion/matched/scripts/audit_omni_training_contract.py",
    "integrations/omni_diffusion/matched/scripts/check_omni_matched_promotion.py",
    "integrations/omni_diffusion/matched/scripts/check_omni_fixed_visual_review.py",
    "integrations/omni_diffusion/matched/scripts/analyze_omni_multi_entity_subset.py",
    "integrations/omni_diffusion/matched/scripts/package_omni_formal_visual_audit.py",
    "integrations/omni_diffusion/matched/scripts/select_omni_stagewise_controller.py",
    "integrations/omni_diffusion/matched/scripts/summarize_omni_controller_sweep.py",
    "integrations/omni_diffusion/matched/scripts/validate_omni_visual_prompts.py",
    "integrations/omni_diffusion/matched/overlay/tools/trainer_v4_51_3.py",
    "reproducibility/omni_visual_prompts.json",
    "reproducibility/omni_visual_prompts_confirm512.json",
    "reproducibility/omni_multi_entity_prompt_rule.json",
    "scripts/paired_bootstrap.py",
    "scripts/bootstrap_passk.py",
    "integrations/puma/scripts/analyze_reveal_order.py",
    "integrations/puma/overlay/eval_dprm_checkpoint.py",
    "integrations/llada_v/scripts/summarize_multimodal_results.py",
    "integrations/llada_v/run_lmms_eval.sh",
    "integrations/llada_v/apply_overlay.sh",
    "integrations/llada_v/run_realworldqa_protocol.sh",
    "integrations/llada_v/run_ai2d_protocol.sh",
    "integrations/llada_v/scripts/package_ai2d_diagnostic.py",
    "results/artifacts/puma_reveal_order_summary.json",
    "src/dprm/omni_order.py",
)
LEGACY_SCHEMA_PATHS = {
    Path("integrations/omni_diffusion/matched/scripts/audit_omni_training_contract.py"),
    Path("tests/test_omni_training_contract.py"),
}

OMNI_RELEASED_CONFIG_TOKENS = (
    "step-1000 confidence-trained checkpoint",
    "0.70`, `0.85`,\n  `0.90`, and `0.95",
    "128 development prompts",
    "512 disjoint confirmation prompts",
    "fixed to `8`",
    "CLIP-B/32",
    "5,000 paired bootstrap",
)

SDPO_PAPER_DEFAULTS = {
    "integrations/sdpo/overlay/scripts/run_sdpo_dna_variant.sh": (
        "DPRM_PHASE_BINS=${DPRM_PHASE_BINS:-1}",
    ),
    "integrations/sdpo/overlay/scripts/run_sdpo_dna_eval_compare.sh": (
        "DPRM_PHASE_BINS=${DPRM_PHASE_BINS:-1}",
        '--dprm_phase_bins "${DPRM_PHASE_BINS}"',
    ),
    "integrations/sdpo/overlay/finetune_sdpo.py": (
        "--dprm_phase_bins', type=int, default=1",
    ),
    "integrations/sdpo/overlay/eval_dna_bootstrap.py": (
        '"--dprm_phase_bins", type=int, default=1',
    ),
}

PUMA_PAPER_CONFIGS = (
    "tinygsm_puma_dprm.yaml",
    "tinygsm_puma_dprm_random.yaml",
    "tinygsm_block_puma_dprm.yaml",
    "sudoku_puma_dprm.yaml",
)


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    for relative_path in REQUIRED_REPRO_FILES:
        if not (ROOT / relative_path).is_file():
            fail(f"missing reproduction file: {relative_path}")

    omni_readme = (ROOT / "integrations/omni_diffusion/README.md").read_text(
        encoding="utf-8"
    )
    for token in OMNI_RELEASED_CONFIG_TOKENS:
        if token not in omni_readme:
            fail(f"Omni README is missing released configuration text: {token!r}")

    for relative_path, tokens in SDPO_PAPER_DEFAULTS.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                fail(f"SDPO paper configuration drift in {relative_path}: {token!r}")

    for name in PUMA_PAPER_CONFIGS:
        text = (
            ROOT / "integrations/puma/overlay/yaml_files" / name
        ).read_text(encoding="utf-8")
        if "sampled_shortlist: true" not in text:
            fail(f"PUMA paper configuration drift in {name}: sampled shortlist disabled")

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    if len(experiments) != 9:
        fail(f"expected 9 experiments, found {len(experiments)}")

    ids = [row["id"] for row in experiments]
    if len(ids) != len(set(ids)):
        fail("experiment ids are not unique")
    hosts = {row["host"] for row in experiments}
    if hosts != EXPECTED_HOSTS:
        fail(f"registry hosts differ: {sorted(hosts ^ EXPECTED_HOSTS)}")

    registered_result_variants = {}
    reported_variant_parents = {}
    result_variant_parents = {}
    for experiment in experiments:
        commit = experiment.get("upstream_commit", "")
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            fail(f"{experiment['id']} has invalid upstream_commit")
        for field in ("evaluation_unit", "statistics_command"):
            if not experiment.get(field, "").strip():
                fail(f"{experiment['id']} has no {field}")
        variants = experiment.get("variants", [])
        if experiment.get("execution_root") not in {"host", "release"}:
            fail(f"{experiment['id']} has invalid execution_root")
        working_subdir = experiment.get("working_subdir", "")
        if not working_subdir or Path(working_subdir).is_absolute() or ".." in Path(working_subdir).parts:
            fail(f"{experiment['id']} has invalid working_subdir")
        if len(variants) != 4:
            fail(f"{experiment['id']} has {len(variants)} variants, expected 4")
        variant_ids = [row["id"] for row in variants]
        if len(variant_ids) != len(set(variant_ids)):
            fail(f"{experiment['id']} has duplicate variant ids")
        result_variants = experiment.get("result_variants", {})
        if not isinstance(result_variants, dict):
            fail(f"{experiment['id']} result_variants must map labels to parents")
        unknown_parents = set(result_variants.values()) - set(variant_ids)
        if unknown_parents:
            fail(
                f"{experiment['id']} result variants have unknown parents: "
                f"{sorted(unknown_parents)}"
            )
        result_variant_ids = variant_ids + list(result_variants)
        if len(result_variant_ids) != len(set(result_variant_ids)):
            fail(f"{experiment['id']} has duplicate result variant ids")
        registered_result_variants[experiment["host"]] = set(result_variant_ids)
        result_variant_parents[experiment["host"]] = {
            variant_id: variant_id for variant_id in variant_ids
        } | result_variants
        reported_variant_parents[experiment["host"]] = {
            variant["id"]
            for variant in variants
            if variant["status"].startswith("reported")
        }
        for field in ("integration_readme", "entrypoint", "result_file"):
            path = ROOT / experiment[field]
            if not path.is_file():
                fail(f"{experiment['id']} missing {field}: {path}")
        readme_text = (ROOT / experiment["integration_readme"]).read_text(
            encoding="utf-8"
        )
        if "reproducibility/experiments.json" not in readme_text:
            fail(f"{experiment['id']} README does not link the command registry")
        if not re.search(r"^## Paper Configuration\s*$", readme_text, re.M):
            fail(f"{experiment['id']} README has no Paper Configuration section")
        if not re.search(r"bootstrap|confidence interval|uncertainty", readme_text, re.I):
            fail(f"{experiment['id']} README does not describe uncertainty output")
        for variant in variants:
            if not variant.get("command", "").strip():
                fail(f"{experiment['id']}/{variant['id']} has no command")
            execution_root = variant.get("execution_root", experiment["execution_root"])
            if execution_root not in {"host", "release"}:
                fail(
                    f"{experiment['id']}/{variant['id']} has invalid execution_root"
                )
            variant_subdir = variant.get("working_subdir", working_subdir)
            if (
                not variant_subdir
                or Path(variant_subdir).is_absolute()
                or ".." in Path(variant_subdir).parts
            ):
                fail(
                    f"{experiment['id']}/{variant['id']} has invalid working_subdir"
                )
            if variant.get("status") not in {
                "reported",
                "reported_control",
                "reported_ai2d",
                "reported_development_gate",
                "formal_pending",
                "formal_not_promoted",
                "implemented_control",
            }:
                fail(f"{experiment['id']}/{variant['id']} has invalid status")

    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result_hosts = {row["host"] for row in rows}
    reported_hosts = {
        experiment["host"]
        for experiment in experiments
        if any(variant["status"].startswith("reported") for variant in experiment["variants"])
    }
    if not reported_hosts.issubset(result_hosts):
        fail(f"results missing reported hosts: {sorted(reported_hosts - result_hosts)}")
    represented_parents = {
        host: {
            result_variant_parents[host][row["variant"]]
            for row in rows
            if row["host"] == host
        }
        for host in registered_result_variants
    }
    for host, required_parents in reported_variant_parents.items():
        missing = required_parents - represented_parents[host]
        if missing:
            fail(f"results missing reported variants for {host}: {sorted(missing)}")
    for row_number, row in enumerate(rows, start=2):
        if row["variant"] not in registered_result_variants[row["host"]]:
            fail(
                f"unregistered result variant on CSV line {row_number}: "
                f"{row['host']}/{row['variant']}"
            )
        try:
            float(row["value"])
        except ValueError as error:
            fail(f"invalid result value on CSV line {row_number}: {error}")
        if row["direction"] not in {"higher", "lower"}:
            fail(f"invalid metric direction on CSV line {row_number}")

    violations = []
    scan_roots = [
        ROOT / "README.md",
        ROOT / "src",
        ROOT / "integrations",
        ROOT / "reproducibility",
        ROOT / "results",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    for scan_root in scan_roots:
        paths = [scan_root] if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in FORBIDDEN:
                relative_path = path.relative_to(ROOT)
                if token == "t" + "pami_" and relative_path in LEGACY_SCHEMA_PATHS:
                    continue
                if token in text:
                    violations.append(f"{relative_path}: {token}")
    if violations:
        fail("release text contains private/history residue:\n  " + "\n  ".join(violations))

    print(
        "release audit passed: "
        f"9 hosts, 36 variants, {len(reported_hosts)} hosts with formal results, "
        f"{len(rows)} result rows"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, json.JSONDecodeError) as error:
        print(f"release audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
