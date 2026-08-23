from __future__ import annotations

import sys
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_OMNI_ROOT = Path(os.environ.get("OMNI_ROOT", ""))
HAS_UPSTREAM_OMNI = bool(os.environ.get("OMNI_ROOT")) and (
    UPSTREAM_OMNI_ROOT / "omni_diffusion"
).is_dir()
sys.path.insert(0, str(REPO_ROOT / "src"))

from dprm.omni_order import (
    OMNI_SINGLE_PATH_REPEAT_PENALTY,
    OmniBucketTableDPRM,
    OmniOrderConfig,
    OmniOrderScorer,
    OmniRankBucketDPRM,
    OmniStageRankCodeDPRM,
    adjusted_order_scores,
    build_action_features,
    entropy_penalty_order_scores,
    negative_entropy,
    candidate_visual_indices,
    load_omni_order_controller,
    visual_candidate_mask,
)


def test_stage_rank_code_controller_uses_provisional_visual_code():
    values = [[[0.0 for _ in range(4)] for _ in range(64)]]
    counts = [[[0 for _ in range(4)] for _ in range(64)]]
    values[0][57][3] = 0.5
    counts[0][57][3] = 10
    controller = OmniStageRankCodeDPRM(
        active_steps=(64,),
        rank_bins=64,
        code_bins=4,
        reward_values=tuple(tuple(tuple(row) for row in stage) for stage in values),
        counts=tuple(tuple(tuple(row) for row in stage) for stage in counts),
        beta=2.0,
        min_count=8,
    )
    confidence = torch.linspace(-2.0, 0.0, 100)
    token_ids = torch.full((100,), 168072, dtype=torch.long)
    # The rank-57 cell spans candidates near the 0.90 quantile.
    token_ids[89:91] = 168072 + 7000
    adjusted, reward = controller.score(
        confidence,
        step=64,
        provisional_token_ids=token_ids,
    )
    assert reward.max().item() == pytest.approx(0.5)
    assert torch.any(adjusted > confidence)
    unchanged, zero = controller.score(
        confidence,
        step=63,
        provisional_token_ids=token_ids,
    )
    assert torch.equal(unchanged, confidence)
    assert torch.count_nonzero(zero) == 0


def test_prompt_text_deduplication_preserves_one_record_per_order():
    import importlib.util

    script = (
        Path(__file__).parents[1]
        / "integrations"
        / "omni_diffusion"
        / "matched"
        / "scripts"
        / "build_omni_dprm_table.py"
    )
    spec = importlib.util.spec_from_file_location("build_omni_table", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    records = [
        {"prompt": "same", "order": "random", "prompt_id": "prompt_1"},
        {"prompt": "same", "order": "progressive_confidence", "prompt_id": "prompt_1"},
        {"prompt": "same", "order": "random", "prompt_id": "prompt_2"},
        {"prompt": "same", "order": "progressive_confidence", "prompt_id": "prompt_2"},
        {"prompt": "new", "order": "random", "prompt_id": "prompt_3"},
        {"prompt": "new", "order": "progressive_confidence", "prompt_id": "prompt_3"},
    ]
    deduplicated = module.deduplicate_prompt_text(records)
    assert [(row["prompt"], row["order"]) for row in deduplicated] == [
        ("same", "random"),
        ("same", "progressive_confidence"),
        ("new", "random"),
        ("new", "progressive_confidence"),
    ]


def test_table_builder_can_credit_only_deployed_action_steps():
    import importlib.util

    script = (
        Path(__file__).parents[1]
        / "integrations"
        / "omni_diffusion"
        / "matched"
        / "scripts"
        / "build_omni_dprm_table.py"
    )
    spec = importlib.util.spec_from_file_location("build_omni_table_active", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [{"step": 64}, {"step": 96}, {"step": 128}]
    assert module.active_trace_rows(rows, set()) is rows
    assert module.active_trace_rows(rows, {96, 128}) == rows[1:]
    assert [module.spatial_bin(index, 4) for index in (0, 15, 128, 255)] == [
        0,
        1,
        2,
        3,
    ]
    assert module.spatial_bin(68, 16) == 5


def test_dual_clip_terminal_utility_normalizes_each_encoder_separately():
    import importlib.util

    script = (
        Path(__file__).parents[1]
        / "integrations"
        / "omni_diffusion"
        / "matched"
        / "scripts"
        / "build_omni_dprm_table.py"
    )
    spec = importlib.util.spec_from_file_location("build_omni_table_dual", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    records = [
        {
            "prompt_id": "p0",
            "order": "random",
            "trace_path": "p0-r",
            "clip_cosine": 0.20,
            "clip_b32_cosine": 0.30,
        },
        {
            "prompt_id": "p0",
            "order": "confidence",
            "trace_path": "p0-c",
            "clip_cosine": 0.24,
            "clip_b32_cosine": 0.28,
        },
        {
            "prompt_id": "p1",
            "order": "random",
            "trace_path": "p1-r",
            "clip_cosine": 0.31,
            "clip_b32_cosine": 0.21,
        },
        {
            "prompt_id": "p1",
            "order": "confidence",
            "trace_path": "p1-c",
            "clip_cosine": 0.29,
            "clip_b32_cosine": 0.27,
        },
    ]
    rewards, stats = module.normalized_rewards(
        records,
        "paired_prompt_advantage",
        {"clip_cosine": 0.25, "clip_b32_cosine": 0.75},
    )
    assert set(rewards) == {"p0-r", "p0-c", "p1-r", "p1-c"}
    assert sum(rewards.values()) == pytest.approx(0.0)
    assert stats["metric_weights"] == {
        "clip_cosine": 0.25,
        "clip_b32_cosine": 0.75,
    }
    assert records[0]["dprm_reward"] == pytest.approx(rewards["p0-r"])
    assert set(records[0]["dprm_reward_components"]) == {
        "clip_cosine",
        "clip_b32_cosine",
    }


def test_negative_entropy_prefers_peaked_distribution() -> None:
    peaked = torch.tensor([[8.0, 0.0, 0.0]])
    flat = torch.tensor([[0.0, 0.0, 0.0]])
    scores = negative_entropy(torch.cat([peaked, flat], dim=0))
    assert scores[0] > scores[1]


def test_negative_entropy_matches_upstream_bfloat16_sampler() -> None:
    torch.manual_seed(17)
    logits = torch.randn(7, 31, dtype=torch.bfloat16)
    probabilities = torch.softmax(logits, dim=-1)
    upstream = torch.sum(
        probabilities * torch.log(probabilities + 1e-10), dim=-1
    )
    actual = negative_entropy(logits)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, upstream, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_UPSTREAM_OMNI, reason="requires an Omni-Diffusion checkout")
def test_training_entropy_penalty_matches_omni_sampler() -> None:
    omni_root = UPSTREAM_OMNI_ROOT
    sys.path.insert(0, str(omni_root))
    from omni_diffusion.models.dream.generation_utils import sample_tokens

    torch.manual_seed(23)
    logits = torch.randn(127, 211, dtype=torch.float32)
    past = torch.tensor([0, 151666, 3, 9, 47, 9])
    expected_scores, expected_tokens = sample_tokens(
        logits.clone(),
        temperature=0.0,
        top_p=0.9,
        neg_entropy=True,
        repeat_penalty=1.2,
        max_position_penalty=2.0,
        past_x=past,
        mask_id=151666,
    )
    actual_scores, actual_tokens = entropy_penalty_order_scores(
        logits,
        top_p=0.9,
        repeat_penalty=1.2,
        max_position_penalty=2.0,
        past_tokens=past,
        mask_id=151666,
    )
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_UPSTREAM_OMNI, reason="requires an Omni-Diffusion checkout")
def test_single_path_training_score_matches_deployed_omni_without_history() -> None:
    omni_root = UPSTREAM_OMNI_ROOT
    sys.path.insert(0, str(omni_root))
    from omni_diffusion.models.dream.generation_utils import sample_tokens

    torch.manual_seed(29)
    logits = torch.randn(256, 211, dtype=torch.float32)
    # Upstream uses repeat_penalty=1.0 whenever histories is None. That is the
    # deployed path because return_dict_in_generate is false.
    expected_scores, expected_tokens = sample_tokens(
        logits.clone(),
        temperature=0.0,
        top_p=0.9,
        neg_entropy=True,
        repeat_penalty=OMNI_SINGLE_PATH_REPEAT_PENALTY,
        past_x=[],
        mask_id=151666,
        max_position_penalty=2.0,
    )
    actual_scores, actual_tokens = entropy_penalty_order_scores(
        logits,
        top_p=0.9,
        repeat_penalty=OMNI_SINGLE_PATH_REPEAT_PENALTY,
        max_position_penalty=2.0,
        past_tokens=torch.tensor([3, 9, 47, 9]),
        mask_id=151666,
    )
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)


def test_omni_trainer_uses_single_path_repetition_contract() -> None:
    trainer = (
        REPO_ROOT
        / "integrations"
        / "omni_diffusion"
        / "matched"
        / "overlay"
        / "tools"
        / "trainer_v4_51_3.py"
    ).read_text(encoding="utf-8")
    assert "repeat_penalty=OMNI_SINGLE_PATH_REPEAT_PENALTY" in trainer
    assert "OMNI_SINGLE_PATH_REPEAT_PENALTY," in trainer


@pytest.mark.skipif(not HAS_UPSTREAM_OMNI, reason="requires an Omni-Diffusion checkout")
def test_fixed_t2i_scaffold_leaves_visual_canvas_masked() -> None:
    import importlib.util

    script = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "omni_t2i_smoke.py"
    spec = importlib.util.spec_from_file_location("omni_smoke", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class Tokenizer:
        ids = {
            "<|begin_of_image|>": 151667,
            "<|end_of_image|>": 151668,
            "<|im_end|>": 151645,
        }

        def encode(self, text, add_special_tokens=False):
            assert text == "\n" and not add_special_tokens
            return [198]

        def convert_tokens_to_ids(self, token):
            return self.ids[token]

    prompt_length = 19
    mask_id = 151666
    x = torch.full((1, prompt_length + 260), mask_id)
    hook = module.make_fixed_t2i_scaffold_hook(Tokenizer(), prompt_length)
    actual = hook(None, x, None)
    assert actual[0, prompt_length : prompt_length + 260].tolist()[:1] == [151667]
    assert actual[0, prompt_length + 257 : prompt_length + 260].tolist() == [151668, 151645, 198]
    assert torch.all(actual[0, prompt_length + 1 : prompt_length + 257] == mask_id)


def test_shared_action_features_are_deterministic_and_well_shaped() -> None:
    cfg = OmniOrderConfig()
    confidence = torch.full((6,), -torch.inf)
    confidence[torch.tensor([1, 3, 5])] = torch.tensor([-0.9, -0.3, -0.6])
    candidates = torch.tensor([1, 3, 5])
    visual_indices = torch.tensor([0, 120, 254])
    masked = torch.ones(256, dtype=torch.bool)
    masked[10] = False
    provisional = torch.full((6,), cfg.image_token_offset + 100)

    left = build_action_features(
        confidence=confidence,
        candidate_indices=candidates,
        visual_indices=visual_indices,
        masked_visual=masked,
        provisional_token_ids=provisional,
        step=96,
        config=cfg,
    )
    right = build_action_features(
        confidence=confidence,
        candidate_indices=candidates,
        visual_indices=visual_indices,
        masked_visual=masked,
        provisional_token_ids=provisional,
        step=96,
        config=cfg,
    )
    assert left.shape == (3, cfg.feature_dim)
    torch.testing.assert_close(left, right)


def test_adjusted_scores_change_only_candidates() -> None:
    cfg = OmniOrderConfig(hidden_size=8)
    scorer = OmniOrderScorer(cfg)
    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.zero_()
        scorer.network[-1].bias.fill_(0.25)
    confidence = torch.tensor([-0.8, -0.4, -0.2])
    candidates = torch.tensor([0, 2])
    features = torch.zeros((2, cfg.feature_dim))
    adjusted, advantages = adjusted_order_scores(
        confidence=confidence,
        candidate_indices=candidates,
        features=features,
        scorer=scorer,
        guidance_scale=2.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(adjusted, torch.tensor([-0.3, -0.4, 0.3]))


def test_scorer_artifact_round_trip(tmp_path) -> None:
    cfg = OmniOrderConfig(hidden_size=8)
    scorer = OmniOrderScorer(cfg)
    artifact = tmp_path / "scorer.pt"
    scorer.save_artifact(artifact, metadata={"split": "development"})
    loaded, metadata = OmniOrderScorer.load_artifact(artifact)
    features = torch.randn(4, cfg.feature_dim)
    torch.testing.assert_close(scorer(features), loaded(features))
    assert metadata == {"split": "development"}


def test_rank_bucket_dprm_is_stage_gated(tmp_path) -> None:
    controller = OmniRankBucketDPRM(
        active_step=96,
        rank_bins=8,
        target_rank_bin=6,
        reward_value=0.004,
        beta=100.0,
    )
    confidence = torch.linspace(-4.0, -3.0, 16)
    inactive, _ = controller.score(confidence, step=95)
    active, reward = controller.score(confidence, step=96)
    torch.testing.assert_close(inactive, confidence)
    assert torch.any(active != confidence)
    assert torch.all(reward[active != confidence] == 0.004)
    artifact = tmp_path / "rank_bucket.json"
    controller.save_artifact(artifact, {"development_split": "offset-2000"})
    loaded, metadata = OmniRankBucketDPRM.load_artifact(artifact)
    assert loaded == controller
    assert metadata["development_split"] == "offset-2000"


def test_visual_candidate_mapping_includes_all_256_codes() -> None:
    # Block-relative positions: BOI=0, visual=1..256, EOI=257.
    block_mask = torch.zeros(300, dtype=torch.bool)
    block_mask[20:280] = True
    mask_index = torch.zeros((1, 300), dtype=torch.bool)
    mask_index[0, 20:280] = True
    visual = visual_candidate_mask(mask_index, block_mask)
    indices = candidate_visual_indices(mask_index, block_mask)
    assert int(visual.sum()) == 256
    assert indices[visual].tolist() == list(range(256))


def test_bucket_table_round_trip_and_train_inference_gate(tmp_path) -> None:
    counts = (((8.0, 0.0), (4.0, 8.0)),)
    exp_reward_sums = (((16.0, 0.0), (4.0, 32.0)),)
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=2,
        spatial_bins=2,
        reward_temperature=1.0,
        guidance_scale=0.5,
        warmup_steps=0,
        switch_steps=4,
        ready_count=8,
        counts=counts,
        exp_reward_sums=exp_reward_sums,
        policy_warmup_steps=2,
        total_steps=8,
    )
    artifact = tmp_path / "bucket_table.json"
    controller.save_artifact(artifact, {"split": "development"})
    loaded, metadata = load_omni_order_controller(artifact)
    assert loaded == controller
    assert metadata == {"split": "development"}

    confidence = torch.log(torch.tensor([0.25, 0.75, 0.75, 0.25]))
    visual_indices = torch.tensor([0, 64, 192, 255])
    warmup_scores, warmup_values = loaded.score(
        confidence, step=1, visual_indices=visual_indices
    )
    torch.testing.assert_close(warmup_scores, confidence)
    torch.testing.assert_close(warmup_values, torch.zeros_like(warmup_values))

    adjusted, values = loaded.score(confidence, step=4, visual_indices=visual_indices)
    assert adjusted[0] > confidence[0]
    assert adjusted[2] > confidence[2]
    assert adjusted[1] == confidence[1]  # Empty bucket falls back to confidence.
    assert values[0] > 0
    assert values[2] > values[0]


def test_bucket_table_uses_frozen_quantile_edges(tmp_path) -> None:
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=3,
        spatial_bins=1,
        reward_temperature=1.0,
        guidance_scale=1.0,
        warmup_steps=0,
        switch_steps=0,
        ready_count=1,
        counts=(((1.0,), (1.0,), (1.0,)),),
        exp_reward_sums=(((1.0,), (2.0,), (4.0,)),),
        confidence_bin_edges=(0.02, 0.08),
        policy_warmup_steps=0,
    )
    confidence = torch.log(torch.tensor([0.01, 0.02, 0.10]))
    adjusted, values = controller.score(
        confidence, step=1, visual_indices=torch.tensor([0, 1, 2])
    )
    torch.testing.assert_close(values, torch.log(torch.tensor([1.0, 2.0, 4.0])))
    assert adjusted[2] > adjusted[1] > adjusted[0]

    artifact = tmp_path / "quantile_bucket.json"
    controller.save_artifact(artifact)
    loaded, _ = load_omni_order_controller(artifact)
    assert loaded.confidence_bin_edges == (0.02, 0.08)


def test_bucket_table_policy_warmup_and_linear_gate_follow_action_index() -> None:
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=1,
        spatial_bins=1,
        reward_temperature=1.0,
        guidance_scale=1.0,
        warmup_steps=0,
        switch_steps=64,
        ready_count=1,
        counts=(((1.0,),),),
        exp_reward_sums=(((torch.exp(torch.tensor(1.0)).item(),),),),
        policy_warmup_steps=32,
        total_steps=260,
    )
    confidence = torch.tensor([-2.0])
    visual_indices = torch.tensor([17])

    score31, value31 = controller.score(
        confidence, step=31, visual_indices=visual_indices
    )
    score32, value32 = controller.score(
        confidence, step=32, visual_indices=visual_indices
    )
    score64, value64 = controller.score(
        confidence, step=64, visual_indices=visual_indices
    )

    torch.testing.assert_close(score31, confidence)
    torch.testing.assert_close(value31, torch.zeros_like(value31))
    torch.testing.assert_close(value32, torch.tensor([0.5]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(value64, torch.tensor([1.0]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(score32 - confidence, value32, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(score64 - confidence, value64, rtol=1e-6, atol=1e-6)


def test_bucket_table_limits_reward_to_fixed_actions_and_ambiguous_candidates() -> None:
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=1,
        spatial_bins=3,
        reward_temperature=1.0,
        guidance_scale=1.0,
        warmup_steps=0,
        switch_steps=0,
        ready_count=1,
        counts=(((1.0, 1.0, 1.0),),),
        exp_reward_sums=(((1.0, 4.0, 20.0),),),
        policy_warmup_steps=0,
        reward_action_steps=(64, 96),
        max_base_score_gap=0.05,
    )
    confidence = torch.tensor([-0.10, -0.13, -0.40])
    visual_indices = torch.tensor([0, 96, 240])
    inactive_scores, inactive_values = controller.score(
        confidence, step=65, visual_indices=visual_indices
    )
    torch.testing.assert_close(inactive_scores, confidence)
    torch.testing.assert_close(inactive_values, torch.zeros_like(inactive_values))
    active_scores, active_values = controller.score(
        confidence, step=64, visual_indices=visual_indices
    )
    assert active_scores[1] > active_scores[0]
    assert active_values[1] > 0
    assert active_values[2] == 0


def test_bucket_table_can_limit_reward_to_low_confidence_bins() -> None:
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=2,
        spatial_bins=1,
        reward_temperature=1.0,
        guidance_scale=1.0,
        warmup_steps=0,
        switch_steps=0,
        ready_count=1,
        counts=(((1.0,), (1.0,)),),
        exp_reward_sums=(((4.0,), (4.0,)),),
        confidence_bin_edges=(0.5,),
        policy_warmup_steps=0,
        max_reward_confidence_bin=0,
    )
    confidence = torch.log(torch.tensor([0.25, 0.75]))
    adjusted, values = controller.score(
        confidence, step=1, visual_indices=torch.tensor([0, 1])
    )
    assert adjusted[0] > confidence[0]
    assert values[0] > 0
    assert adjusted[1] == confidence[1]
    assert values[1] == 0


def test_bucket_table_selects_same_visual_action_in_train_and_inference_layouts() -> None:
    counts = tuple(
        tuple(tuple(8.0 for _ in range(16)) for _ in range(3)) for _ in range(1)
    )
    exp_reward_sums = tuple(
        tuple(
            tuple(8.0 * (1.0 + 0.1 * conf + 0.03 * spatial) for spatial in range(16))
            for conf in range(3)
        )
        for _ in range(1)
    )
    controller = OmniBucketTableDPRM(
        num_phases=1,
        confidence_bins=3,
        spatial_bins=16,
        reward_temperature=1.0,
        guidance_scale=0.2,
        warmup_steps=0,
        switch_steps=0,
        ready_count=1,
        counts=counts,
        exp_reward_sums=exp_reward_sums,
        confidence_bin_edges=(0.03, 0.06),
        policy_warmup_steps=0,
    )
    candidate_visual = torch.tensor([0, 17, 94, 173, 255])
    # Both paths start from the same host logits and therefore the same official
    # Omni negative-entropy order score, not a max-token-probability proxy.
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [4.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    candidate_confidence = negative_entropy(logits)

    # Training scores the compact list of currently masked visual candidates.
    train_scores, train_values = controller.score(
        candidate_confidence, step=96, visual_indices=candidate_visual
    )
    train_selected = candidate_visual[torch.argmax(train_scores)]

    # Inference scores the same candidates in a longer sequence-position layout.
    sequence_positions = torch.tensor([11, 28, 105, 184, 266])
    inference_confidence = torch.full((300,), -torch.inf)
    inference_visual_indices = torch.full((300,), -1)
    inference_confidence[sequence_positions] = candidate_confidence
    inference_visual_indices[sequence_positions] = candidate_visual
    inference_scores, inference_values = controller.score(
        inference_confidence, step=96, visual_indices=inference_visual_indices
    )
    inference_selected = inference_visual_indices[torch.argmax(inference_scores)]

    torch.testing.assert_close(inference_scores[sequence_positions], train_scores)
    torch.testing.assert_close(inference_values[sequence_positions], train_values)
    assert inference_selected.item() == train_selected.item()


def test_fixed_canvas_contract_survives_controller_freeze(tmp_path) -> None:
    counts = [[[8.0 for _ in range(16)] for _ in range(8)]]
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.json"
    decision = tmp_path / "decision.json"
    formal = tmp_path / "formal.json"
    source.write_text(
        json.dumps(
            {
                "cfg": {
                    "num_phases": 1,
                    "confidence_bins": 8,
                    "aux_bins": 16,
                    "reward_temperature": 1.0,
                    "warmup_steps": 0,
                    "switch_steps": 64,
                    "confidence_bin_edges": [0.01 * i for i in range(1, 8)],
                    "base_order_score": "negative_token_entropy",
                    "bucket_coordinate": "exp_negative_token_entropy",
                },
                "counts": counts,
                "exp_reward_sums": counts,
                "metadata": {
                    "prompt_text_deduplicated": True,
                    "fixed_visual_canvas": True,
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "freeze_omni_bucket_controller.py"),
            "--source-table",
            str(source),
            "--output",
            str(candidate),
            "--guidance-scale",
            "0.01",
            "--ready-count",
            "4",
            "--reward-action-steps",
            "64",
            "96",
            "--max-base-score-gap",
            "0.05",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    decision.write_text(
        json.dumps(
            {
                "design": "disjoint development selection",
                "passed": True,
                "selection_metric": "mean paired CLIP-L/14 delta",
                "selected": "candidate_a",
                "candidates": {
                    "candidate_a": {
                        "config": candidate_payload["config"],
                        "metrics": {"mean_delta_vs_confidence": 0.01},
                        "interventions": {"prompt_fraction_with_override": 0.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "prepare_omni_formal_controller.py"),
            "--input",
            str(candidate),
            "--selection-decision",
            str(decision),
            "--table-prompt-range",
            "2000",
            "2047",
            "--selection-prompt-range",
            "2100",
            "2131",
            "--output",
            str(formal),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(formal.read_text(encoding="utf-8"))
    source_summary = payload["metadata"]["source_summary"]
    deployment = payload["metadata"]["deployment_contract"]
    match = payload["metadata"]["train_test_order_match"]
    assert source_summary["fixed_visual_canvas"] is True
    assert payload["metadata"]["source_prompt_text_sha256"] == []
    assert deployment == {
        "paths_per_prompt": 1,
        "positions_per_order_action": 1,
        "terminal_reward_calls_at_test": 0,
        "complete_image_selection": False,
        "fixed_t2i_scaffold": True,
        "ordered_visual_positions": 256,
    }
    assert match["training_state_policy"] == "same_frozen_bucket_controller"
    assert match["inference_policy"] == "same_frozen_bucket_controller"
    assert payload["metadata"]["stagewise_order_contract"] == {
        "reward_action_steps": [64, 96],
        "max_base_score_gap": 0.05,
        "max_reward_confidence_bin": None,
        "fallback": "native confidence order",
    }
    assert payload["metadata"]["development_selection"] == {
        "design": "disjoint development selection",
        "selection_metric": "mean paired CLIP-L/14 delta",
        "selected_label": "candidate_a",
        "candidate_count": 1,
        "table_prompt_range": [2000, 2047],
        "selection_prompt_range": [2100, 2131],
        "selection_prompt_file": None,
        "selection_prompt_file_sha256": None,
        "selection_prompt_text_sha256": None,
        "selected_metrics": {"mean_delta_vs_confidence": 0.01},
        "selected_interventions": {"prompt_fraction_with_override": 0.5},
    }
