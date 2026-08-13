"""Shared visual-token feature contract for train/test-aligned Omni DPRM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


# Omni's single-path T2I deployment has return_dict_in_generate=False, so the
# upstream sampler does not retain histories and applies no repetition penalty.
OMNI_SINGLE_PATH_REPEAT_PENALTY = 1.0


@dataclass(frozen=True)
class OmniOrderConfig:
    total_steps: int = 260
    num_phases: int = 8
    rank_bins: int = 8
    spatial_bins: int = 16
    image_side: int = 16
    image_token_offset: int = 168072
    image_vocab_size: int = 8192
    hidden_size: int = 64

    @property
    def feature_dim(self) -> int:
        # Ten continuous features plus phase/rank/spatial/local/code one-hot blocks.
        return 10 + self.num_phases + self.rank_bins + self.spatial_bins + 3 + 16


class OmniOrderScorer(nn.Module):
    """Small action-value network fitted to terminal-reward advantages."""

    def __init__(self, config: OmniOrderConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def save_artifact(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "format": "omni_dprm_order_scorer_v1",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["OmniOrderScorer", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        if payload.get("format") != "omni_dprm_order_scorer_v1":
            raise ValueError(f"unsupported Omni DPRM scorer artifact: {path}")
        model = cls(OmniOrderConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload.get("metadata", {}))


@dataclass(frozen=True)
class OmniRankBucketDPRM:
    """A stage/rank DPRM table learned from action-conditioned terminal utility."""

    active_step: int
    rank_bins: int
    target_rank_bin: int
    reward_value: float
    beta: float
    total_steps: int = 260

    def save_artifact(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format": "omni_rank_bucket_dprm_v1",
                    "config": asdict(self),
                    "metadata": dict(metadata or {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_artifact(cls, path: str | Path) -> tuple["OmniRankBucketDPRM", dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "omni_rank_bucket_dprm_v1":
            raise ValueError(f"unsupported Omni rank-bucket artifact: {path}")
        return cls(**payload["config"]), dict(payload.get("metadata", {}))

    def score(self, confidence: torch.Tensor, *, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        adjusted = confidence.clone()
        reward = torch.zeros_like(confidence, dtype=torch.float32)
        if int(step) != int(self.active_step):
            return adjusted, reward
        candidates = torch.where(torch.isfinite(confidence))[0]
        if candidates.numel() < 2:
            return adjusted, reward
        _, rank_ids = relative_rank(confidence.float(), candidates, self.rank_bins)
        active = candidates[rank_ids[candidates] == int(self.target_rank_bin)]
        reward[active] = float(self.reward_value)
        adjusted[active] = adjusted[active] + float(self.beta * self.reward_value)
        return adjusted, reward


@dataclass(frozen=True)
class OmniStageRankSpatialDPRM:
    """Frozen multi-stage visual-order controller learned on development rollouts.

    The table indexes exact intervention steps, within-canvas confidence rank,
    and a coarse spatial region. Values are centered log-moment terminal
    utilities. Inference performs one host forward and one table lookup per
    candidate; it does not evaluate terminal rewards or completed alternatives.
    """

    active_steps: tuple[int, ...]
    rank_bins: int
    spatial_bins: int
    reward_values: tuple[tuple[tuple[float, ...], ...], ...]
    counts: tuple[tuple[tuple[int, ...], ...], ...]
    beta: float
    min_count: int = 1
    total_steps: int = 260

    def __post_init__(self) -> None:
        if not self.active_steps:
            raise ValueError("active_steps must not be empty")
        expected = (len(self.active_steps), int(self.rank_bins), int(self.spatial_bins))
        value_shape = _nested_shape(self.reward_values)
        count_shape = _nested_shape(self.counts)
        if value_shape != expected or count_shape != expected:
            raise ValueError(
                f"stage/rank/spatial table shape mismatch: expected {expected}, "
                f"values {value_shape}, counts {count_shape}"
            )

    def save_artifact(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        config = asdict(self)
        config["active_steps"] = list(self.active_steps)
        Path(path).write_text(
            json.dumps(
                {
                    "format": "omni_stage_rank_spatial_dprm_v1",
                    "config": config,
                    "metadata": dict(metadata or {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_artifact(
        cls, path: str | Path
    ) -> tuple["OmniStageRankSpatialDPRM", dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "omni_stage_rank_spatial_dprm_v1":
            raise ValueError(f"unsupported Omni stage/rank/spatial artifact: {path}")
        config = dict(payload["config"])
        config["active_steps"] = tuple(int(step) for step in config["active_steps"])
        config["reward_values"] = _nested_tuple(config["reward_values"], float)
        config["counts"] = _nested_tuple(config["counts"], int)
        return cls(**config), dict(payload.get("metadata", {}))

    def score(
        self,
        confidence: torch.Tensor,
        *,
        step: int,
        visual_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adjusted = confidence.clone()
        reward = torch.zeros_like(confidence, dtype=torch.float32)
        try:
            stage = self.active_steps.index(int(step))
        except ValueError:
            return adjusted, reward
        candidates = torch.where(torch.isfinite(confidence))[0]
        if candidates.numel() < 2:
            return adjusted, reward
        visual_indices = visual_indices.to(confidence.device).long()
        if visual_indices.shape != confidence.shape:
            raise ValueError("visual_indices must match confidence shape")
        _, rank_ids = relative_rank(confidence.float(), candidates, self.rank_bins)
        spatial_ids = spatial_bin_ids(
            visual_indices, spatial_bins=self.spatial_bins, image_side=16
        )
        values = torch.tensor(self.reward_values[stage], device=confidence.device)
        counts = torch.tensor(self.counts[stage], device=confidence.device)
        candidate_ranks = rank_ids[candidates]
        candidate_spatial = spatial_ids[candidates]
        candidate_counts = counts[candidate_ranks, candidate_spatial]
        candidate_values = values[candidate_ranks, candidate_spatial]
        ready = candidate_counts >= int(self.min_count)
        candidate_values = torch.where(ready, candidate_values, torch.zeros_like(candidate_values))
        reward[candidates] = candidate_values
        adjusted[candidates] = adjusted[candidates] + float(self.beta) * candidate_values.to(
            adjusted.dtype
        )
        return adjusted, reward


@dataclass(frozen=True)
class OmniBucketTableDPRM:
    """Frozen phase/confidence/spatial DPRM used throughout Omni decoding."""

    num_phases: int
    confidence_bins: int
    spatial_bins: int
    reward_temperature: float
    guidance_scale: float
    warmup_steps: int
    switch_steps: int
    ready_count: int
    counts: tuple[tuple[tuple[float, ...], ...], ...]
    exp_reward_sums: tuple[tuple[tuple[float, ...], ...], ...]
    confidence_bin_edges: tuple[float, ...] = ()
    policy_warmup_steps: int = 32
    total_steps: int = 260
    eps: float = 1e-6

    def __post_init__(self) -> None:
        expected = (
            int(self.num_phases),
            int(self.confidence_bins),
            int(self.spatial_bins),
        )
        if _nested_shape(self.counts) != expected or _nested_shape(
            self.exp_reward_sums
        ) != expected:
            raise ValueError(f"Omni bucket table must have shape {expected}")
        if self.confidence_bin_edges and len(self.confidence_bin_edges) != int(
            self.confidence_bins
        ) - 1:
            raise ValueError(
                "confidence_bin_edges must contain confidence_bins - 1 boundaries"
            )

    def save_artifact(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format": "omni_bucket_table_dprm_v1",
                    "config": asdict(self),
                    "metadata": dict(metadata or {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_artifact(
        cls, path: str | Path
    ) -> tuple["OmniBucketTableDPRM", dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "omni_bucket_table_dprm_v1":
            raise ValueError(f"unsupported Omni bucket-table artifact: {path}")
        config = dict(payload["config"])
        config["counts"] = _nested_tuple(config["counts"], float)
        config["exp_reward_sums"] = _nested_tuple(
            config["exp_reward_sums"], float
        )
        config["confidence_bin_edges"] = tuple(
            float(value) for value in config.get("confidence_bin_edges", ())
        )
        return cls(**config), dict(payload.get("metadata", {}))

    def score(
        self,
        confidence: torch.Tensor,
        *,
        step: int,
        visual_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adjusted = confidence.clone()
        reward = torch.zeros_like(confidence, dtype=torch.float32)
        candidates = torch.where(torch.isfinite(confidence))[0]
        if candidates.numel() == 0 or int(step) < int(self.policy_warmup_steps):
            return adjusted, reward
        confidence_probability = confidence.float().exp().clamp(
            self.eps, 1.0 - self.eps
        )
        if self.confidence_bin_edges:
            edges = torch.tensor(
                self.confidence_bin_edges,
                device=confidence.device,
                dtype=confidence_probability.dtype,
            )
            # The development table assigns values equal to a learned boundary
            # to the upper bin; keep the deployed lookup identical.
            confidence_ids = torch.bucketize(
                confidence_probability, edges, right=True
            )
        else:
            confidence_ids = torch.floor(
                confidence_probability * int(self.confidence_bins)
            ).long()
        confidence_ids = confidence_ids.clamp(0, int(self.confidence_bins) - 1)
        spatial_ids = spatial_bin_ids(
            visual_indices.to(confidence.device),
            spatial_bins=self.spatial_bins,
            image_side=16,
        )
        phase = min(
            int(step) * int(self.num_phases) // max(int(self.total_steps), 1),
            int(self.num_phases) - 1,
        )
        counts = torch.tensor(self.counts[phase], device=confidence.device)
        exp_sums = torch.tensor(self.exp_reward_sums[phase], device=confidence.device)
        candidate_confidence = confidence_ids[candidates]
        candidate_spatial = spatial_ids[candidates]
        candidate_counts = counts[candidate_confidence, candidate_spatial]
        candidate_exp_sums = exp_sums[candidate_confidence, candidate_spatial]
        safe_mean = torch.where(
            candidate_counts > 0,
            candidate_exp_sums / candidate_counts.clamp_min(1.0),
            torch.ones_like(candidate_exp_sums),
        )
        values = torch.log(safe_mean.clamp_min(self.eps)) / max(
            float(self.reward_temperature), self.eps
        )
        local_gate = (candidate_counts / max(float(self.ready_count), 1.0)).clamp(
            0.0, 1.0
        )
        if int(step) <= int(self.warmup_steps):
            global_gate = 0.0
        elif int(self.switch_steps) <= int(self.warmup_steps):
            global_gate = 1.0
        else:
            global_gate = max(
                0.0,
                min(
                    1.0,
                    (int(step) - int(self.warmup_steps))
                    / max(int(self.switch_steps) - int(self.warmup_steps), 1),
                ),
            )
        gated_values = values * local_gate * global_gate
        reward[candidates] = gated_values
        adjusted[candidates] = adjusted[candidates] + float(
            self.guidance_scale
        ) * gated_values.to(adjusted.dtype)
        return adjusted, reward


def _nested_shape(values: Any) -> tuple[int, ...]:
    shape = []
    current = values
    while isinstance(current, (tuple, list)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _nested_tuple(values: Any, cast) -> Any:
    if isinstance(values, list):
        return tuple(_nested_tuple(value, cast) for value in values)
    return cast(values)


def spatial_bin_ids(
    visual_indices: torch.Tensor,
    *,
    spatial_bins: int,
    image_side: int = 16,
) -> torch.Tensor:
    """Map zero-based visual positions to coarse row-major spatial cells."""
    visual_indices = visual_indices.long()
    if spatial_bins <= 1:
        return torch.zeros_like(visual_indices)
    side_bins = int(round(spatial_bins**0.5))
    if side_bins * side_bins == spatial_bins:
        rows = torch.div(visual_indices.clamp_min(0), image_side, rounding_mode="floor")
        columns = visual_indices.clamp_min(0) % image_side
        row_bin = torch.div(rows * side_bins, image_side, rounding_mode="floor")
        column_bin = torch.div(columns * side_bins, image_side, rounding_mode="floor")
        return (row_bin * side_bins + column_bin).clamp(0, spatial_bins - 1)
    total = image_side * image_side
    return torch.div(
        visual_indices.clamp(0, total - 1) * spatial_bins,
        total,
        rounding_mode="floor",
    ).clamp(0, spatial_bins - 1)


def load_omni_order_controller(
    path: str | Path,
) -> tuple[
    OmniRankBucketDPRM | OmniStageRankSpatialDPRM | OmniBucketTableDPRM,
    dict[str, Any],
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact_format = payload.get("format")
    if artifact_format == "omni_rank_bucket_dprm_v1":
        return OmniRankBucketDPRM.load_artifact(path)
    if artifact_format == "omni_stage_rank_spatial_dprm_v1":
        return OmniStageRankSpatialDPRM.load_artifact(path)
    if artifact_format == "omni_bucket_table_dprm_v1":
        return OmniBucketTableDPRM.load_artifact(path)
    raise ValueError(f"unsupported Omni order-controller format {artifact_format}: {path}")


def visual_candidate_mask(
    mask_index: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    image_tokens: int = 256,
) -> torch.Tensor:
    """Identify visual actions in Omni's compressed masked-position vector.

    A T2I block contains ``<begin_of_image>``, 256 visual codes, and trailing
    special tokens. Visual codes therefore occupy block-relative positions
    1 through 256, inclusive.
    """
    masked_positions = torch.where(mask_index[0])[0]
    block_positions = torch.where(block_mask)[0]
    if block_positions.numel() == 0:
        return torch.zeros_like(masked_positions, dtype=torch.bool)
    relative = masked_positions - int(block_positions[0].item())
    return (relative >= 1) & (relative <= int(image_tokens))


def candidate_visual_indices(
    mask_index: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    """Map compressed masked candidates to zero-based 16x16 canvas indices."""
    masked_positions = torch.where(mask_index[0])[0]
    block_positions = torch.where(block_mask)[0]
    if block_positions.numel() == 0:
        return torch.full_like(masked_positions, -1)
    return masked_positions - int(block_positions[0].item()) - 1


def negative_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Match Omni's deployed entropy-penalty score, including model dtype.

    The upstream sampler computes softmax, log, and reduction in the logits'
    dtype. Preserving that sequence is part of the train/test order contract:
    on bfloat16 checkpoints it also preserves the sampler's exact score ties.
    """
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-10)
    return torch.sum(probabilities * log_probabilities, dim=-1)


def entropy_penalty_order_scores(
    logits: torch.Tensor,
    *,
    top_p: float | None = 0.9,
    repeat_penalty: float = 1.2,
    max_position_penalty: float = 2.0,
    past_tokens: torch.Tensor | None = None,
    mask_id: int = 151666,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce Omni's deterministic ``entropy-penalty`` token proposal.

    The returned pair is the negative-entropy position score and provisional
    argmax token.  Keeping this transformation in the ordering module lets the
    current-model training roll-in use the same score as deployed decoding.
    """
    adjusted = logits.clone()
    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(adjusted, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        remove = cumulative_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(adjusted, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, remove)
        adjusted = adjusted.masked_fill(mask, torch.finfo(adjusted.dtype).min)

    if float(repeat_penalty) != 1.0 and past_tokens is not None:
        visible = past_tokens[(past_tokens != 0) & (past_tokens != int(mask_id))]
        for token in torch.unique(visible).tolist():
            token = int(token)
            negative = adjusted[:, token] < 0
            adjusted[:, token][negative] *= float(repeat_penalty)
            adjusted[:, token][~negative] /= float(repeat_penalty)

    if float(max_position_penalty) != 1.0 and adjusted.shape[-2] > 100:
        token_length = int(adjusted.shape[-2])
        # Match the upstream Python-float construction exactly. Replacing this
        # with vectorized torch arithmetic changes a few float32 values by one
        # ULP and can alter a tied position order.
        tail = [
            i / (token_length - 100) * (float(max_position_penalty) - 1.0) + 1.0
            for i in range(token_length - 100)
        ]
        position_penalty = torch.tensor(tail).unsqueeze(-1).to(
            adjusted.device, adjusted.dtype
        )
        position_penalty = torch.cat(
            [torch.ones_like(adjusted[:100, :1]), position_penalty], dim=0
        )
        position_penalty = position_penalty.repeat(1, adjusted.shape[-1])
        negative = adjusted < 0
        adjusted[negative] *= position_penalty[negative]
        adjusted[~negative] /= position_penalty[~negative]

    provisional = torch.argmax(torch.softmax(adjusted, dim=-1), dim=-1)
    return negative_entropy(adjusted), provisional


def confidence_probability(confidence: torch.Tensor) -> torch.Tensor:
    """Monotone bounded coordinate used by the learned feature map."""
    return confidence.detach().float().exp().clamp(1e-6, 1.0 - 1e-6)


def relative_rank(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantiles = torch.zeros_like(scores, dtype=torch.float32)
    rank_bins = torch.zeros_like(scores, dtype=torch.long)
    if candidates.numel() == 0:
        return quantiles, rank_bins
    order = candidates[torch.argsort(scores[candidates])]
    denominator = max(int(order.numel()) - 1, 1)
    ordered_quantiles = torch.arange(
        order.numel(), device=scores.device, dtype=torch.float32
    ) / float(denominator)
    quantiles[order] = ordered_quantiles
    rank_bins[order] = torch.floor(ordered_quantiles * max(int(bins), 1)).long().clamp(
        0, max(int(bins), 1) - 1
    )
    return quantiles, rank_bins


def _one_hot(index: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(index.long().clamp(0, classes - 1), classes).float()


def encode_action_features(
    *,
    step_fraction: torch.Tensor,
    rank_quantile: torch.Tensor,
    row_normalized: torch.Tensor,
    column_normalized: torch.Tensor,
    center_distance: torch.Tensor,
    local_revealed_fraction: torch.Tensor,
    nearest_revealed: torch.Tensor,
    candidate_fraction: torch.Tensor,
    confidence: torch.Tensor,
    confidence_gap: torch.Tensor,
    phase_ids: torch.Tensor,
    rank_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    local_ids: torch.Tensor,
    code_ids: torch.Tensor,
    config: OmniOrderConfig,
) -> torch.Tensor:
    continuous = torch.stack(
        [
            step_fraction,
            rank_quantile,
            row_normalized,
            column_normalized,
            center_distance,
            local_revealed_fraction,
            nearest_revealed,
            candidate_fraction,
            confidence,
            confidence_gap,
        ],
        dim=1,
    ).float()
    features = torch.cat(
        [
            continuous,
            _one_hot(phase_ids, config.num_phases),
            _one_hot(rank_ids, config.rank_bins),
            _one_hot(spatial_ids, config.spatial_bins),
            _one_hot(local_ids, 3),
            _one_hot(code_ids, 16),
        ],
        dim=1,
    )
    if features.shape[1] != config.feature_dim:
        raise RuntimeError(f"Omni action feature width {features.shape[1]} != {config.feature_dim}")
    return features


def build_action_features(
    *,
    confidence: torch.Tensor,
    candidate_indices: torch.Tensor,
    visual_indices: torch.Tensor,
    masked_visual: torch.Tensor,
    provisional_token_ids: torch.Tensor,
    step: int,
    config: OmniOrderConfig,
) -> torch.Tensor:
    """Build the exact feature tensor consumed in both training and inference.

    ``candidate_indices`` index ``confidence`` and ``provisional_token_ids``.
    ``visual_indices`` are the corresponding zero-based positions on the 16x16 canvas.
    ``masked_visual`` is a length-256 Boolean mask for the current partial canvas.
    """
    device = confidence.device
    candidate_indices = candidate_indices.long().to(device)
    visual_indices = visual_indices.long().to(device).clamp(0, config.image_side**2 - 1)
    masked_visual = masked_visual.bool().to(device)
    if candidate_indices.numel() != visual_indices.numel():
        raise ValueError("candidate_indices and visual_indices must have equal length")
    if candidate_indices.numel() == 0:
        return torch.empty((0, config.feature_dim), device=device, dtype=torch.float32)

    conf_prob = confidence_probability(confidence)
    quantiles, rank_bins = relative_rank(conf_prob, candidate_indices, config.rank_bins)
    selected_conf = conf_prob[candidate_indices]
    default_conf = torch.max(selected_conf)

    side = config.image_side
    rows = torch.div(visual_indices, side, rounding_mode="floor")
    columns = visual_indices % side
    row_norm = rows.float() / float(max(side - 1, 1))
    column_norm = columns.float() / float(max(side - 1, 1))
    center = (side - 1) / 2.0
    center_distance = torch.sqrt((rows.float() - center) ** 2 + (columns.float() - center) ** 2)
    center_distance = center_distance / max(center * 2**0.5, 1.0)

    revealed_visual = torch.where(~masked_visual)[0]
    local_fractions = []
    nearest_distances = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        neighbor_count = 0
        revealed_count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, column + dc
                if 0 <= nr < side and 0 <= nc < side:
                    neighbor_count += 1
                    revealed_count += int(not masked_visual[nr * side + nc].item())
        local_fractions.append(revealed_count / max(neighbor_count, 1))
        if revealed_visual.numel():
            rr = torch.div(revealed_visual, side, rounding_mode="floor")
            cc = revealed_visual % side
            nearest = torch.min(torch.abs(rr - row) + torch.abs(cc - column)).item()
        else:
            nearest = side * 2
        nearest_distances.append(float(nearest) / float(side * 2))

    local = torch.tensor(local_fractions, device=device, dtype=torch.float32)
    nearest = torch.tensor(nearest_distances, device=device, dtype=torch.float32)
    candidate_fraction = torch.full_like(selected_conf, candidate_indices.numel() / 256.0)
    step_fraction = torch.full_like(selected_conf, int(step) / max(float(config.total_steps), 1.0))

    phase = min(int(step) * config.num_phases // max(config.total_steps, 1), config.num_phases - 1)
    phase_ids = torch.full_like(candidate_indices, phase)
    if config.spatial_bins == 16:
        spatial_ids = torch.div(rows, 4, rounding_mode="floor") * 4 + torch.div(
            columns, 4, rounding_mode="floor"
        )
    else:
        spatial_ids = torch.div(
            visual_indices * config.spatial_bins, side * side, rounding_mode="floor"
        ).clamp(0, config.spatial_bins - 1)
    local_ids = torch.floor(local * 3).long().clamp(0, 2)
    provisional = provisional_token_ids[candidate_indices].long()
    visual_codes = (provisional - config.image_token_offset).clamp(0, config.image_vocab_size - 1)
    code_ids = torch.div(visual_codes * 16, config.image_vocab_size, rounding_mode="floor").clamp(0, 15)

    return encode_action_features(
        step_fraction=step_fraction,
        rank_quantile=quantiles[candidate_indices],
        row_normalized=row_norm,
        column_normalized=column_norm,
        center_distance=center_distance,
        local_revealed_fraction=local,
        nearest_revealed=nearest,
        candidate_fraction=candidate_fraction,
        confidence=selected_conf,
        confidence_gap=selected_conf - default_conf,
        phase_ids=phase_ids,
        rank_ids=rank_bins[candidate_indices],
        spatial_ids=spatial_ids,
        local_ids=local_ids,
        code_ids=code_ids,
        config=config,
    )


def features_from_counterfactual_records(
    records: list[dict[str, Any]],
    config: OmniOrderConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode saved shared-state actions using the live policy feature contract."""
    if not records:
        return torch.empty((0, config.feature_dim), device=device)

    def tensor(key: str, scale: float = 1.0) -> torch.Tensor:
        return torch.tensor(
            [float(row[key]) / scale for row in records], device=device, dtype=torch.float32
        )

    visual_codes = torch.tensor(
        [
            max(
                0,
                min(
                    int(row["provisional_token_id"]) - config.image_token_offset,
                    config.image_vocab_size - 1,
                ),
            )
            for row in records
        ],
        device=device,
    )
    local = tensor("local_revealed_fraction")
    return encode_action_features(
        step_fraction=tensor("step", max(config.total_steps, 1)),
        rank_quantile=tensor("rank_quantile"),
        row_normalized=tensor("row_normalized"),
        column_normalized=tensor("column_normalized"),
        center_distance=tensor("center_distance", max((config.image_side - 1) / 2 * 2**0.5, 1.0)),
        local_revealed_fraction=local,
        nearest_revealed=tensor("nearest_revealed_manhattan", config.image_side * 2),
        candidate_fraction=tensor("candidate_count", 256),
        confidence=tensor("confidence"),
        confidence_gap=tensor("confidence_gap_from_default"),
        phase_ids=torch.tensor([int(row["phase"]) for row in records], device=device),
        rank_ids=torch.tensor([int(row["rank_bin"]) for row in records], device=device),
        spatial_ids=torch.tensor([int(row["aux_bin"]) for row in records], device=device),
        local_ids=torch.floor(local * 3).long().clamp(0, 2),
        code_ids=torch.div(visual_codes * 16, config.image_vocab_size, rounding_mode="floor").clamp(0, 15),
        config=config,
    )


def adjusted_order_scores(
    *,
    confidence: torch.Tensor,
    candidate_indices: torch.Tensor,
    features: torch.Tensor,
    scorer: OmniOrderScorer,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = scorer(features.to(next(scorer.parameters()).device)).to(confidence.device)
    adjusted = confidence.clone()
    adjusted[candidate_indices] = adjusted[candidate_indices] + float(guidance_scale) * advantages
    return adjusted, advantages


def visual_token_runs(
    clean_ids: torch.Tensor,
    *,
    image_token_offset: int = 168072,
    image_vocab_size: int = 8192,
) -> list[torch.Tensor]:
    """Return contiguous visual-code runs from one packed training sequence."""
    visual = (clean_ids >= image_token_offset) & (
        clean_ids < image_token_offset + image_vocab_size
    )
    positions = torch.where(visual)[0]
    if positions.numel() == 0:
        return []
    breaks = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    return [chunk for chunk in torch.tensor_split(positions, breaks.tolist()) if chunk.numel()]
