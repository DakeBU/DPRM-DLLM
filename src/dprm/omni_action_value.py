"""Feature contract for Omni shared-state action-value estimation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


PROMPT_FAMILIES = {
    "person": ("person", "portrait", "woman", "man", "girl", "boy", "human", "face"),
    "animal": ("animal", "cat", "dog", "bird", "horse", "fish", "bear", "dragon"),
    "architecture": ("building", "house", "church", "mosque", "temple", "city", "room", "interior"),
    "landscape": ("landscape", "mountain", "forest", "beach", "ocean", "river", "sky", "garden"),
    "still_life": ("still life", "bottle", "cup", "mug", "food", "fruit", "product", "object"),
    "text_graphic": ("text", "logo", "poster", "sign", "typography", "diagram"),
    "photo": ("photo", "photograph", "realistic", "cinematic", "camera", "lens"),
    "painting": ("painting", "watercolor", "oil paint", "illustration", "sketch", "drawing"),
    "render_3d": ("3d", "render", "octane", "unreal engine", "ray tracing"),
}


class PhaseRankActionValueRegressor(RegressorMixin, BaseEstimator):
    """Zero-shrunk table over decode phase and within-state confidence rank."""

    def __init__(self, shrinkage: float = 4.0) -> None:
        self.shrinkage = shrinkage

    @staticmethod
    def _bucket(features: dict[str, Any]) -> tuple[str, str]:
        phase = next((key for key in features if key.startswith("phase=")), "phase=unknown")
        rank = next((key for key in features if key.startswith("rank=")), "rank=unknown")
        return phase, rank

    def fit(
        self,
        features: list[dict[str, Any]],
        targets: np.ndarray,
    ) -> "PhaseRankActionValueRegressor":
        sums: dict[tuple[str, str], float] = defaultdict(float)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for row, target in zip(features, targets):
            bucket = self._bucket(row)
            sums[bucket] += float(target)
            counts[bucket] += 1
        shrinkage = max(float(self.shrinkage), 0.0)
        self.values_ = {
            bucket: sums[bucket] / (counts[bucket] + shrinkage)
            for bucket in counts
        }
        self.counts_ = dict(counts)
        return self

    def predict(self, features: list[dict[str, Any]]) -> np.ndarray:
        if not hasattr(self, "values_"):
            raise RuntimeError("PhaseRankActionValueRegressor is not fitted")
        return np.asarray(
            [self.values_.get(self._bucket(row), 0.0) for row in features],
            dtype=np.float64,
        )


def prompt_feature_dict(prompt: str) -> dict[str, float]:
    normalized = " ".join(str(prompt).lower().split())
    words = normalized.split()
    features = {
        "prompt_word_count": min(len(words), 100) / 100.0,
        "prompt_digit_fraction": sum(char.isdigit() for char in normalized)
        / max(len(normalized), 1),
    }
    matched = False
    for family, terms in PROMPT_FAMILIES.items():
        active = any(term in normalized for term in terms)
        if active:
            features[f"prompt={family}"] = 1.0
            matched = True
    if not matched:
        features["prompt=other"] = 1.0
    return features


def action_feature_dict(
    *,
    step: int,
    rank_quantile: float,
    rank_bin: int,
    aux_bin: int,
    row_normalized: float,
    column_normalized: float,
    center_distance: float,
    local_revealed_fraction: float,
    nearest_revealed_manhattan: float,
    candidate_count: int,
    confidence: float,
    confidence_gap_from_default: float,
    provisional_token_id: int,
    prompt: str = "",
    image_token_offset: int = 168072,
    total_steps: int = 260,
    num_phases: int = 8,
) -> dict[str, float]:
    phase = min(int(step) * int(num_phases) // max(int(total_steps), 1), int(num_phases) - 1)
    local_bin = min(int(float(local_revealed_fraction) * 3.0), 2)
    visual_code = max(0, min(int(provisional_token_id) - int(image_token_offset), 8191))
    code_bin_16 = visual_code // 512
    code_bin_64 = visual_code // 128
    features = {
        "step_fraction": int(step) / max(float(total_steps), 1.0),
        "rank_quantile": float(rank_quantile),
        "row_normalized": float(row_normalized),
        "column_normalized": float(column_normalized),
        "center_distance": float(center_distance) / 11.0,
        "local_revealed_fraction": float(local_revealed_fraction),
        "nearest_revealed_manhattan": float(nearest_revealed_manhattan) / 32.0,
        "candidate_fraction": int(candidate_count) / 255.0,
        "confidence": float(confidence),
        "confidence_gap_from_default": float(confidence_gap_from_default),
        f"phase={phase}": 1.0,
        f"rank={int(rank_bin)}": 1.0,
        f"spatial={int(aux_bin)}": 1.0,
        f"local={local_bin}": 1.0,
        f"phase_rank={phase}:{int(rank_bin)}": 1.0,
        f"rank_spatial={int(rank_bin)}:{int(aux_bin)}": 1.0,
        f"rank_local={int(rank_bin)}:{local_bin}": 1.0,
        f"code16={code_bin_16}": 1.0,
        f"code64={code_bin_64}": 1.0,
        f"rank_code16={int(rank_bin)}:{code_bin_16}": 1.0,
    }
    prompt_features = prompt_feature_dict(prompt)
    features.update(prompt_features)
    for key in prompt_features:
        if not key.startswith("prompt="):
            continue
        family = key.removeprefix("prompt=")
        features[f"prompt_rank={family}:{int(rank_bin)}"] = 1.0
        features[f"prompt_phase={family}:{phase}"] = 1.0
    return features
