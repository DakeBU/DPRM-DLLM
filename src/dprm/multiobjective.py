from __future__ import annotations

from typing import Literal, Optional

import torch


Scalarization = Literal["weighted_sum", "smooth_tchebycheff"]


def _normalized_weights(
    weights: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    if weights.ndim != 1 or weights.numel() != reference.shape[-1]:
        raise ValueError(
            "weights must be one-dimensional and match the objective dimension"
        )
    if torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be nonnegative and have positive sum")
    return weights / weights.sum()


def scalarize_benefits(
    benefits: torch.Tensor,
    weights: torch.Tensor,
    *,
    method: Scalarization = "weighted_sum",
    ideal: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    augmentation: float = 0.05,
) -> torch.Tensor:
    """Convert normalized maximization benefits into one terminal utility.

    Smooth Tchebycheff emphasizes the largest weighted shortfall from the ideal
    point. The augmentation term breaks near-ties using the weighted mean.
    """

    if benefits.ndim < 1:
        raise ValueError("benefits must have at least one dimension")
    weights = _normalized_weights(weights, benefits)
    weighted_mean = (benefits * weights).sum(dim=-1)

    if method == "weighted_sum":
        return weighted_mean
    if method != "smooth_tchebycheff":
        raise ValueError(f"unknown scalarization method: {method}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if augmentation < 0:
        raise ValueError("augmentation must be nonnegative")

    if ideal is None:
        ideal = torch.ones(
            benefits.shape[-1], dtype=benefits.dtype, device=benefits.device
        )
    else:
        ideal = torch.as_tensor(ideal, dtype=benefits.dtype, device=benefits.device)
    if ideal.ndim != 1 or ideal.numel() != benefits.shape[-1]:
        raise ValueError("ideal must match the objective dimension")

    weighted_shortfall = weights * (ideal - benefits)
    smooth_worst = temperature * torch.logsumexp(
        weighted_shortfall / temperature, dim=-1
    )
    return 1.0 - smooth_worst + augmentation * weighted_mean
