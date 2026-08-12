from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch


Scalarization = Literal["weighted_sum", "smooth_tchebycheff"]


def molecular_token_class_ids(tokens: Sequence[str]) -> torch.Tensor:
    """Map SAFE tokenizer entries to eight coarse chemistry/syntax classes."""

    classes = []
    special = {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}
    syntax = {"#", "%", "(", ")", "+", "-", "/", "=", "@", "[", "\\", "]"}
    for raw_token in tokens:
        token = str(raw_token).replace("##", "")
        if token in special or token == ".":
            token_class = 0
        elif token in syntax or any(character.isdigit() for character in token):
            token_class = 1
        elif "Cl" in token or "Br" in token or token in {"F", "I"}:
            token_class = 6
        elif "N" in token or "n" in token:
            token_class = 3
        elif "O" in token or "o" in token:
            token_class = 4
        elif any(element in token for element in ("S", "s", "P", "p")):
            token_class = 5
        elif "C" in token or "c" in token:
            token_class = 2
        else:
            token_class = 7
        classes.append(token_class)
    return torch.tensor(classes, dtype=torch.long)


def sparse_reconstruction_benefits(
    predicted: torch.Tensor,
    target: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    max_bin_distance: float,
) -> torch.Tensor:
    """Return nonzero recovery, nonzero MAE benefit, and zero accuracy."""

    if predicted.shape != target.shape or predicted.shape != selected_mask.shape:
        raise ValueError("predicted, target, and selected_mask must have matching shapes")
    if max_bin_distance <= 0:
        raise ValueError("max_bin_distance must be positive")
    selected_mask = selected_mask.bool()
    selected_nonzero = selected_mask & (target != 0)
    nonzero_count = selected_nonzero.sum(dim=-1)
    nonzero_recovery = ((predicted == target) & selected_nonzero).sum(dim=-1).float()
    nonzero_recovery = torch.where(
        nonzero_count > 0,
        nonzero_recovery / nonzero_count.clamp_min(1).float(),
        torch.zeros_like(nonzero_recovery),
    )
    nonzero_mae = (
        ((predicted.float() - target.float()).abs() * selected_nonzero).sum(dim=-1)
        / (nonzero_count.clamp_min(1).float() * float(max_bin_distance))
    ).clamp(0.0, 1.0)
    nonzero_mae_benefit = torch.where(
        nonzero_count > 0,
        1.0 - nonzero_mae,
        torch.zeros_like(nonzero_mae),
    )
    selected_zero = selected_mask & (target == 0)
    zero_count = selected_zero.sum(dim=-1)
    zero_accuracy = ((predicted == 0) & selected_zero).sum(dim=-1).float()
    zero_accuracy = torch.where(
        zero_count > 0,
        zero_accuracy / zero_count.clamp_min(1).float(),
        torch.zeros_like(zero_accuracy),
    )
    return torch.stack((nonzero_recovery, nonzero_mae_benefit, zero_accuracy), dim=-1)


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

    ``weighted_sum`` returns ``sum_j lambda_j * r_j``. For
    ``smooth_tchebycheff`` we apply STCH to the minimization losses
    ``f_j = z_j - r_j`` and negate the result so that larger remains better::

        1 - mu * logsumexp(lambda_j * (z_j - r_j) / mu)
            + augmentation * sum_j lambda_j * r_j

    The additive one does not affect ordering when the ideal is the all-ones
    vector. ``augmentation`` is optional and only breaks near-ties; setting it
    to zero recovers the direct maximization counterpart of STCH.
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
    if not torch.isfinite(benefits).all() or not torch.isfinite(ideal).all():
        raise ValueError("benefits and ideal must be finite")

    weighted_shortfall = weights * (ideal - benefits)
    smooth_worst = temperature * torch.logsumexp(
        weighted_shortfall / temperature, dim=-1
    )
    return 1.0 - smooth_worst + augmentation * weighted_mean
