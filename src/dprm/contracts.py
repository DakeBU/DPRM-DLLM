from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HostDPRMBatch:
    """Host-to-controller contract for a single DPRM decision.

    Shapes are expected to be batch-major:
      * confidence: [B, L]
      * candidate_mask: [B, L]
      * phase_ids: [B]
      * aux_bin_ids: [B, L] or None
    """

    confidence: torch.Tensor
    candidate_mask: torch.Tensor
    phase_ids: torch.Tensor
    aux_bin_ids: Optional[torch.Tensor] = None
    global_step: int = 0
    force_full_dprm: bool = False


@dataclass
class DPRMSelection:
    selected_mask: torch.Tensor
    score: torch.Tensor
    base_score: torch.Tensor
    dprm_value: torch.Tensor
    gate: torch.Tensor
    confidence_bins: torch.Tensor
