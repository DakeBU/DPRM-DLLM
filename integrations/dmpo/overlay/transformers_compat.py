"""Compatibility guard for loading LLaDA with recent Transformers releases."""

from __future__ import annotations

from transformers import modeling_utils


def install_llada_tp_plan_guard() -> None:
    """Treat an absent tensor-parallel plan as the empty plan it represents."""

    current = modeling_utils.caching_allocator_warmup
    if getattr(current, "_dprm_llada_tp_plan_guard", False):
        return

    def warmup_without_tensor_parallel_plan(model, expanded_device_map, factor=2):
        if getattr(model, "_tp_plan", None) is None:
            model._tp_plan = {}
        return current(model, expanded_device_map, factor=factor)

    warmup_without_tensor_parallel_plan._dprm_llada_tp_plan_guard = True
    modeling_utils.caching_allocator_warmup = warmup_without_tensor_parallel_plan

