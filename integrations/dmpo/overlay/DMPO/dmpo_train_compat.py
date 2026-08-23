"""Run upstream DMPO with a Transformers compatibility guard.

Some remote model classes leave ``_tp_plan`` as ``None`` while recent
Transformers releases iterate over it during distributed allocator warmup.
An absent tensor-parallel plan is equivalent to an empty plan here.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))

from transformers_compat import install_llada_tp_plan_guard

install_llada_tp_plan_guard()
runpy.run_path(str(Path(__file__).with_name("dmpo_train.py")), run_name="__main__")
