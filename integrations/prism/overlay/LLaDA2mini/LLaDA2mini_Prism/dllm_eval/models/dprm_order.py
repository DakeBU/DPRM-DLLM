from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve()
for parent in _ROOT.parents:
    src = parent / "src"
    if src.exists():
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        break

from dprm.adapters.prism import OnlineDPRMSoftBON

__all__ = ["OnlineDPRMSoftBON"]
