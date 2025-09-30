from __future__ import annotations

import importlib
from typing import Any, Optional


def try_import_torch() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        optim = importlib.import_module("torch.optim")
        return torch, nn, optim
    except Exception:
        return None, None, None
