from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def cuda_bf16_supported() -> bool:
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(torch.cuda.is_available() and is_supported is not None and is_supported())


def cuda_autocast_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name in (None, "fp32", "float32"):
        return None
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16 if cuda_bf16_supported() else torch.float16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def autocast_context(device: torch.device, dtype_name: str | None) -> ContextManager[None]:
    dtype = cuda_autocast_dtype(dtype_name)
    if dtype is None:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)
