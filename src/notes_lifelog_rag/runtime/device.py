from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from notes_lifelog_rag.runtime.cuda import CudaStatus, collect_cuda_status


@dataclass(frozen=True)
class DeviceInfo:
    requested_device: str
    resolved_device: str
    torch_device: str
    is_cuda: bool
    cuda_available: bool
    reason: str
    warning: str | None
    dtype_recommendation: str
    device_count: int
    selected_device_name: str | None = None
    selected_device_memory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeviceResolutionError(RuntimeError):
    pass


def resolve_device(
    requested_device: str = "auto",
    *,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
    dtype: str = "auto",
    cuda_status: CudaStatus | None = None,
) -> DeviceInfo:
    requested = (requested_device or "auto").strip().lower()
    _validate_dtype(dtype)
    if requested == "gpu":
        requested = "cuda"
    if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
        raise DeviceResolutionError(f"unsupported device '{requested_device}'. Use auto, cpu, cuda, or cuda:N.")

    status = cuda_status or collect_cuda_status()
    hard_cuda = require_cuda or not allow_cpu_fallback

    if requested == "cpu":
        if require_cuda:
            raise DeviceResolutionError("--device cpu cannot be combined with --require-cuda")
        return _device_info(
            requested,
            "cpu",
            status,
            reason="CPU explicitly requested",
            warning=None,
            dtype=dtype,
        )

    if requested == "auto":
        if status.cuda_available and status.device_count > 0:
            return _device_info(
                requested,
                "cuda:0",
                status,
                reason="CUDA is available; selected cuda:0",
                warning=None,
                dtype=dtype,
            )
        if hard_cuda:
            raise DeviceResolutionError(f"CUDA required but unavailable: {status.likely_reason}")
        return _device_info(
            requested,
            "cpu",
            status,
            reason=f"CUDA unavailable; using CPU fallback ({status.likely_reason})",
            warning=f"CUDA unavailable; using CPU fallback ({status.likely_reason})",
            dtype=dtype,
        )

    index = 0
    if requested.startswith("cuda:"):
        index_text = requested.split(":", 1)[1]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise DeviceResolutionError(f"invalid CUDA device index in '{requested_device}'") from exc

    if status.cuda_available and 0 <= index < status.device_count:
        return _device_info(
            requested,
            f"cuda:{index}",
            status,
            reason=f"CUDA requested and available; selected cuda:{index}",
            warning=None,
            dtype=dtype,
        )

    reason = (
        f"CUDA device cuda:{index} is unavailable; device_count={status.device_count}, "
        f"reason={status.likely_reason}"
    )
    if hard_cuda:
        raise DeviceResolutionError(reason)
    return _device_info(
        requested,
        "cpu",
        status,
        reason=f"{reason}; using CPU fallback",
        warning=f"{reason}; using CPU fallback",
        dtype=dtype,
    )


def effective_dtype(dtype: str, device_info: DeviceInfo) -> str:
    requested = (dtype or "auto").lower()
    if requested != "auto":
        return requested
    return device_info.dtype_recommendation


def torch_dtype(torch_obj: Any, dtype: str, device_info: DeviceInfo) -> Any | None:
    selected = effective_dtype(dtype, device_info)
    if selected == "float16":
        return getattr(torch_obj, "float16", None)
    if selected == "bfloat16":
        return getattr(torch_obj, "bfloat16", None)
    if selected == "float32":
        return getattr(torch_obj, "float32", None)
    return None


def autocast_context(torch_obj: Any, device_info: DeviceInfo, dtype_value: Any | None):
    if not device_info.is_cuda or dtype_value is None:
        return nullcontext()
    float32 = getattr(torch_obj, "float32", None)
    if dtype_value == float32 or not hasattr(torch_obj, "autocast"):
        return nullcontext()
    return torch_obj.autocast(device_type="cuda", dtype=dtype_value)


def _device_info(
    requested: str,
    resolved: str,
    status: CudaStatus,
    *,
    reason: str,
    warning: str | None,
    dtype: str,
) -> DeviceInfo:
    is_cuda = resolved.startswith("cuda")
    index = int(resolved.split(":", 1)[1]) if ":" in resolved and is_cuda else None
    selected = status.devices[index] if index is not None and index < len(status.devices) else None
    memory = None
    if selected and selected.memory_total_mb is not None:
        free = selected.memory_free_mb if selected.memory_free_mb is not None else 0
        memory = f"{free}/{selected.memory_total_mb} MB free"
    return DeviceInfo(
        requested_device=requested,
        resolved_device=resolved,
        torch_device=resolved,
        is_cuda=is_cuda,
        cuda_available=status.cuda_available,
        reason=reason,
        warning=warning,
        dtype_recommendation=_dtype_recommendation(dtype, is_cuda),
        device_count=status.device_count,
        selected_device_name=selected.name if selected else None,
        selected_device_memory=memory,
    )


def _dtype_recommendation(dtype: str, is_cuda: bool) -> str:
    requested = (dtype or "auto").lower()
    if requested in {"float32", "float16", "bfloat16"}:
        return requested
    if not is_cuda:
        return "float32"
    try:
        import torch

        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float16"


def _validate_dtype(dtype: str) -> None:
    requested = (dtype or "auto").lower()
    if requested not in {"auto", "float32", "float16", "bfloat16"}:
        raise DeviceResolutionError(f"unsupported dtype '{dtype}'. Use auto, float32, float16, or bfloat16.")
