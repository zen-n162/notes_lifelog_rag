from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any

from notes_lifelog_rag.config import load_model_config, load_model_defaults
from notes_lifelog_rag.runtime.device import DeviceInfo, resolve_device


@dataclass(frozen=True)
class ModelStatus:
    purpose: str
    name: str
    path: str
    path_exists: bool
    runtime_available: bool
    enabled: bool
    reason: str
    runtime_status: str = "ready"
    cuda_status: str = "not_requested"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_statuses(
    *,
    requested_device: str = "auto",
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
    device_info: DeviceInfo | None = None,
) -> list[ModelStatus]:
    resolved = device_info or resolve_device(
        requested_device,
        require_cuda=require_cuda,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    statuses: list[ModelStatus] = []
    for purpose, entries in load_model_config().items():
        for entry in entries:
            statuses.append(status_for_entry(purpose, entry, device_info=resolved))
    return statuses


def status_for_entry(purpose: str, entry: dict[str, str], *, device_info: DeviceInfo | None = None) -> ModelStatus:
    name = str(entry.get("name") or "")
    path = str(entry.get("path") or "")
    path_exists = Path(path).exists()
    runtime_available, reason, runtime_status = _runtime_status(purpose, device_info)
    enabled = path_exists and runtime_available and purpose in {"text_generation", "embedding", "reranker"}
    if not path_exists:
        reason = "model path is missing; feature disabled"
        runtime_status = "missing_model"
    elif purpose not in {"text_generation", "embedding", "reranker"}:
        reason = "path exists; not used by Phase 3-4 text MVP"
        runtime_status = "not_used"
    return ModelStatus(
        purpose=purpose,
        name=name,
        path=path,
        path_exists=path_exists,
        runtime_available=runtime_available,
        enabled=enabled,
        reason=reason,
        runtime_status=runtime_status,
        cuda_status=_cuda_status(purpose, device_info),
    )


def resolve_model_entry(purpose: str, model_name: str | None = None) -> dict[str, str] | None:
    config = load_model_config()
    entries = config.get(purpose, [])
    selected = model_name or load_model_defaults().get(purpose)
    if selected:
        for entry in entries:
            if entry.get("name") == selected:
                return entry
    return entries[0] if entries else None


def status_for_model(
    purpose: str,
    model_name: str | None = None,
    *,
    device_info: DeviceInfo | None = None,
) -> ModelStatus | None:
    entry = resolve_model_entry(purpose, model_name)
    return status_for_entry(purpose, entry, device_info=device_info) if entry else None


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _runtime_status(purpose: str, device_info: DeviceInfo | None = None) -> tuple[bool, str, str]:
    if purpose == "embedding":
        if has_module("sentence_transformers") or (has_module("transformers") and has_module("torch")):
            return True, "local embedding runtime is available", "ready"
        return False, "missing runtime dependency: sentence_transformers or transformers+torch", "missing_dependency"
    if purpose == "reranker":
        if has_module("sentence_transformers") or (has_module("transformers") and has_module("torch")):
            return True, "local reranker runtime is available", "ready"
        return False, "missing runtime dependency: sentence_transformers or transformers+torch", "missing_dependency"
    if purpose == "text_generation":
        if has_module("transformers") and has_module("torch"):
            return True, "local text-generation runtime is available", "ready"
        return False, "missing runtime dependency: transformers+torch", "missing_dependency"
    if purpose == "asr":
        ok = has_module("faster_whisper")
        return ok, "requires faster_whisper runtime", "ready" if ok else "missing_dependency"
    if purpose == "vision_ocr":
        return False, "vision/OCR models are outside Phase 3-4 text MVP", "not_used"
    return False, "unknown model purpose", "unknown"


def _cuda_status(purpose: str, device_info: DeviceInfo | None) -> str:
    if purpose not in {"text_generation", "embedding", "reranker"}:
        return "not_requested"
    if device_info is None or device_info.requested_device == "cpu":
        return "not_requested"
    if device_info.is_cuda:
        return "available"
    if device_info.warning:
        return "fallback_cpu"
    return "unavailable"
