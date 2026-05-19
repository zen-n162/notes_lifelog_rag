from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import re
import shutil
import subprocess
import sys
import warnings
from typing import Any, Callable


@dataclass(frozen=True)
class CudaDevice:
    index: int
    name: str
    capability: str
    memory_free_mb: int | None = None
    memory_total_mb: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NvidiaSmiGpu:
    index: int
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CudaStatus:
    python_version: str
    torch_installed: bool
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_build: bool = False
    cuda_available: bool = False
    device_count: int = 0
    devices: list[CudaDevice] = field(default_factory=list)
    cuda_visible_devices: str | None = None
    nvidia_smi_path: str | None = None
    nvidia_smi_driver_version: str | None = None
    nvidia_smi_cuda_version: str | None = None
    nvidia_smi_gpus: list[NvidiaSmiGpu] = field(default_factory=list)
    init_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    likely_reason: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["devices"] = [device.to_dict() for device in self.devices]
        value["nvidia_smi_gpus"] = [gpu.to_dict() for gpu in self.nvidia_smi_gpus]
        return value


SmiRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def collect_cuda_status(
    *,
    torch_module: Any | None = None,
    nvidia_smi_runner: SmiRunner | None = None,
) -> CudaStatus:
    torch_obj = torch_module
    torch_installed = True
    torch_version = None
    torch_cuda_version = None
    torch_cuda_build = False
    cuda_available = False
    device_count = 0
    devices: list[CudaDevice] = []
    init_error = None
    captured_warnings: list[str] = []

    if torch_obj is None:
        try:
            import torch as torch_obj  # type: ignore[no-redef]
        except Exception as exc:
            torch_installed = False
            torch_obj = None
            init_error = f"torch import failed: {exc}"

    if torch_obj is not None:
        torch_version = str(getattr(torch_obj, "__version__", "unknown"))
        torch_cuda_version = str(getattr(getattr(torch_obj, "version", None), "cuda", "") or "")
        torch_cuda_version = torch_cuda_version or None
        torch_cuda_build = torch_cuda_version is not None
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                cuda_available = bool(torch_obj.cuda.is_available())
            captured_warnings.extend(str(record.message) for record in records)
        except Exception as exc:
            cuda_available = False
            init_error = f"torch.cuda.is_available failed: {exc}"

        try:
            if cuda_available:
                device_count = int(torch_obj.cuda.device_count())
        except Exception as exc:
            captured_warnings.append(f"torch.cuda.device_count failed: {exc}")
            device_count = 0

        if cuda_available and device_count > 0:
            for index in range(device_count):
                devices.append(_torch_device_info(torch_obj, index))

    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_driver_version = None
    nvidia_smi_cuda_version = None
    nvidia_smi_gpus: list[NvidiaSmiGpu] = []
    if nvidia_smi_path:
        nvidia_smi_driver_version, nvidia_smi_cuda_version, nvidia_smi_gpus = _nvidia_smi_summary(
            nvidia_smi_path,
            nvidia_smi_runner,
        )

    likely_reason = _infer_reason(
        torch_installed=torch_installed,
        torch_cuda_build=torch_cuda_build,
        cuda_available=cuda_available,
        nvidia_smi_path=nvidia_smi_path,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        warnings_list=captured_warnings,
        init_error=init_error,
    )
    recommendations = _recommendations(likely_reason, cuda_available)
    return CudaStatus(
        python_version=sys.version.split()[0],
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_build=torch_cuda_build,
        cuda_available=cuda_available,
        device_count=device_count,
        devices=devices,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nvidia_smi_path=nvidia_smi_path,
        nvidia_smi_driver_version=nvidia_smi_driver_version,
        nvidia_smi_cuda_version=nvidia_smi_cuda_version,
        nvidia_smi_gpus=nvidia_smi_gpus,
        init_error=init_error,
        warnings=captured_warnings,
        likely_reason=likely_reason,
        recommendations=recommendations,
    )


def _torch_device_info(torch_obj: Any, index: int) -> CudaDevice:
    try:
        name = str(torch_obj.cuda.get_device_name(index))
    except Exception as exc:
        return CudaDevice(index=index, name="unknown", capability="unknown", error=str(exc))
    try:
        major, minor = torch_obj.cuda.get_device_capability(index)
        capability = f"{major}.{minor}"
    except Exception as exc:
        capability = "unknown"
        error = str(exc)
    else:
        error = None
    try:
        free_bytes, total_bytes = torch_obj.cuda.mem_get_info(index)
        free_mb = int(free_bytes // (1024 * 1024))
        total_mb = int(total_bytes // (1024 * 1024))
    except Exception:
        free_mb = None
        total_mb = None
    return CudaDevice(
        index=index,
        name=name,
        capability=capability,
        memory_free_mb=free_mb,
        memory_total_mb=total_mb,
        error=error,
    )


def _nvidia_smi_summary(
    nvidia_smi_path: str,
    runner: SmiRunner | None,
) -> tuple[str | None, str | None, list[NvidiaSmiGpu]]:
    run = runner or _run_nvidia_smi
    driver_version = None
    cuda_version = None
    gpus: list[NvidiaSmiGpu] = []
    try:
        text_result = run([nvidia_smi_path])
        text = text_result.stdout or ""
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
        if cuda_match:
            cuda_version = cuda_match.group(1)
        driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
        if driver_match:
            driver_version = driver_match.group(1)
    except Exception:
        pass
    try:
        query_result = run(
            [
                nvidia_smi_path,
                "--query-gpu=index,driver_version,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in (query_result.stdout or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            index_text, driver_text, name, total_text, used_text = parts[:5]
            driver_version = driver_version or driver_text
            gpus.append(
                NvidiaSmiGpu(
                    index=_int_or_zero(index_text),
                    name=name,
                    memory_total_mb=_int_or_none(total_text),
                    memory_used_mb=_int_or_none(used_text),
                    driver_version=driver_text or driver_version,
                )
            )
    except Exception:
        pass
    return driver_version, cuda_version, gpus


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def _infer_reason(
    *,
    torch_installed: bool,
    torch_cuda_build: bool,
    cuda_available: bool,
    nvidia_smi_path: str | None,
    cuda_visible_devices: str | None,
    warnings_list: list[str],
    init_error: str | None,
) -> str:
    if not torch_installed:
        return "missing_torch"
    if not torch_cuda_build:
        return "torch_cpu_only"
    hidden_values = {"", "-1", "none", "None"}
    if cuda_visible_devices in hidden_values:
        return "cuda_visible_devices_hidden"
    warning_blob = "\n".join(warnings_list + ([init_error] if init_error else []))
    if "driver" in warning_blob.lower() and "old" in warning_blob.lower():
        return "driver_too_old"
    if not nvidia_smi_path and not cuda_available:
        return "nvidia_smi_missing"
    if init_error:
        return "torch_cuda_initialization_error"
    if not cuda_available:
        return "cuda_unavailable"
    return "cuda_available"


def _recommendations(reason: str, cuda_available: bool) -> list[str]:
    if cuda_available:
        return ["CUDA is available. GPU execution can be requested with --device auto or --device cuda."]
    if reason == "missing_torch":
        return ["Install the local runtime dependencies in the conda environment, then retry cuda-status."]
    if reason == "torch_cpu_only":
        return ["Current PyTorch appears to be CPU-only. Install a PyTorch build matching the local CUDA/driver stack."]
    if reason == "driver_too_old":
        return ["The NVIDIA driver appears too old for this PyTorch CUDA build. Update the driver or use a matching PyTorch build."]
    if reason == "cuda_visible_devices_hidden":
        return ["CUDA_VISIBLE_DEVICES hides GPUs. Unset or adjust it if GPU use is intended."]
    if reason == "nvidia_smi_missing":
        return ["nvidia-smi is not visible. Check the NVIDIA driver installation. CPU fallback remains available."]
    return ["CUDA is unavailable in this process. Continue with CPU fallback or fix the driver/PyTorch mismatch."]


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: str) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else 0
