from __future__ import annotations

import subprocess

import pytest

from notes_lifelog_rag.runtime.cuda import CudaStatus, collect_cuda_status
from notes_lifelog_rag.runtime.device import DeviceResolutionError, resolve_device


def _status(cuda_available: bool, device_count: int = 0) -> CudaStatus:
    return CudaStatus(
        python_version="3.11",
        torch_installed=True,
        torch_version="test",
        torch_cuda_version="12.1",
        torch_cuda_build=True,
        cuda_available=cuda_available,
        device_count=device_count,
        likely_reason="cuda_available" if cuda_available else "cuda_unavailable",
    )


def test_resolve_device_auto_uses_cuda_when_available() -> None:
    info = resolve_device("auto", cuda_status=_status(True, 1))
    assert info.resolved_device == "cuda:0"
    assert info.is_cuda is True


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable() -> None:
    info = resolve_device("auto", cuda_status=_status(False, 0))
    assert info.resolved_device == "cpu"
    assert info.warning


def test_resolve_device_cuda_required_fails_when_unavailable() -> None:
    with pytest.raises(DeviceResolutionError):
        resolve_device("cuda", require_cuda=True, cuda_status=_status(False, 0))


def test_resolve_device_invalid_cuda_index_falls_back_or_fails() -> None:
    fallback = resolve_device("cuda:3", cuda_status=_status(True, 1))
    assert fallback.resolved_device == "cpu"
    with pytest.raises(DeviceResolutionError):
        resolve_device("cuda:3", require_cuda=True, cuda_status=_status(True, 1))


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1 if self._available else 0

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (8, 9)

    def mem_get_info(self, index: int) -> tuple[int, int]:
        return (1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)


class _FakeTorch:
    __version__ = "2.test"

    class version:
        cuda = "12.1"

    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)


def test_collect_cuda_status_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr("notes_lifelog_rag.runtime.cuda.shutil.which", lambda name: None)
    status = collect_cuda_status(torch_module=_FakeTorch(False))
    assert status.torch_installed is True
    assert status.cuda_available is False
    assert status.nvidia_smi_path is None


def test_collect_cuda_status_with_cuda_available_and_mocked_smi(monkeypatch) -> None:
    monkeypatch.setattr("notes_lifelog_rag.runtime.cuda.shutil.which", lambda name: "/usr/bin/nvidia-smi")

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,driver_version,name,memory.total,memory.used" in args:
            stdout = "0, 555.1, Fake GPU, 24576, 1024\n"
        else:
            stdout = "| NVIDIA-SMI 555.1 Driver Version: 555.1 CUDA Version: 12.5 |\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    status = collect_cuda_status(torch_module=_FakeTorch(True), nvidia_smi_runner=runner)
    assert status.cuda_available is True
    assert status.device_count == 1
    assert status.devices[0].name == "Fake GPU 0"
    assert status.nvidia_smi_gpus[0].memory_total_mb == 24576
