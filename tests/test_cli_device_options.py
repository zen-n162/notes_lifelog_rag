from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from notes_lifelog_rag.analysis.service import AnalysisSummary
from notes_lifelog_rag.cli import app
from notes_lifelog_rag.embeddings.engines import MockEmbeddingBackend
from notes_lifelog_rag.runtime.cuda import CudaStatus


runner = CliRunner()


def _cuda_status(cuda_available: bool = False) -> CudaStatus:
    return CudaStatus(
        python_version="3.11",
        torch_installed=True,
        torch_version="test",
        torch_cuda_version="12.1",
        torch_cuda_build=True,
        cuda_available=cuda_available,
        device_count=1 if cuda_available else 0,
        likely_reason="cuda_available" if cuda_available else "cuda_unavailable",
    )


def test_build_embeddings_cli_accepts_device_cpu(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_get_embedding_backend(*args, **kwargs):
        captured["device_info"] = kwargs.get("device_info")
        return MockEmbeddingBackend()

    monkeypatch.setattr("notes_lifelog_rag.cli.get_embedding_backend", fake_get_embedding_backend)
    result = runner.invoke(
        app,
        [
            "build-embeddings",
            "--backend",
            "mock",
            "--device",
            "cpu",
            "--dry-run",
            "--db",
            str(tmp_path / "notes.db"),
        ],
    )
    assert result.exit_code == 0
    assert captured["device_info"].resolved_device == "cpu"


def test_build_embeddings_cli_auto_falls_back_to_cpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("notes_lifelog_rag.runtime.device.collect_cuda_status", lambda: _cuda_status(False))
    result = runner.invoke(
        app,
        [
            "build-embeddings",
            "--backend",
            "mock",
            "--device",
            "auto",
            "--dry-run",
            "--show-device",
            "--db",
            str(tmp_path / "notes.db"),
        ],
    )
    assert result.exit_code == 0
    assert "Resolved device" in result.output
    assert "cpu" in result.output


def test_build_embeddings_cli_require_cuda_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("notes_lifelog_rag.runtime.device.collect_cuda_status", lambda: _cuda_status(False))
    result = runner.invoke(
        app,
        [
            "build-embeddings",
            "--backend",
            "mock",
            "--device",
            "cuda",
            "--require-cuda",
            "--dry-run",
            "--db",
            str(tmp_path / "notes.db"),
        ],
    )
    assert result.exit_code == 1
    assert "Device error" in result.output


def test_analyze_all_cli_default_limit_is_none(monkeypatch) -> None:
    captured = {}

    def fake_analyze_all(*args, **kwargs):
        captured["limit"] = kwargs.get("limit")
        return [AnalysisSummary(task_name="summary", model_name="mock", dry_run=True)]

    monkeypatch.setattr("notes_lifelog_rag.cli.analyze_all", fake_analyze_all)
    result = runner.invoke(app, ["analyze-all", "--backend", "mock", "--device", "cpu", "--dry-run"])

    assert result.exit_code == 0
    assert captured["limit"] is None


def test_analyze_all_cli_explicit_limit_is_passed(monkeypatch) -> None:
    captured = {}

    def fake_analyze_all(*args, **kwargs):
        captured["limit"] = kwargs.get("limit")
        return [AnalysisSummary(task_name="summary", model_name="mock", dry_run=True, selected_notes=10)]

    monkeypatch.setattr("notes_lifelog_rag.cli.analyze_all", fake_analyze_all)
    result = runner.invoke(app, ["analyze-all", "--backend", "mock", "--device", "cpu", "--dry-run", "--limit", "10"])

    assert result.exit_code == 0
    assert captured["limit"] == 10


def test_analyze_all_cli_require_cuda_fails(monkeypatch) -> None:
    monkeypatch.setattr("notes_lifelog_rag.runtime.device.collect_cuda_status", lambda: _cuda_status(False))
    result = runner.invoke(
        app,
        [
            "analyze-all",
            "--backend",
            "mock",
            "--device",
            "cuda",
            "--require-cuda",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "Device error" in result.output
