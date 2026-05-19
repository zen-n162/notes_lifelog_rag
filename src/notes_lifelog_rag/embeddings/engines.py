from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Protocol

from notes_lifelog_rag.embeddings.vector import normalize, stable_hash_embedding
from notes_lifelog_rag.models.status import has_module, resolve_model_entry, status_for_model
from notes_lifelog_rag.runtime.device import (
    DeviceInfo,
    autocast_context,
    effective_dtype,
    resolve_device,
    torch_dtype,
)


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model_name: str
    dimension: int
    status: str = "success"
    error_message: str | None = None


class EmbeddingBackend(Protocol):
    model_name: str

    def is_available(self) -> bool:
        ...

    def availability_error(self) -> str | None:
        ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


@dataclass
class DisabledEmbeddingBackend:
    model_name: str = "disabled"
    reason: str = "embedding backend disabled"

    def is_available(self) -> bool:
        return False

    def availability_error(self) -> str | None:
        return self.reason

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult([], self.model_name, 0, status="disabled", error_message=self.reason)
            for _ in texts
        ]


@dataclass
class MockEmbeddingBackend:
    model_name: str = "mock-local-embedding"
    dimension: int = 64

    def is_available(self) -> bool:
        return True

    def availability_error(self) -> str | None:
        return None

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(stable_hash_embedding(text, self.dimension), self.model_name, self.dimension)
            for text in texts
        ]


@dataclass
class LocalEmbeddingBackend:
    model_name: str
    model_path: Path
    device_info: DeviceInfo | None = None
    dtype: str = "auto"
    batch_size: int = 16
    local_files_only: bool = True
    _runtime: Any = field(default=None, init=False, repr=False)
    _runtime_kind: str | None = field(default=None, init=False, repr=False)

    def is_available(self) -> bool:
        return self.availability_error() is None

    def availability_error(self) -> str | None:
        if not self.model_path.exists():
            return f"embedding model path does not exist: {self.model_path}"
        if has_module("sentence_transformers") or (has_module("transformers") and has_module("torch")):
            return None
        return "missing runtime dependency: sentence_transformers or transformers+torch"

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        error = self.availability_error()
        if error:
            return [
                EmbeddingResult([], self.model_name, 0, status="disabled", error_message=error)
                for _ in texts
            ]
        try:
            vectors = self._encode(texts, batch_size=self.batch_size)
        except Exception as exc:
            if not _is_oom(exc) or len(texts) <= 1:
                return self._failed_results(texts, f"local embedding failed: {exc}")
            retry_batch_size = max(1, self.batch_size // 2)
            try:
                vectors = self._encode(texts, batch_size=retry_batch_size)
            except Exception as retry_exc:  # pragma: no cover - depends on local ML runtime.
                message = f"local embedding failed after OOM retry batch_size={retry_batch_size}: {retry_exc}"
                return self._failed_results(texts, message)
            self.batch_size = retry_batch_size
        try:
            return [
                EmbeddingResult(normalize([float(value) for value in vector]), self.model_name, len(vector))
                for vector in vectors
            ]
        except Exception as exc:
            return self._failed_results(texts, f"local embedding failed: {exc}")

    def _encode(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        runtime = self._load_runtime()
        if self._runtime_kind == "sentence_transformers":
            encoded = runtime.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return _to_nested_float_lists(encoded)

        tokenizer, model, torch = runtime
        encoded_batches: list[list[float]] = []
        device_info = self._device_info()
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device_info.torch_device)
            with torch.inference_mode(), autocast_context(torch, device_info, dtype_value):
                outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            encoded_batches.extend(_to_nested_float_lists(pooled))
        return encoded_batches

    def _load_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        device_info = self._device_info()
        if has_module("sentence_transformers"):
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "device": device_info.torch_device,
            }
            try:
                self._runtime = SentenceTransformer(
                    str(self.model_path),
                    local_files_only=self.local_files_only,
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("local_files_only", None)
                self._runtime = SentenceTransformer(str(self.model_path), **kwargs)
            self._runtime_kind = "sentence_transformers"
            return self._runtime

        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=self.local_files_only, trust_remote_code=True
        )
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        model = AutoModel.from_pretrained(
            str(self.model_path),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            torch_dtype=dtype_value,
        )
        model.to(device_info.torch_device)
        model.eval()
        self._runtime = (tokenizer, model, torch)
        self._runtime_kind = "transformers_mean_pooling"
        return self._runtime

    def _device_info(self) -> DeviceInfo:
        if self.device_info is None:
            self.device_info = resolve_device("auto", dtype=self.dtype)
        return self.device_info

    def device_label(self) -> str:
        info = self._device_info()
        return f"{info.resolved_device} / {effective_dtype(self.dtype, info)}"

    def _failed_results(self, texts: list[str], message: str) -> list[EmbeddingResult]:
        return [
            EmbeddingResult([], self.model_name, 0, status="failed", error_message=message)
            for _ in texts
        ]


def get_embedding_backend(
    backend: str = "auto",
    model_name: str | None = None,
    *,
    allow_mock_fallback: bool = False,
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 16,
) -> EmbeddingBackend:
    if backend == "mock":
        return MockEmbeddingBackend()
    if backend in {"none", "disabled"}:
        return DisabledEmbeddingBackend(reason="embedding backend disabled by option")

    entry = resolve_model_entry("embedding", model_name)
    if not entry:
        return MockEmbeddingBackend() if allow_mock_fallback else DisabledEmbeddingBackend(reason="no embedding model configured")

    status = status_for_model("embedding", entry.get("name"), device_info=device_info)
    if status and status.enabled:
        return LocalEmbeddingBackend(
            model_name=entry["name"],
            model_path=Path(entry["path"]),
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
        )
    if backend == "auto" and allow_mock_fallback:
        return MockEmbeddingBackend()
    reason = status.reason if status else "embedding model is not enabled"
    return DisabledEmbeddingBackend(model_name=entry.get("name", "disabled"), reason=reason)


def _to_nested_float_lists(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas" in text
