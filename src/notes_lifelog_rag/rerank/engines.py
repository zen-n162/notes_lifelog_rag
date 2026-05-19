from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Protocol

from notes_lifelog_rag.models.status import has_module, resolve_model_entry, status_for_model
from notes_lifelog_rag.runtime.device import DeviceInfo, autocast_context, resolve_device, torch_dtype

_RERANKER_RUNTIME_CACHE: dict[tuple[str, str, str, str], tuple[Any, str]] = {}


@dataclass(frozen=True)
class RerankCandidate:
    id: str
    text: str
    original_score: float


@dataclass(frozen=True)
class RerankResult:
    id: str
    score: float
    status: str = "success"
    error_message: str | None = None


class Reranker(Protocol):
    model_name: str

    def is_available(self) -> bool:
        ...

    def availability_error(self) -> str | None:
        ...

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        ...


@dataclass
class DisabledReranker:
    model_name: str = "disabled"
    reason: str = "reranker disabled"

    def is_available(self) -> bool:
        return False

    def availability_error(self) -> str | None:
        return self.reason

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        return [
            RerankResult(candidate.id, candidate.original_score, status="disabled", error_message=self.reason)
            for candidate in candidates
        ]


@dataclass
class MockReranker:
    model_name: str = "mock-local-reranker"

    def is_available(self) -> bool:
        return True

    def availability_error(self) -> str | None:
        return None

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        compact_query = query.casefold().replace(" ", "").replace("　", "")
        query_terms = [query.casefold(), compact_query]
        results: list[RerankResult] = []
        for candidate in candidates:
            compact_text = candidate.text.casefold().replace(" ", "").replace("　", "")
            lexical = 0.0
            for term in query_terms:
                if term and term in compact_text:
                    lexical += 0.35
            char_hits = sum(1 for char in set(compact_query) if char in compact_text)
            lexical += min(0.25, char_hits * 0.025)
            score = min(1.0, max(0.0, candidate.original_score) * 0.55 + lexical)
            results.append(RerankResult(candidate.id, round(score, 5)))
        return sorted(results, key=lambda item: item.score, reverse=True)


@dataclass
class LocalReranker:
    model_name: str
    model_path: Path
    device_info: DeviceInfo | None = None
    dtype: str = "auto"
    batch_size: int = 8
    local_files_only: bool = True
    _runtime: Any = field(default=None, init=False, repr=False)
    _runtime_kind: str | None = field(default=None, init=False, repr=False)

    def is_available(self) -> bool:
        return self.availability_error() is None

    def availability_error(self) -> str | None:
        if not self.model_path.exists():
            return f"reranker model path does not exist: {self.model_path}"
        if has_module("sentence_transformers") or (has_module("transformers") and has_module("torch")):
            return None
        return "missing runtime dependency: sentence_transformers or transformers+torch"

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        error = self.availability_error()
        if error:
            return [
                RerankResult(candidate.id, candidate.original_score, status="disabled", error_message=error)
                for candidate in candidates
            ]
        try:
            scores = self._score_pairs([(query, candidate.text) for candidate in candidates])
        except Exception as exc:  # pragma: no cover - depends on local ML runtime.
            message = f"local reranker failed: {exc}"
            return [
                RerankResult(candidate.id, candidate.original_score, status="failed", error_message=message)
                for candidate in candidates
            ]
        return sorted(
            [
                RerankResult(candidate.id, float(score))
                for candidate, score in zip(candidates, scores)
            ],
            key=lambda item: item.score,
            reverse=True,
        )

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        runtime = self._load_runtime()
        if self._runtime_kind == "sentence_transformers_cross_encoder":
            try:
                values = runtime.predict(pairs, batch_size=self.batch_size)
            except TypeError:
                values = runtime.predict(pairs)
            return [float(value) for value in values]

        tokenizer, model, torch = runtime
        scores: list[float] = []
        device_info = self._device_info()
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        for query, text in pairs:
            inputs = tokenizer(query, text, truncation=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device_info.torch_device)
            with torch.inference_mode(), autocast_context(torch, device_info, dtype_value):
                outputs = model(**inputs)
            logits = outputs.logits
            value = logits[0][-1] if logits.ndim > 1 else logits[0]
            scores.append(float(value.detach().cpu().item()))
        return scores

    def _load_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        device_info = self._device_info()
        cache_key = (self.model_name, str(self.model_path), device_info.torch_device, self.dtype)
        cached = _RERANKER_RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            self._runtime, self._runtime_kind = cached
            return self._runtime
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if has_module("sentence_transformers"):
            from sentence_transformers import CrossEncoder

            try:
                self._runtime = CrossEncoder(
                    str(self.model_path),
                    local_files_only=self.local_files_only,
                    trust_remote_code=True,
                    device=device_info.torch_device,
                )
            except TypeError:
                self._runtime = CrossEncoder(
                    str(self.model_path),
                    trust_remote_code=True,
                    device=device_info.torch_device,
                )
            self._runtime_kind = "sentence_transformers_cross_encoder"
            _RERANKER_RUNTIME_CACHE[cache_key] = (self._runtime, self._runtime_kind)
            return self._runtime

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=self.local_files_only, trust_remote_code=True
        )
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            torch_dtype=dtype_value,
        )
        model.to(device_info.torch_device)
        model.eval()
        self._runtime = (tokenizer, model, torch)
        self._runtime_kind = "transformers_sequence_classification"
        _RERANKER_RUNTIME_CACHE[cache_key] = (self._runtime, self._runtime_kind)
        return self._runtime

    def _device_info(self) -> DeviceInfo:
        if self.device_info is None:
            self.device_info = resolve_device("auto", dtype=self.dtype)
        return self.device_info


def get_reranker(
    backend: str = "auto",
    model_name: str | None = None,
    *,
    allow_mock_fallback: bool = False,
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 8,
) -> Reranker:
    if backend == "mock":
        return MockReranker()
    if backend in {"none", "disabled"}:
        return DisabledReranker(reason="reranker disabled by option")
    entry = resolve_model_entry("reranker", model_name)
    if not entry:
        return MockReranker() if allow_mock_fallback else DisabledReranker(reason="no reranker model configured")
    status = status_for_model("reranker", entry.get("name"), device_info=device_info)
    if status and status.enabled:
        return LocalReranker(
            model_name=entry["name"],
            model_path=Path(entry["path"]),
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
        )
    if backend == "auto" and allow_mock_fallback:
        return MockReranker()
    reason = status.reason if status else "reranker model is not enabled"
    return DisabledReranker(model_name=entry.get("name", "disabled"), reason=reason)
