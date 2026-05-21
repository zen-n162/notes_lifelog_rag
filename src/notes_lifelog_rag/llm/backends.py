from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Protocol

from notes_lifelog_rag.llm.json_utils import LenientJSONError, parse_json_lenient
from notes_lifelog_rag.llm.mock import MockLLMBackend
from notes_lifelog_rag.models.status import has_module, resolve_model_entry, status_for_model
from notes_lifelog_rag.runtime.device import (
    DeviceInfo,
    autocast_context,
    effective_dtype,
    resolve_device,
    torch_dtype,
)


class LLMBackend(Protocol):
    model_name: str

    def is_available(self) -> bool:
        ...

    def availability_error(self) -> str | None:
        ...

    def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
        ...


@dataclass
class DisabledLLMBackend:
    model_name: str = "disabled"
    reason: str = "LLM backend disabled"

    def is_available(self) -> bool:
        return False

    def availability_error(self) -> str | None:
        return self.reason

    def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
        raise RuntimeError(self.reason)


@dataclass
class LocalTransformersLLMBackend:
    model_name: str
    model_path: Path
    device_info: DeviceInfo | None = None
    dtype: str = "auto"
    local_files_only: bool = True
    max_new_tokens: int = 512
    _runtime: Any = field(default=None, init=False, repr=False)
    _uses_device_map: bool = field(default=False, init=False, repr=False)

    def is_available(self) -> bool:
        return self.availability_error() is None

    def availability_error(self) -> str | None:
        if not self.model_path.exists():
            return f"text generation model path does not exist: {self.model_path}"
        if has_module("transformers") and has_module("torch"):
            return None
        return "missing runtime dependency: transformers+torch"

    def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
        payload, _raw_output = self.generate_with_raw(task_name, note, categories=categories)
        return payload

    def generate_with_raw(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> tuple[dict, str]:
        error = self.availability_error()
        if error:
            raise RuntimeError(error)
        prompt = _prompt_for(task_name, note, categories or [])
        tokenizer, model, torch = self._load_runtime()
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        device_info = self._device_info()
        if hasattr(inputs, "to"):
            inputs = inputs.to(device_info.torch_device)
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["pad_token_id"] = eos_token_id
        with torch.inference_mode(), autocast_context(torch, device_info, dtype_value):
            output_ids = model.generate(**inputs, **generate_kwargs)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return _parse_json_object(generated), generated

    def _load_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        device_info = self._device_info()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=self.local_files_only, trust_remote_code=True
        )
        dtype_value = torch_dtype(torch, self.dtype, device_info)
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
            "torch_dtype": dtype_value,
        }
        if device_info.is_cuda and device_info.requested_device == "auto" and has_module("accelerate"):
            kwargs["device_map"] = "auto"
            self._uses_device_map = True
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **kwargs,
        )
        if not self._uses_device_map:
            model.to(device_info.torch_device)
        model.eval()
        self._runtime = (tokenizer, model, torch)
        return self._runtime

    def _device_info(self) -> DeviceInfo:
        if self.device_info is None:
            self.device_info = resolve_device("auto", dtype=self.dtype)
        return self.device_info

    def device_label(self) -> str:
        info = self._device_info()
        return f"{info.resolved_device} / {effective_dtype(self.dtype, info)}"


def get_llm_backend(
    backend: str = "auto",
    model_name: str | None = None,
    *,
    allow_mock_fallback: bool = True,
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    max_new_tokens: int = 512,
) -> LLMBackend:
    if backend == "mock":
        return MockLLMBackend()
    if backend in {"none", "disabled"}:
        return DisabledLLMBackend(reason="LLM backend disabled by option")
    entry = resolve_model_entry("text_generation", model_name)
    if not entry:
        return MockLLMBackend() if allow_mock_fallback else DisabledLLMBackend(reason="no text generation model configured")
    status = status_for_model("text_generation", entry.get("name"), device_info=device_info)
    if status and status.enabled:
        return LocalTransformersLLMBackend(
            model_name=entry["name"],
            model_path=Path(entry["path"]),
            device_info=device_info,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    if backend == "auto" and allow_mock_fallback:
        return MockLLMBackend()
    reason = status.reason if status else "text generation model is not enabled"
    return DisabledLLMBackend(model_name=entry.get("name", "disabled"), reason=reason)


def _prompt_for(task_name: str, note: dict, categories: list[str]) -> str:
    safe_body = str(note.get("body") or "")[:6000]
    base = (
        "あなたはローカル専用のApple Notesライフログ分析器です。"
        "外部情報を使わず、原文に支持された内容だけをJSONで返してください。"
        "推論は控えめに書き、evidenceには短いquoteを必ず含めてください。\n\n"
        f"note_id: {note.get('id')}\n"
        f"title: {note.get('title')}\n"
        f"body:\n{safe_body}\n\n"
    )
    if task_name == "summary":
        return base + (
            "次のJSON objectのみを返してください: "
            "{generated_title, one_line_summary, detailed_summary, important_points, "
            "revisit_reason, confidence, evidence}"
        )
    if task_name == "categories":
        return base + (
            "候補カテゴリ: " + ", ".join(categories) + "\n"
            "次のJSON objectのみを返してください: {categories:[{name, confidence, evidence}]}"
        )
    if task_name == "events":
        return base + (
            "次のJSON objectのみを返してください: "
            "{events:[{title, summary, event_type, event_date, date_label, date_confidence, "
            "importance, confidence, evidence}]}"
        )
    if task_name == "thoughts":
        return base + (
            "次のJSON objectのみを返してください: "
            "{thoughts:[{title, summary, thought_type, themes, emotion_label, emotion_intensity, "
            "date_label, importance, confidence, remember_reason, evidence}]}"
        )
    return base + "JSON objectのみを返してください。"


def build_prompt(task_name: str, note: dict, categories: list[str] | None = None) -> str:
    return _prompt_for(task_name, note, categories or [])


def _parse_json_object(text: str) -> dict:
    value = parse_json_lenient(text)
    if not isinstance(value, dict):
        raise LenientJSONError(
            "schema_validation_error",
            f"Expected a JSON object but got {type(value).__name__}.",
            raw_output=text,
        )
    return value
