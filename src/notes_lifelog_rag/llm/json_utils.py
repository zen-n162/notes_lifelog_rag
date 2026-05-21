from __future__ import annotations

import json
import re
from typing import Any


class LenientJSONError(ValueError):
    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        raw_output: str = "",
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.raw_output = raw_output
        self.original_exception = original_exception


def strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    value = re.sub(r"^```(?:json|JSON)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_object(text: str) -> str | None:
    return _extract_balanced(text, "{", "}")


def extract_json_array(text: str) -> str | None:
    return _extract_balanced(text, "[", "]")


def repair_json_text(text: str) -> str:
    value = strip_code_fences(text)
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "＂": '"',
        "’": "'",
        "，": ",",
        "：": ":",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r",\s*([}\]])", r"\1", value)
    return value.strip()


def parse_json_lenient(text: str) -> Any:
    raw = text or ""
    if not raw.strip():
        raise LenientJSONError("empty_output", "LLM returned empty output.", raw_output=raw)
    candidates = _candidate_json_texts(raw)
    last_exc: Exception | None = None
    for candidate in candidates:
        repaired = repair_json_text(candidate)
        if not repaired:
            continue
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
    error_type = classify_json_error(raw, last_exc)
    message = str(last_exc) if last_exc else "No JSON object or array was found in the model output."
    raise LenientJSONError(error_type, message, raw_output=raw, original_exception=last_exc)


def classify_json_error(text: str, exception: Exception | None = None) -> str:
    value = (text or "").strip()
    if not value:
        return "empty_output"
    if _looks_truncated(value):
        return "truncated_output"
    if isinstance(exception, json.JSONDecodeError):
        if exception.pos >= max(len(value) - 3, 0):
            return "truncated_output"
        return "json_parse_error"
    return "json_parse_error"


def _candidate_json_texts(text: str) -> list[str]:
    stripped = strip_code_fences(text)
    candidates = [stripped]
    obj = extract_json_object(stripped)
    if obj and obj not in candidates:
        candidates.append(obj)
    arr = extract_json_array(stripped)
    if arr and arr not in candidates:
        candidates.append(arr)
    return candidates


def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _looks_truncated(text: str) -> bool:
    stack: list[str] = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in {"}", "]"}:
            if not stack or stack.pop() != char:
                return False
    return bool(in_string or stack)
