from __future__ import annotations

import hashlib
import json
import math
import re


def stable_hash_embedding(text: str, dimension: int = 64) -> list[float]:
    vector = [0.0 for _ in range(dimension)]
    for token in _features(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 0.5 + (digest[5] / 255.0)
        vector[index] += sign * weight
    return normalize(vector)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def vector_to_json(vector: list[float]) -> str:
    return json.dumps([round(float(value), 8) for value in vector], separators=(",", ":"))


def vector_from_json(payload: str | None) -> list[float]:
    if not payload:
        return []
    values = json.loads(payload)
    return [float(value) for value in values]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _features(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if not normalized:
        return []
    features: list[str] = []
    words = re.findall(r"[a-z0-9_]+|[ぁ-んァ-ン一-龥ー]+", normalized)
    features.extend(words)
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) <= 2:
        features.append(compact)
    else:
        features.extend(compact[index : index + 2] for index in range(len(compact) - 1))
        features.extend(compact[index : index + 3] for index in range(len(compact) - 2))
    return features

