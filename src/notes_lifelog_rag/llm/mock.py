from __future__ import annotations

import re

from notes_lifelog_rag.utils.dates import parse_date_label


class MockLLMBackend:
    """Deterministic local-only backend for tests, smoke runs, and fallback use."""

    model_name = "mock-local-llm"

    def is_available(self) -> bool:
        return True

    def availability_error(self) -> str | None:
        return None

    def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
        body = str(note.get("body") or "")
        title = str(note.get("title") or "無題")
        quote = _short_quote(body)
        if task_name == "summary":
            return {
                "generated_title": title,
                "one_line_summary": _one_line(body, fallback=f"{title}に関するメモです。"),
                "detailed_summary": _detailed_summary(body, title),
                "important_points": _important_points(body),
                "revisit_reason": "関連する過去メモを確認したいときに手がかりになります。",
                "confidence": 0.55,
                "evidence": [{"note_id": note["id"], "quote": quote}],
            }
        if task_name == "categories":
            return {"categories": _categories(body, title, categories or [] , note["id"], quote)}
        if task_name == "events":
            return {"events": _events(body, title, note["id"], quote)}
        if task_name == "thoughts":
            return {"thoughts": _thoughts(body, title, note["id"], quote)}
        return {"evidence": [{"note_id": note["id"], "quote": quote}], "confidence": 0.0}


def _short_quote(text: str, max_chars: int = 80) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    line = re.sub(r"\s+", " ", line)
    return line[:max_chars]


def _one_line(text: str, *, fallback: str) -> str:
    line = _short_quote(text, max_chars=90)
    if not line:
        return fallback
    return line if line.endswith("。") else f"{line}。"


def _detailed_summary(text: str, title: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"{title}の内容を記録したメモです。"
    joined = " / ".join(lines[:3])
    return joined[:220]


def _important_points(text: str) -> list[str]:
    points = []
    for line in text.splitlines():
        stripped = line.strip(" -・\t")
        if len(stripped) >= 8:
            points.append(stripped[:120])
        if len(points) >= 3:
            break
    return points or ["原文メモを確認する価値があります。"]


def _categories(body: str, title: str, categories: list[str], note_id: str, quote: str) -> list[dict]:
    haystack = f"{title}\n{body}"
    selected: list[dict] = []
    keyword_map = {
        "研究": ["研究", "論文", "実験", "評価"],
        "修論": ["修論", "修士", "卒論"],
        "ハイパースペクトル": ["ハイパースペクトル", "HSI", "スペクトル"],
        "月面探査": ["月面", "月", "探査"],
        "機械学習": ["機械学習", "AI", "LLM", "モデル"],
        "アプリ開発": ["アプリ", "実装", "UI", "CLI"],
        "就職活動": ["就活", "ES", "面接", "企業"],
        "予定・タスク": ["やること", "予定", "タスク", "TODO"],
        "日記・出来事": ["今日", "昨日", "旅行", "行った"],
        "感情・内省": ["思う", "考え", "不安", "嬉しい", "後悔"],
        "ブレイキン": ["ダンス", "ブレイク", "ブレイキン"],
    }
    for category in categories:
        keywords = keyword_map.get(category, [category])
        if any(keyword and keyword in haystack for keyword in keywords):
            selected.append(
                {
                    "name": category,
                    "confidence": 0.62,
                    "evidence": [{"note_id": note_id, "quote": quote}],
                }
            )
    if not selected and "その他" in categories:
        selected.append({"name": "その他", "confidence": 0.4, "evidence": [{"note_id": note_id, "quote": quote}]})
    return selected[:5]


def _events(body: str, title: str, note_id: str, quote: str) -> list[dict]:
    date_result = parse_date_label(f"{title}\n{body}")
    if not quote:
        return []
    event_type = "plan" if any(token in body for token in ("予定", "やる", "したい", "行く")) else "note"
    return [
        {
            "title": title[:80],
            "summary": _one_line(body, fallback=f"{title}に関する出来事・予定の可能性があります。"),
            "event_type": event_type,
            "event_date": date_result.iso_date,
            "date_label": date_result.date_label,
            "date_confidence": date_result.confidence,
            "importance": 0.45,
            "confidence": 0.5,
            "evidence": [{"note_id": note_id, "quote": quote}],
        }
    ]


def _thoughts(body: str, title: str, note_id: str, quote: str) -> list[dict]:
    if not any(token in body for token in ("思", "考え", "必要", "かもしれない", "不安", "目標", "決め", "気づ")):
        return []
    return [
        {
            "title": f"{title[:60]} の考え",
            "summary": _one_line(body, fallback=f"{title}に関する考えが含まれている可能性があります。"),
            "thought_type": "reflection",
            "themes": [],
            "emotion_label": None,
            "emotion_intensity": None,
            "date_label": parse_date_label(f"{title}\n{body}").date_label,
            "importance": 0.5,
            "confidence": 0.48,
            "remember_reason": "当時の考え方や判断材料を振り返る手がかりになります。",
            "evidence": [{"note_id": note_id, "quote": quote}],
        }
    ]
