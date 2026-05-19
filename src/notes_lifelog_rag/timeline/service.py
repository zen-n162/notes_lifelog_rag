from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from notes_lifelog_rag.db.schema import connect, init_db


@dataclass(frozen=True)
class TimelineItem:
    item_type: str
    note_id: str
    title: str
    summary: str
    date_label: str
    date_confidence: str
    source_title: str
    source_path: str
    confidence: float
    importance: float
    evidence: list[dict[str, str]]


@dataclass(frozen=True)
class ReflectionReport:
    month: str
    main_events: list[str]
    main_thoughts: list[str]
    important_changes: list[str]
    rediscovery_points: list[str]
    reminder_messages: list[str]
    evidence: list[dict[str, str]]
    confidence: float
    importance: float
    coverage: dict[str, float]
    quality_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_timeline(month: str | None = None, *, db_path: str | Path | None = None, limit: int = 100) -> list[TimelineItem]:
    init_db(db_path)
    selected_month = _normalize_month(month)
    with connect(db_path) as conn:
        event_rows = conn.execute(
            """
            SELECT
                'event' AS item_type,
                events.note_id,
                events.title,
                events.summary,
                COALESCE(events.event_date, events.date_label, '') AS date_key,
                COALESCE(events.date_label, events.event_date, '') AS date_label,
                COALESCE(events.date_confidence, 'unknown') AS date_confidence,
                COALESCE(events.confidence, 0.0) AS confidence,
                COALESCE(events.importance, 0.0) AS importance,
                COALESCE(events.evidence_json, '[]') AS evidence_json,
                notes.title AS source_title,
                notes.source_relative_path AS source_path
            FROM events
            JOIN notes ON notes.id = events.note_id
            WHERE (? IS NULL OR COALESCE(events.event_date, events.date_label, notes.note_date, notes.modified_at) LIKE ?)
            ORDER BY COALESCE(events.event_date, events.date_label, notes.note_date, notes.modified_at) ASC
            LIMIT ?
            """,
            (selected_month, f"{selected_month}%" if selected_month else None, limit),
        ).fetchall()
        thought_rows = conn.execute(
            """
            SELECT
                'thought' AS item_type,
                thoughts.note_id,
                thoughts.title,
                thoughts.summary,
                COALESCE(thoughts.date_label, notes.note_date, notes.modified_at, '') AS date_key,
                COALESCE(thoughts.date_label, notes.note_date, '') AS date_label,
                'low' AS date_confidence,
                COALESCE(thoughts.confidence, 0.0) AS confidence,
                COALESCE(thoughts.importance, 0.0) AS importance,
                COALESCE(thoughts.evidence_json, '[]') AS evidence_json,
                notes.title AS source_title,
                notes.source_relative_path AS source_path
            FROM thoughts
            JOIN notes ON notes.id = thoughts.note_id
            WHERE (? IS NULL OR COALESCE(thoughts.date_label, notes.note_date, notes.modified_at) LIKE ?)
            ORDER BY COALESCE(thoughts.date_label, notes.note_date, notes.modified_at) ASC
            LIMIT ?
            """,
            (selected_month, f"{selected_month}%" if selected_month else None, limit),
        ).fetchall()
        rows = list(event_rows) + list(thought_rows)
        if not rows:
            rows = conn.execute(
                """
                SELECT
                    'note' AS item_type,
                    notes.id AS note_id,
                    notes.title AS title,
                    COALESCE(note_summaries.one_line_summary, notes.title) AS summary,
                    COALESCE(notes.note_date, notes.modified_at, '') AS date_key,
                    COALESCE(notes.date_label, notes.note_date, '') AS date_label,
                    COALESCE(notes.date_confidence, 'unknown') AS date_confidence,
                    COALESCE(note_summaries.confidence, 0.35) AS confidence,
                    COALESCE(note_summaries.importance, 0.35) AS importance,
                    COALESCE(note_summaries.evidence_json, '[]') AS evidence_json,
                    notes.title AS source_title,
                    notes.source_relative_path AS source_path
                FROM notes
                LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
                WHERE (? IS NULL OR COALESCE(notes.note_date, notes.modified_at) LIKE ?)
                ORDER BY COALESCE(notes.note_date, notes.modified_at) ASC
                LIMIT ?
                """,
                (selected_month, f"{selected_month}%" if selected_month else None, limit),
            ).fetchall()
    items = [
        TimelineItem(
            item_type=row["item_type"],
            note_id=row["note_id"],
            title=row["title"],
            summary=row["summary"] or "",
            date_label=row["date_label"] or row["date_key"] or "日付不明",
            date_confidence=row["date_confidence"] or "unknown",
            source_title=row["source_title"],
            source_path=row["source_path"],
            confidence=float(row["confidence"] or 0.0),
            importance=float(row["importance"] or 0.0),
            evidence=_evidence(row["evidence_json"], row["note_id"]),
        )
        for row in sorted(rows, key=lambda item: str(item["date_key"] or ""))
    ]
    return items


def build_monthly_reflection(
    month: str | None = None,
    *,
    db_path: str | Path | None = None,
    force: bool = False,
) -> ReflectionReport:
    _ = force
    selected_month = _normalize_month(month) or _latest_month(db_path)
    items = build_timeline(selected_month, db_path=db_path, limit=200)
    events = [item for item in items if item.item_type in {"event", "note"}]
    thoughts = [item for item in items if item.item_type == "thought"]
    evidence: list[dict[str, str]] = []
    for item in sorted(items, key=lambda x: x.importance, reverse=True)[:6]:
        evidence.extend(item.evidence[:1] or [{"note_id": item.note_id, "quote": item.summary[:80]}])
    if not evidence and items:
        evidence.append({"note_id": items[0].note_id, "quote": items[0].summary[:80]})
    coverage = _reflection_coverage(selected_month, db_path)
    quality_warnings = _reflection_warnings(items, coverage)
    report = ReflectionReport(
        month=selected_month or "unknown",
        main_events=[_short(item.title, 80) for item in sorted(events, key=lambda x: x.importance, reverse=True)[:5]],
        main_thoughts=[_short(item.summary or item.title, 120) for item in sorted(thoughts, key=lambda x: x.importance, reverse=True)[:5]],
        important_changes=_important_changes(items),
        rediscovery_points=_rediscovery_points(items),
        reminder_messages=_reminder_messages(items) + quality_warnings[:2],
        evidence=evidence[:6],
        confidence=_average([item.confidence for item in items], default=0.25),
        importance=max([item.importance for item in items] or [0.25]),
        coverage=coverage,
        quality_warnings=quality_warnings,
    )
    _store_reflection(report, db_path=db_path)
    return report


def format_timeline_markdown(items: list[TimelineItem], *, month: str | None = None) -> str:
    title = f"## Timeline {month or ''}".strip()
    if not items:
        return f"{title}\n\nこの月のタイムライン候補はまだありません。"
    lines = [title, ""]
    for item in items:
        evidence = item.evidence[0]["quote"] if item.evidence else ""
        lines.extend(
            [
                f"### {item.date_label or '日付不明'} · {item.title}",
                f"- type: `{item.item_type}`",
                f"- summary: {item.summary}",
                f"- confidence: `{item.confidence:.2f}` / importance: `{item.importance:.2f}` / date_confidence: `{item.date_confidence}`",
                f"- evidence: {evidence}",
                f"- source: `{item.source_title}` (`{item.note_id[:12]}`)",
                "",
            ]
        )
    return "\n".join(lines)


def format_reflection_markdown(report: ReflectionReport) -> str:
    def section(title: str, values: list[str]) -> list[str]:
        if not values:
            return [f"### {title}", "- まだ十分な材料がありません。", ""]
        return [f"### {title}", *[f"- {value}" for value in values], ""]

    lines = [
        f"## {report.month} Reflection",
        f"confidence: `{report.confidence:.2f}` / importance: `{report.importance:.2f}`",
        f"coverage: notes `{report.coverage.get('notes', 0):.0f}`, events `{report.coverage.get('event_notes', 0):.0f}`, thoughts `{report.coverage.get('thought_notes', 0):.0f}`",
        "",
        *section("Main Events", report.main_events),
        *section("Main Thoughts", report.main_thoughts),
        *section("Important Changes", report.important_changes),
        *section("Rediscovery Points", report.rediscovery_points),
        *section("Reminder Messages", report.reminder_messages),
        "### Evidence",
    ]
    lines.extend([f"- `{item['note_id'][:12]}`: {item['quote']}" for item in report.evidence] or ["- なし"])
    if report.quality_warnings:
        lines.extend(["", "### Quality Warnings", *[f"- {item}" for item in report.quality_warnings]])
    return "\n".join(lines)


def _store_reflection(report: ReflectionReport, *, db_path: str | Path | None) -> None:
    init_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO monthly_reflections(month, summary_json, evidence_json, confidence, importance, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                summary_json = excluded.summary_json,
                evidence_json = excluded.evidence_json,
                confidence = excluded.confidence,
                importance = excluded.importance,
                updated_at = excluded.updated_at
            """,
            (
                report.month,
                json.dumps(report.to_dict(), ensure_ascii=False),
                json.dumps(report.evidence, ensure_ascii=False),
                report.confidence,
                report.importance,
                now,
                now,
            ),
        )


def _latest_month(db_path: str | Path | None) -> str | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT SUBSTR(COALESCE(note_date, modified_at, imported_at), 1, 7) AS month
            FROM notes
            WHERE COALESCE(note_date, modified_at, imported_at) IS NOT NULL
            ORDER BY COALESCE(note_date, modified_at, imported_at) DESC
            LIMIT 1
            """
        ).fetchone()
    return row["month"] if row and row["month"] else None


def _important_changes(items: list[TimelineItem]) -> list[str]:
    candidates = [item for item in items if item.importance >= 0.5]
    return [f"{item.title} が後で見返す価値の高い記録として残っています。" for item in candidates[:4]]


def _rediscovery_points(items: list[TimelineItem]) -> list[str]:
    return [
        f"{item.source_title}: {item.summary[:90]}"
        for item in sorted(items, key=lambda x: x.confidence + x.importance, reverse=True)[:4]
    ]


def _reminder_messages(items: list[TimelineItem]) -> list[str]:
    if not items:
        return ["この月の材料が少ないため、まず notes の import と analyze-all を実行すると振り返りが豊かになります。"]
    return [
        "当時の事実と考えを分けて読み返すと、次の判断材料を取り出しやすくなります。",
        "importance が高い項目から元メモを開き、現在の関心との接点を確認してください。",
    ]


def _reflection_coverage(month: str | None, db_path: str | Path | None) -> dict[str, float]:
    init_db(db_path)
    with connect(db_path) as conn:
        params = (f"{month}%" if month else "%",)
        notes = conn.execute(
            "SELECT COUNT(*) AS count FROM notes WHERE COALESCE(note_date, modified_at, imported_at, '') LIKE ?",
            params,
        ).fetchone()["count"]
        summary_notes = conn.execute(
            """
            SELECT COUNT(DISTINCT notes.id) AS count
            FROM notes JOIN note_summaries ON note_summaries.note_id = notes.id
            WHERE COALESCE(notes.note_date, notes.modified_at, notes.imported_at, '') LIKE ?
            """,
            params,
        ).fetchone()["count"]
        event_notes = conn.execute(
            """
            SELECT COUNT(DISTINCT notes.id) AS count
            FROM notes JOIN events ON events.note_id = notes.id
            WHERE COALESCE(events.event_date, events.date_label, notes.note_date, notes.modified_at, '') LIKE ?
            """,
            params,
        ).fetchone()["count"]
        thought_notes = conn.execute(
            """
            SELECT COUNT(DISTINCT notes.id) AS count
            FROM notes JOIN thoughts ON thoughts.note_id = notes.id
            WHERE COALESCE(thoughts.date_label, notes.note_date, notes.modified_at, '') LIKE ?
            """,
            params,
        ).fetchone()["count"]
    return {
        "notes": float(notes or 0),
        "summary_notes": float(summary_notes or 0),
        "event_notes": float(event_notes or 0),
        "thought_notes": float(thought_notes or 0),
    }


def _reflection_warnings(items: list[TimelineItem], coverage: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    notes = max(coverage.get("notes", 0.0), 1.0)
    if coverage.get("thought_notes", 0.0) / notes < 0.25:
        warnings.append("この月はthought extractionが少ないため、内省の振り返りはまだ薄い可能性があります。")
    if coverage.get("event_notes", 0.0) / notes < 0.25:
        warnings.append("この月はevent extractionが少ないため、Timelineはまだ不完全な可能性があります。")
    if not any(item.evidence and item.evidence[0].get("quote") for item in items):
        warnings.append("evidence quoteが弱いため、元メモ本文で確認してください。")
    return warnings


def _evidence(payload: str | None, note_id: str) -> list[dict[str, str]]:
    try:
        value = json.loads(payload or "[]")
    except json.JSONDecodeError:
        value = []
    if isinstance(value, list) and value:
        return [
            {"note_id": str(item.get("note_id") or note_id), "quote": str(item.get("quote") or "")[:120]}
            for item in value
            if isinstance(item, dict)
        ]
    return [{"note_id": note_id, "quote": ""}]


def _average(values: list[float], *, default: float) -> float:
    return sum(values) / len(values) if values else default


def _normalize_month(month: str | None) -> str | None:
    if not month:
        return None
    value = month.strip()
    if len(value) == 7 and value[4] == "-":
        return value
    return value[:7] if len(value) >= 7 else value


def _short(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"
