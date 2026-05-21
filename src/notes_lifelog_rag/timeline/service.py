from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
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


@dataclass(frozen=True)
class TimelineMonthSummary:
    month: str
    notes_count: int
    summaries_count: int
    events_count: int
    thoughts_count: int
    suggestions_count: int
    has_snapshot: bool
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MonthTimelineItem:
    id: str
    month: str
    date_start: str
    date_end: str
    date_label: str
    item_type: str
    title: str
    summary: str
    detail: str
    themes: list[str]
    categories: list[str]
    emotion: dict[str, Any]
    evidence: list[dict[str, str]]
    source_table: str
    source_id: str
    source_note_id: str
    confidence: float
    importance: float
    date_confidence: float
    sort_key: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MonthTimelineSnapshot:
    id: str
    month: str
    title: str
    overview: str
    thought_summary: str
    event_summary: str
    important_changes: list[str]
    key_themes: list[str]
    dominant_categories: list[str]
    rediscovery_points: list[str]
    revisit_reasons: list[str]
    evidence: list[dict[str, str]]
    quality: dict[str, Any]
    source_counts: dict[str, int]
    source_hash: str
    model_name: str
    generated_by: str
    confidence: float
    importance: float
    created_at: str
    updated_at: str
    items: list[MonthTimelineItem]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimelineBuildLimits:
    max_thoughts: int = 10
    max_events: int = 10
    max_summaries: int = 10
    max_suggestions: int = 5
    max_fallback: int = 3


DEFAULT_TIMELINE_LIMITS = TimelineBuildLimits()
UNKNOWN_MONTHS = {"1900-01", "0000-00", "unknown"}
LOW_PRIORITY_TAG = "low_priority"


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


def list_timeline_months(
    *,
    db_path: str | Path | None = None,
    order: str = "desc",
    include_unknown: bool = False,
) -> list[TimelineMonthSummary]:
    init_db(db_path)
    with connect(db_path) as conn:
        months = sorted(
            {
                month
                for month in _timeline_month_values(conn)
                if month
                and (include_unknown or not _is_unknown_month(month))
            },
            reverse=(order != "asc"),
        )
        output: list[TimelineMonthSummary] = []
        for month in months:
            counts = _month_source_counts(conn, month)
            snapshot = conn.execute(
                """
                SELECT quality_json
                FROM monthly_timeline_snapshots
                WHERE month = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (month,),
            ).fetchone()
            quality = "missing"
            if snapshot:
                quality_data = _json_obj(snapshot["quality_json"])
                warnings = quality_data.get("warnings") or []
                quality = "needs_review" if warnings else "ready"
            output.append(
                TimelineMonthSummary(
                    month=month,
                    notes_count=counts["notes"],
                    summaries_count=counts["summaries"],
                    events_count=counts["events"],
                    thoughts_count=counts["thoughts"],
                    suggestions_count=counts["suggestions"],
                    has_snapshot=snapshot is not None,
                    quality=quality,
                )
            )
    return output


def get_month_sources(month: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    init_db(db_path)
    selected_month = _normalize_month(month)
    if not selected_month:
        return _empty_month_sources(month or "")
    with connect(db_path) as conn:
        all_notes = [dict(row) for row in conn.execute(
            """
            SELECT notes.*, note_summaries.generated_title, note_summaries.one_line_summary,
                   note_summaries.detailed_summary, note_summaries.important_points_json,
                   note_summaries.revisit_reason, note_summaries.confidence AS summary_confidence,
                   note_summaries.importance AS summary_importance,
                   note_summaries.evidence_json AS summary_evidence_json
            FROM notes
            LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
            """,
        ).fetchall()]
        for row in all_notes:
            row["_timeline_date_value"] = _note_timeline_date(row)
        notes = sorted(
            [row for row in all_notes if _month_matches(row.get("_timeline_date_value"), selected_month)],
            key=lambda row: str(row.get("_timeline_date_value") or ""),
        )
        summaries = [row for row in notes if row.get("generated_title") or row.get("one_line_summary")]
        event_rows = [dict(row) for row in conn.execute(
            """
            SELECT events.*, notes.title AS note_title, notes.source_relative_path,
                   notes.created_at AS note_created_at,
                   notes.modified_at AS note_modified_at,
                   notes.note_date AS note_note_date,
                   notes.imported_at AS note_imported_at
            FROM events
            JOIN notes ON notes.id = events.note_id
            """,
        ).fetchall()]
        for row in event_rows:
            row["note_date_value"] = _note_timeline_date(
                {
                    "created_at": row.get("note_created_at"),
                    "modified_at": row.get("note_modified_at"),
                    "note_date": row.get("note_note_date"),
                    "imported_at": row.get("note_imported_at"),
                }
            )
            row["_timeline_date_value"] = _first_date_value(row.get("event_date"), row.get("date_label"), row.get("note_date_value"))
        events = sorted(
            [row for row in event_rows if _month_matches(row.get("_timeline_date_value"), selected_month)],
            key=lambda row: str(row.get("_timeline_date_value") or ""),
        )
        thought_rows = [dict(row) for row in conn.execute(
            """
            SELECT thoughts.*, notes.title AS note_title, notes.source_relative_path,
                   notes.created_at AS note_created_at,
                   notes.modified_at AS note_modified_at,
                   notes.note_date AS note_note_date,
                   notes.imported_at AS note_imported_at
            FROM thoughts
            JOIN notes ON notes.id = thoughts.note_id
            """,
        ).fetchall()]
        for row in thought_rows:
            row["note_date_value"] = _note_timeline_date(
                {
                    "created_at": row.get("note_created_at"),
                    "modified_at": row.get("note_modified_at"),
                    "note_date": row.get("note_note_date"),
                    "imported_at": row.get("note_imported_at"),
                }
            )
            row["_timeline_date_value"] = _first_date_value(row.get("date_label"), row.get("note_date_value"))
        thoughts = sorted(
            [row for row in thought_rows if _month_matches(row.get("_timeline_date_value"), selected_month)],
            key=lambda row: str(row.get("_timeline_date_value") or ""),
        )
        selected_note_ids = {str(row.get("id")) for row in notes}
        categories = [dict(row) for row in conn.execute(
            """
            SELECT notes.id AS note_id, categories.name, note_categories.confidence, note_categories.importance,
                   note_categories.evidence_json
            FROM notes
            JOIN note_categories ON note_categories.note_id = notes.id
            JOIN categories ON categories.id = note_categories.category_id
            """,
        ).fetchall()]
        categories = [row for row in categories if str(row.get("note_id")) in selected_note_ids]
        suggestion_rows = [dict(row) for row in conn.execute(
            """
            SELECT suggestions.*, notes.title AS note_title, notes.source_relative_path,
                   notes.created_at AS note_created_at,
                   notes.modified_at AS note_modified_at,
                   notes.note_date AS note_note_date,
                   notes.imported_at AS note_imported_at
            FROM suggestions
            LEFT JOIN notes ON notes.id = suggestions.note_id
            ORDER BY COALESCE(suggestions.importance, 0.0) DESC, suggestions.created_at DESC
            """
        ).fetchall()]
        for row in suggestion_rows:
            row["note_date_value"] = _note_timeline_date(
                {
                    "created_at": row.get("note_created_at"),
                    "modified_at": row.get("note_modified_at"),
                    "note_date": row.get("note_note_date"),
                    "imported_at": row.get("note_imported_at"),
                }
            )
        note_dates = _note_date_lookup(conn)
        suggestions = [
            row
            for row in (_attribute_suggestion_to_month(row, note_dates) for row in suggestion_rows)
            if row is not None and row.get("_timeline_month") == selected_month
        ]
        reflections = [dict(row) for row in conn.execute(
            "SELECT * FROM monthly_reflections WHERE month = ?",
            (selected_month,),
        ).fetchall()]
    return {
        "month": selected_month,
        "notes": notes,
        "summaries": summaries,
        "events": events,
        "thoughts": thoughts,
        "categories": categories,
        "suggestions": suggestions,
        "monthly_reflections": reflections,
    }


def build_month_timeline_items(
    month: str,
    *,
    db_path: str | Path | None = None,
    limits: TimelineBuildLimits | None = None,
) -> list[MonthTimelineItem]:
    sources = get_month_sources(month, db_path=db_path)
    return _build_month_timeline_items_from_sources(sources, limits=limits)


def generate_month_timeline_snapshot(
    month: str,
    *,
    db_path: str | Path | None = None,
    backend: str = "rule",
    force: bool = False,
    dry_run: bool = False,
    show_sources: bool = False,
    limits: TimelineBuildLimits | None = None,
) -> MonthTimelineSnapshot:
    _ = show_sources
    sources = get_month_sources(month, db_path=db_path)
    selected_month = sources.get("month") or _normalize_month(month) or month
    items = _build_month_timeline_items_from_sources(sources, limits=limits)
    source_counts = _source_counts_from_sources(sources)
    source_hash = _source_hash(sources, items)
    now = datetime.now(tz=timezone.utc).isoformat()
    title = _month_title(selected_month, sources, items)
    quality = _timeline_quality(sources, items)
    snapshot = MonthTimelineSnapshot(
        id=_stable_id("monthly_timeline_snapshot", selected_month, source_hash),
        month=selected_month,
        title=title,
        overview=_month_overview(selected_month, sources, items, title),
        thought_summary=_thought_summary(sources, items),
        event_summary=_event_summary(sources, items),
        important_changes=_important_timeline_changes(items),
        key_themes=_key_themes(sources, items),
        dominant_categories=_dominant_categories(sources),
        rediscovery_points=_timeline_rediscovery_points(items),
        revisit_reasons=_revisit_reasons(sources, items),
        evidence=_timeline_evidence(items),
        quality=quality,
        source_counts=source_counts,
        source_hash=source_hash,
        model_name="timeline-rule-v1",
        generated_by=backend if backend in {"rule", "local", "mock"} else "rule",
        confidence=_timeline_confidence(items, quality),
        importance=_timeline_importance(items),
        created_at=now,
        updated_at=now,
        items=items,
    )
    if not dry_run:
        _store_month_timeline_snapshot(snapshot, db_path=db_path, force=force)
    return snapshot


def generate_timeline_snapshots(
    *,
    months: list[str] | None = None,
    all_months: bool = False,
    limit_months: int | None = None,
    db_path: str | Path | None = None,
    backend: str = "rule",
    force: bool = False,
    dry_run: bool = False,
    limits: TimelineBuildLimits | None = None,
    include_unknown: bool = False,
    progress_callback=None,
) -> list[MonthTimelineSnapshot]:
    values = months or []
    if all_months or not values:
        values = [
            row.month
            for row in list_timeline_months(db_path=db_path, order="desc", include_unknown=include_unknown)
        ]
    if limit_months is not None:
        values = values[: max(0, int(limit_months))]
    snapshots = []
    total = len(values)
    for index, month in enumerate(values, start=1):
        if progress_callback:
            progress_callback(index - 1, total, month)
        try:
            snapshots.append(
                generate_month_timeline_snapshot(
                    month,
                    db_path=db_path,
                    backend=backend,
                    force=force,
                    dry_run=dry_run,
                    limits=limits,
                )
            )
        except Exception:
            # All-month generation should be resume-safe: one malformed month
            # must not prevent other month cards from being built.
            continue
        if progress_callback:
            progress_callback(index, total, month)
    return snapshots


def get_month_timeline_snapshot(
    month: str,
    *,
    db_path: str | Path | None = None,
    generate_if_missing: bool = True,
) -> MonthTimelineSnapshot | None:
    selected_month = _normalize_month(month)
    if not selected_month:
        return None
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM monthly_timeline_snapshots
            WHERE month = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (selected_month,),
        ).fetchone()
        if row:
            items = _stored_month_items(conn, selected_month)
            return _snapshot_from_row(row, items)
    if generate_if_missing:
        return generate_month_timeline_snapshot(selected_month, db_path=db_path, dry_run=True)
    return None


def list_month_timeline_snapshots(
    *,
    year: str | None = None,
    db_path: str | Path | None = None,
    order: str = "desc",
    limit: int | None = None,
    include_unknown: bool = False,
) -> list[MonthTimelineSnapshot]:
    months = [
        item.month
        for item in list_timeline_months(db_path=db_path, order=order, include_unknown=include_unknown)
    ]
    if year:
        months = [month for month in months if month.startswith(str(year))]
    if limit is not None:
        months = months[: max(0, int(limit))]
    output = []
    for month in months:
        snapshot = get_month_timeline_snapshot(month, db_path=db_path, generate_if_missing=True)
        if snapshot:
            output.append(snapshot)
    return output


def format_month_timeline_markdown(snapshot: MonthTimelineSnapshot) -> str:
    theme_lines = [f"- {theme}" for theme in snapshot.key_themes] or ["- まだ十分なテーマがありません。"]
    change_lines = [f"- {item}" for item in snapshot.important_changes] or ["- まだ十分な材料がありません。"]
    revisit_lines = [f"- {item}" for item in snapshot.rediscovery_points + snapshot.revisit_reasons] or ["- まだ十分な材料がありません。"]
    lines = [
        f"## {snapshot.month} {snapshot.title}",
        "",
        "### この月の概要",
        snapshot.overview,
        "",
        "### この月に考えていたこと",
        snapshot.thought_summary or "thoughtsがまだ少ないため、分析後に再生成してください。",
        "",
        "### この月の出来事",
        snapshot.event_summary or "eventsがまだ少ないため、分析後に再生成してください。",
        "",
        "### 重要テーマ",
        *theme_lines,
        "",
        "### 重要な変化",
        *change_lines,
        "",
        "### 見返す価値",
        *revisit_lines,
        "",
        "### 根拠",
    ]
    lines.extend([f"- `{item['note_id'][:12]}` {item['quote']}" for item in snapshot.evidence] or ["- evidence is missing"])
    warnings = snapshot.quality.get("warnings") or []
    if warnings:
        lines.extend(["", "### Quality Warnings", *[f"- {warning}" for warning in warnings]])
    def append_item_section(title: str, values: list[MonthTimelineItem], max_count: int) -> None:
        lines.extend(["", f"### {title}"])
        if not values:
            lines.append("- なし")
            return
        for item in sorted(values, key=lambda row: row.sort_key)[:max_count]:
            lines.append(
                f"- {item.date_label or '日付不明'} [{item.item_type}] "
                f"{_short(item.title, 80)}: {_short(item.summary, 120)} (`{item.source_note_id[:12]}`)"
            )
        if len(values) > max_count:
            lines.append(f"- ... (+{len(values) - max_count} more items; open the GUI Timeline tab for detail)")

    main_items = [item for item in snapshot.items if item.item_type != "suggestion" and not _is_low_priority_item(item)]
    suggestion_items = [item for item in snapshot.items if item.item_type == "suggestion" and not _is_low_priority_item(item)]
    low_priority_items = [item for item in snapshot.items if _is_low_priority_item(item)]
    append_item_section("Main Timeline Items", main_items, 20)
    append_item_section("Supporting Suggestions", suggestion_items, 10)
    append_item_section("Low Priority / Needs Review", low_priority_items, 10)
    return "\n".join(lines)


def format_timeline_report(
    snapshots: list[MonthTimelineSnapshot],
    *,
    title: str = "Timeline Report",
    order: str = "asc",
) -> str:
    values = sorted(snapshots, key=lambda item: item.month, reverse=(order == "desc"))
    lines = [f"# {title}", ""]
    for snapshot in values:
        lines.append(format_month_timeline_markdown(snapshot))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def timeline_qa(
    *,
    month: str | None = None,
    all_months: bool = False,
    db_path: str | Path | None = None,
    include_unknown: bool = False,
) -> list[dict[str, Any]]:
    months = [month] if month else []
    if all_months or not months:
        months = [
            item.month
            for item in list_timeline_months(db_path=db_path, order="desc", include_unknown=include_unknown)
        ]
    rows = []
    for value in months:
        if not value:
            continue
        snapshot = get_month_timeline_snapshot(value, db_path=db_path, generate_if_missing=True)
        if snapshot is None:
            rows.append(
                {
                    "month": value,
                    "quality_score": 0.0,
                    "warnings": ["monthly_timeline_snapshot is missing"],
                    "source_counts": {},
                    "recommended_action": "generate-timelineを実行してください。",
                }
            )
            continue
        warnings = list(snapshot.quality.get("warnings") or [])
        if not snapshot.thought_summary.strip():
            warnings.append("thought_summary is empty")
        if not snapshot.event_summary.strip():
            warnings.append("event_summary is empty")
        if not snapshot.evidence:
            warnings.append("evidence is missing")
        score = max(0.0, min(1.0, float(snapshot.quality.get("quality_score", 0.0))))
        rows.append(
            {
                "month": snapshot.month,
                "quality_score": score,
                "warnings": warnings,
                "source_counts": snapshot.source_counts,
                "recommended_action": _timeline_qa_action(warnings),
            }
        )
    return rows


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


def _empty_month_sources(month: str) -> dict[str, Any]:
    return {
        "month": month,
        "notes": [],
        "summaries": [],
        "events": [],
        "thoughts": [],
        "categories": [],
        "suggestions": [],
        "monthly_reflections": [],
    }


def _source_counts_from_sources(sources: dict[str, Any]) -> dict[str, int]:
    return {
        "notes": len(sources.get("notes") or []),
        "summaries": len(sources.get("summaries") or []),
        "events": len(sources.get("events") or []),
        "thoughts": len(sources.get("thoughts") or []),
        "categories": len(sources.get("categories") or []),
        "suggestions": len(sources.get("suggestions") or []),
        "monthly_reflections": len(sources.get("monthly_reflections") or []),
    }


def _timeline_month_values(conn) -> list[str]:
    months: set[str] = set()
    note_dates = _note_date_lookup(conn)
    months.update(
        month
        for date_value in note_dates.values()
        if (month := _normalize_month_key(date_value))
    )
    event_rows = conn.execute(
        """
        SELECT events.event_date, events.date_label,
               notes.created_at AS note_created_at,
               notes.modified_at AS note_modified_at,
               notes.note_date AS note_note_date,
               notes.imported_at AS note_imported_at
        FROM events
        JOIN notes ON notes.id = events.note_id
        """
    ).fetchall()
    for row in event_rows:
        note_date = _note_timeline_date(
            {
                "created_at": row["note_created_at"],
                "modified_at": row["note_modified_at"],
                "note_date": row["note_note_date"],
                "imported_at": row["note_imported_at"],
            }
        )
        if month := _normalize_month_key(_first_date_value(row["event_date"], row["date_label"], note_date)):
            months.add(month)
    thought_rows = conn.execute(
        """
        SELECT thoughts.date_label,
               notes.created_at AS note_created_at,
               notes.modified_at AS note_modified_at,
               notes.note_date AS note_note_date,
               notes.imported_at AS note_imported_at
        FROM thoughts
        JOIN notes ON notes.id = thoughts.note_id
        """
    ).fetchall()
    for row in thought_rows:
        note_date = _note_timeline_date(
            {
                "created_at": row["note_created_at"],
                "modified_at": row["note_modified_at"],
                "note_date": row["note_note_date"],
                "imported_at": row["note_imported_at"],
            }
        )
        if month := _normalize_month_key(_first_date_value(row["date_label"], note_date)):
            months.add(month)
    suggestion_rows = conn.execute(
        """
        SELECT suggestions.*, notes.title AS note_title, notes.source_relative_path,
               notes.created_at AS note_created_at,
               notes.modified_at AS note_modified_at,
               notes.note_date AS note_note_date,
               notes.imported_at AS note_imported_at
        FROM suggestions
        LEFT JOIN notes ON notes.id = suggestions.note_id
        """
    ).fetchall()
    for row in suggestion_rows:
        row_dict = dict(row)
        row_dict["note_date_value"] = _note_timeline_date(
            {
                "created_at": row_dict.get("note_created_at"),
                "modified_at": row_dict.get("note_modified_at"),
                "note_date": row_dict.get("note_note_date"),
                "imported_at": row_dict.get("note_imported_at"),
            }
        )
        attributed = _attribute_suggestion_to_month(row_dict, note_dates)
        if attributed and attributed.get("_timeline_month"):
            months.add(str(attributed["_timeline_month"]))
    for table in ("monthly_reflections", "monthly_timeline_snapshots"):
        rows = conn.execute(f"SELECT month FROM {table} WHERE month IS NOT NULL AND month != ''").fetchall()
        months.update(
            month
            for row in rows
            if (month := _normalize_month_key(row["month"]))
        )
    return sorted(months)


def _month_source_counts(conn, month: str) -> dict[str, int]:
    notes_rows = conn.execute(
        "SELECT id, created_at, modified_at, note_date, imported_at FROM notes"
    ).fetchall()
    note_dates = {
        str(row["id"]): _note_timeline_date(row)
        for row in notes_rows
    }
    notes = sum(1 for date_value in note_dates.values() if _month_matches(date_value, month))
    summaries = conn.execute(
        "SELECT note_id FROM note_summaries"
    ).fetchall()
    summaries = sum(1 for row in summaries if _month_matches(note_dates.get(str(row["note_id"])), month))
    event_rows = conn.execute(
        """
        SELECT events.event_date, events.date_label, events.note_id
        FROM events
        """
    ).fetchall()
    events = sum(
        1
        for row in event_rows
        if _month_matches(_first_date_value(row["event_date"], row["date_label"], note_dates.get(str(row["note_id"]))), month)
    )
    thought_rows = conn.execute(
        "SELECT thoughts.date_label, thoughts.note_id FROM thoughts"
    ).fetchall()
    thoughts = sum(
        1
        for row in thought_rows
        if _month_matches(_first_date_value(row["date_label"], note_dates.get(str(row["note_id"]))), month)
    )
    suggestions = _month_suggestion_count(conn, month)
    return {
        "notes": int(notes or 0),
        "summaries": int(summaries or 0),
        "events": int(events or 0),
        "thoughts": int(thoughts or 0),
        "suggestions": int(suggestions or 0),
    }


def _build_month_timeline_items_from_sources(
    sources: dict[str, Any],
    *,
    limits: TimelineBuildLimits | None = None,
) -> list[MonthTimelineItem]:
    limits = limits or DEFAULT_TIMELINE_LIMITS
    month = sources.get("month") or ""
    category_by_note = _categories_by_note(sources.get("categories") or [])
    by_type: dict[str, list[MonthTimelineItem]] = {
        "thought": [],
        "event": [],
        "note_summary": [],
        "suggestion": [],
        "fallback": [],
    }
    now = datetime.now(tz=timezone.utc).isoformat()
    for row in sources.get("events") or []:
        evidence = _evidence(row.get("evidence_json"), row.get("note_id", ""))
        date_value = _first_date_value(row.get("event_date"), row.get("date_label"), row.get("note_date_value"))
        categories = category_by_note.get(row.get("note_id"), [])
        title = row.get("title") or "Event"
        summary = row.get("summary") or ""
        categories = _with_quality_tags(
            categories,
            _low_priority_reasons(title, summary, row.get("event_type") or "", evidence, categories, row.get("source_relative_path")),
        )
        by_type["event"].append(
            MonthTimelineItem(
                id=_stable_id("event", month, row.get("id"), row.get("note_id"), row.get("title")),
                month=month,
                date_start=_date_start(date_value),
                date_end="",
                date_label=row.get("date_label") or row.get("event_date") or "日付不明",
                item_type="event",
                title=title,
                summary=summary,
                detail=row.get("event_type") or "",
                themes=[],
                categories=categories,
                emotion={},
                evidence=evidence,
                source_table="events",
                source_id=str(row.get("id") or ""),
                source_note_id=row.get("note_id") or "",
                confidence=_score_confidence(row.get("confidence"), evidence=evidence, date_value=date_value),
                importance=_score_importance(row.get("importance"), evidence=evidence, categories=categories, date_value=date_value),
                date_confidence=_date_confidence_score(row.get("date_confidence")),
                sort_key=_sort_key(date_value, "event", row.get("id")),
                created_at=now,
            )
        )
    for row in sources.get("thoughts") or []:
        evidence = _evidence(row.get("evidence_json"), row.get("note_id", ""))
        date_value = _first_date_value(row.get("date_label"), row.get("note_date_value"))
        themes = _json_list(row.get("themes_json"))
        categories = category_by_note.get(row.get("note_id"), [])
        title = row.get("title") or "Thought"
        summary = row.get("summary") or ""
        categories = _with_quality_tags(
            categories,
            _low_priority_reasons(title, summary, row.get("remember_reason") or "", evidence, categories, row.get("source_relative_path")),
        )
        by_type["thought"].append(
            MonthTimelineItem(
                id=_stable_id("thought", month, row.get("id"), row.get("note_id"), row.get("title")),
                month=month,
                date_start=_date_start(date_value),
                date_end="",
                date_label=row.get("date_label") or row.get("note_date_value") or "日付不明",
                item_type="thought",
                title=title,
                summary=summary,
                detail=row.get("remember_reason") or "",
                themes=[str(item) for item in themes[:6]],
                categories=categories,
                emotion={"label": row.get("emotion_label"), "intensity": row.get("emotion_intensity")},
                evidence=evidence,
                source_table="thoughts",
                source_id=str(row.get("id") or ""),
                source_note_id=row.get("note_id") or "",
                confidence=_score_confidence(row.get("confidence"), evidence=evidence, date_value=date_value),
                importance=_score_importance(row.get("importance"), evidence=evidence, categories=categories, has_reminder=bool(row.get("remember_reason")), date_value=date_value),
                date_confidence=_date_confidence_score("low" if date_value else "unknown"),
                sort_key=_sort_key(date_value, "thought", row.get("id")),
                created_at=now,
            )
        )
    for row in sources.get("summaries") or []:
        evidence = _evidence(row.get("summary_evidence_json"), row.get("id", ""))
        date_value = _first_date_value(row.get("_timeline_date_value"), row.get("created_at"), row.get("modified_at"), row.get("note_date"), row.get("imported_at"))
        categories = category_by_note.get(row.get("id"), [])
        points = " / ".join(str(item) for item in _json_list(row.get("important_points_json"))[:3])
        revisit = row.get("revisit_reason") or ""
        base_importance = row.get("summary_importance")
        if not revisit and _float_safe(base_importance) < 0.6:
            continue
        title = row.get("generated_title") or row.get("title") or "Note"
        summary = row.get("one_line_summary") or points or row.get("title") or ""
        categories = _with_quality_tags(
            categories,
            _low_priority_reasons(title, summary, revisit, evidence, categories, row.get("source_relative_path")),
        )
        by_type["note_summary"].append(
            MonthTimelineItem(
                id=_stable_id("summary", month, row.get("id"), row.get("content_hash")),
                month=month,
                date_start=_date_start(date_value),
                date_end="",
                date_label=(date_value or "日付不明")[:10],
                item_type="note_summary",
                title=title,
                summary=summary,
                detail=revisit,
                themes=[],
                categories=categories,
                emotion={},
                evidence=evidence,
                source_table="note_summaries",
                source_id=row.get("id") or "",
                source_note_id=row.get("id") or "",
                confidence=_score_confidence(row.get("summary_confidence"), evidence=evidence, date_value=date_value),
                importance=_score_importance(base_importance, evidence=evidence, categories=categories, has_reminder=bool(revisit), date_value=date_value),
                date_confidence=_date_confidence_score("medium" if date_value else "unknown"),
                sort_key=_sort_key(date_value, "note_summary", row.get("id")),
                created_at=now,
            )
        )
    for row in sources.get("suggestions") or []:
        note_id = row.get("note_id") or ""
        evidence = _evidence(row.get("evidence_json"), note_id)
        date_value = row.get("_timeline_date_value") or row.get("target_date") or ""
        categories = category_by_note.get(note_id, [])
        title = row.get("title") or "Suggestion"
        summary = row.get("message") or ""
        reasons = _low_priority_reasons(title, summary, row.get("suggestion_type") or "", evidence, categories, row.get("source_relative_path"))
        if row.get("_timeline_attribution_warning"):
            reasons.append(str(row["_timeline_attribution_warning"]))
        categories = _with_quality_tags(categories, reasons)
        by_type["suggestion"].append(
            MonthTimelineItem(
                id=_stable_id("suggestion", month, row.get("id"), note_id, row.get("title")),
                month=month,
                date_start=_date_start(date_value),
                date_end="",
                date_label=(date_value or "日付不明")[:10],
                item_type="suggestion",
                title=title,
                summary=summary,
                detail=row.get("suggestion_type") or "",
                themes=[],
                categories=categories,
                emotion={},
                evidence=evidence,
                source_table="suggestions",
                source_id=str(row.get("id") or ""),
                source_note_id=note_id,
                confidence=_score_confidence(row.get("confidence"), evidence=evidence, date_value=date_value),
                importance=max(
                    0.0,
                    _score_importance(row.get("importance"), evidence=evidence, categories=categories, date_value=date_value)
                    - _suggestion_importance_penalty(row),
                ),
                date_confidence=_date_confidence_score(row.get("_timeline_date_confidence") or ("medium" if date_value else "unknown")),
                sort_key=_sort_key(date_value, "suggestion", row.get("id")),
                created_at=now,
            )
        )
    selected = (
        _select_limited_items(by_type["thought"], limits.max_thoughts)
        + _select_limited_items(by_type["event"], limits.max_events)
        + _select_limited_items(by_type["note_summary"], limits.max_summaries)
        + _select_limited_items(by_type["suggestion"], limits.max_suggestions)
        + _select_limited_items(by_type["fallback"], limits.max_fallback)
    )
    return sorted(selected, key=lambda item: (item.sort_key, _item_type_rank(item.item_type), _is_low_priority_item(item), -item.importance))


def _categories_by_note(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        note_id = row.get("note_id")
        name = row.get("name")
        if note_id and name:
            result.setdefault(note_id, []).append(str(name))
    return result


def _note_date_lookup(conn) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT id, created_at, modified_at, note_date, imported_at
        FROM notes
        """
    ).fetchall()
    return {str(row["id"]): _note_timeline_date(row) for row in rows}


def _note_timeline_date(row: dict[str, Any] | Any) -> str:
    created_at = str(row["created_at"] or "") if "created_at" in row.keys() else str(row.get("created_at") or "")
    modified_at = str(row["modified_at"] or "") if "modified_at" in row.keys() else str(row.get("modified_at") or "")
    note_date = str(row["note_date"] or "") if "note_date" in row.keys() else str(row.get("note_date") or "")
    imported_at = str(row["imported_at"] or "") if "imported_at" in row.keys() else str(row.get("imported_at") or "")
    if created_at and not _created_at_looks_imported(created_at, imported_at, modified_at, note_date):
        return created_at
    return _first_date_value(note_date, modified_at, created_at, imported_at)


def _created_at_looks_imported(created_at: str, imported_at: str, modified_at: str, note_date: str) -> bool:
    created_month = _normalize_month_key(created_at)
    imported_month = _normalize_month_key(imported_at)
    if not created_month or not imported_month or created_month != imported_month:
        return False
    alternate_months = {
        month
        for value in (note_date, modified_at)
        if (month := _normalize_month_key(value)) and not _is_unknown_month(month)
    }
    return bool(alternate_months and created_month not in alternate_months)


def _month_matches(date_value: Any, month: str) -> bool:
    return _normalize_month_key(str(date_value or "")) == _normalize_month_key(month)


def _month_suggestion_count(conn, month: str) -> int:
    note_dates = _note_date_lookup(conn)
    rows = conn.execute(
        """
        SELECT suggestions.*, notes.title AS note_title, notes.source_relative_path,
               COALESCE(notes.created_at, notes.modified_at, notes.note_date, notes.imported_at, '') AS note_date_value
        FROM suggestions
        LEFT JOIN notes ON notes.id = suggestions.note_id
        """
    ).fetchall()
    return sum(
        1
        for row in (_attribute_suggestion_to_month(dict(row), note_dates) for row in rows)
        if row is not None and row.get("_timeline_month") == month
    )


def _attribute_suggestion_to_month(row: dict[str, Any], note_dates: dict[str, str]) -> dict[str, Any] | None:
    target_date = str(row.get("target_date") or "")
    created_at = str(row.get("created_at") or "")
    target_month = _normalize_month_key(target_date)
    created_month = _normalize_month_key(created_at)
    evidence_date = _suggestion_evidence_date(row, note_dates)
    evidence_month = _normalize_month_key(evidence_date)
    target_looks_generated = bool(target_month and created_month and target_month == created_month)

    if _is_unknown_month(target_month or ""):
        target_month = None
    if _is_unknown_month(evidence_month or ""):
        evidence_month = None

    if target_month and not target_looks_generated:
        row["_timeline_month"] = target_month
        row["_timeline_date_value"] = target_date
        row["_timeline_date_confidence"] = "medium"
        return row
    if evidence_month:
        row["_timeline_month"] = evidence_month
        row["_timeline_date_value"] = evidence_date
        row["_timeline_date_confidence"] = "medium"
        if target_looks_generated:
            row["_timeline_attribution_warning"] = "generated_date_used_for_suggestion"
        return row
    return None


def _suggestion_evidence_date(row: dict[str, Any], note_dates: dict[str, str]) -> str:
    note_id = str(row.get("note_id") or "")
    if note_id and note_dates.get(note_id):
        return note_dates[note_id]
    for item in _evidence(row.get("evidence_json"), note_id):
        evidence_note_id = str(item.get("note_id") or "")
        if evidence_note_id and note_dates.get(evidence_note_id):
            return note_dates[evidence_note_id]
    return str(row.get("note_date_value") or "")


def _first_date_value(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if _normalize_month_key(text):
            return text
    return ""


def _select_limited_items(items: list[MonthTimelineItem], limit: int) -> list[MonthTimelineItem]:
    if limit <= 0:
        return []
    ranked = sorted(
        items,
        key=lambda item: (
            _is_low_priority_item(item),
            -item.importance,
            -item.confidence,
            item.sort_key,
        ),
    )
    return ranked[:limit]


def _item_type_rank(item_type: str) -> int:
    return {"thought": 0, "event": 1, "note_summary": 2, "suggestion": 3, "fallback": 4}.get(item_type, 5)


def _suggestion_importance_penalty(row: dict[str, Any]) -> float:
    penalty = 0.12
    suggestion_type = str(row.get("suggestion_type") or "")
    if suggestion_type in {"important_thought", "important_event"} and not row.get("_timeline_attribution_warning"):
        penalty = 0.05
    if row.get("_timeline_attribution_warning"):
        penalty += 0.1
    return penalty


def _with_quality_tags(categories: list[str], reasons: list[str]) -> list[str]:
    values = list(dict.fromkeys(str(item) for item in categories if item))
    if reasons:
        values.append(LOW_PRIORITY_TAG)
        values.extend(f"quality:{reason}" for reason in reasons[:3])
    return list(dict.fromkeys(values))


def _low_priority_reasons(
    title: str,
    summary: str,
    detail: str,
    evidence: list[dict[str, str]],
    categories: list[str],
    source_path: str | None,
) -> list[str]:
    text = f"{title}\n{summary}\n{detail}\n{source_path or ''}".lower()
    reasons: list[str] = []
    lyric_tokens = ["歌詞", "lyrics", "ado", "sixtones", "six tones", "マスカラ", "風のゆくえ"]
    shopping_tokens = ["買い物", "購入", "メニュー", "値段", "金額", "スーパー", "円"]
    if any(token.lower() in text for token in lyric_tokens) or _looks_like_repeated_lyrics(summary):
        reasons.append("lyric_or_music")
    if any(token.lower() in text for token in shopping_tokens):
        reasons.append("shopping_or_menu")
    if "pdf" in text or "｜｜" in text or _looks_noisy(summary):
        reasons.append("noisy_pdf_or_text")
    if re.search(r"https?://|www\.", text) and len(summary.strip()) < 80:
        reasons.append("link_only")
    if len(summary.strip()) < 12:
        reasons.append("very_short_summary")
    if categories and set(categories) <= {"その他"}:
        reasons.append("only_other_category")
    return list(dict.fromkeys(reasons))


def _looks_like_repeated_lyrics(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    short_lines = sum(1 for line in lines if len(line) <= 24)
    repeated = len(lines) - len(set(lines))
    return short_lines / len(lines) >= 0.7 or repeated >= 3


def _looks_noisy(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    odd = sum(1 for char in value if char in "|｜□■�")
    return odd >= 4 or (odd / max(len(value), 1)) > 0.08


def _is_low_priority_item(item: MonthTimelineItem) -> bool:
    if item.importance < 0.25:
        return True
    if item.confidence < 0.35 and item.importance < 0.5:
        return True
    return LOW_PRIORITY_TAG in item.categories or any(str(value).startswith("quality:") for value in item.categories)


def is_low_priority_timeline_item(item: MonthTimelineItem) -> bool:
    return _is_low_priority_item(item)


def _main_material_items(items: list[MonthTimelineItem]) -> list[MonthTimelineItem]:
    return [item for item in items if item.item_type in {"thought", "event", "note_summary"} and not _is_low_priority_item(item)]


def _direct_items(items: list[MonthTimelineItem]) -> list[MonthTimelineItem]:
    return [item for item in items if item.item_type in {"thought", "event", "note_summary"}]


def _source_hash(sources: dict[str, Any], items: list[MonthTimelineItem]) -> str:
    payload = {
        "month": sources.get("month"),
        "counts": _source_counts_from_sources(sources),
        "notes": sorted(str(row.get("id")) + ":" + str(row.get("content_hash")) for row in sources.get("notes") or []),
        "items": sorted(item.id for item in items),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _month_title(month: str, sources: dict[str, Any], items: list[MonthTimelineItem]) -> str:
    categories = _dominant_categories(sources)
    main_items = _main_material_items(items)
    if len(categories) >= 2:
        return f"{categories[0]}と{categories[1]}を見返す月"
    if categories:
        return f"{categories[0]}を中心にした月"
    top = next(
        (
            item.title
            for item in sorted(main_items or items, key=lambda item: item.importance, reverse=True)
            if item.title
        ),
        "",
    )
    return f"{top}を見返す月" if top else f"{month}の記憶カード"


def _month_overview(month: str, sources: dict[str, Any], items: list[MonthTimelineItem], title: str) -> str:
    counts = _source_counts_from_sources(sources)
    categories = "、".join(_dominant_categories(sources)[:4])
    main_items = _main_material_items(items)
    top_items = sorted(main_items, key=lambda item: item.importance, reverse=True)[:4]
    top_text = "、".join(item.title for item in top_items if item.title)
    if counts["notes"] == 0:
        return f"{month} はまだメモが少ないため、Timelineの材料が不足しています。"
    if top_text:
        provisional = ""
        if counts["thoughts"] < 2 or counts["events"] < 2:
            provisional = "ただし、この月の直接抽出されたthought/eventはまだ少ないため、要約は暫定です。"
        return (
            f"{month}は、{categories or title}に関する記録が目立つ月です。"
            f"メモ上では、{top_text} などが重要な手がかりとして残っています。"
            f"{provisional}"
            "根拠が弱い項目は元メモを開いて確認してください。"
        )
    return f"{month}は、{categories or '複数テーマ'}に関するメモが残っている月です。分析を進めると振り返りが豊かになります。"


def _thought_summary(sources: dict[str, Any], items: list[MonthTimelineItem]) -> str:
    thoughts = [item for item in items if item.item_type == "thought" and not _is_low_priority_item(item)]
    if not thoughts:
        summaries = [item for item in items if item.item_type == "note_summary" and not _is_low_priority_item(item)]
        if summaries:
            snippets = " / ".join(_short(item.summary or item.title, 70) for item in summaries[:4])
            return (
                "この月のthought抽出はまだ少ないため、何を考えていたかはnote summaryからの控えめな推定になります。"
                f"主な手がかりは、{snippets} です。"
            )
        return "この月のthought抽出はまだ少ないため、何を考えていたかはまだ十分に判断できません。"
    top = sorted(thoughts, key=lambda item: item.importance + item.confidence, reverse=True)[:4]
    themes = "、".join(_key_themes(sources, top)[:4])
    snippets = " / ".join(_short(item.summary or item.title, 70) for item in top)
    return f"この月は{themes or '複数のテーマ'}について考えていた可能性があります。主な手がかりは、{snippets} です。"


def _event_summary(sources: dict[str, Any], items: list[MonthTimelineItem]) -> str:
    events = [item for item in items if item.item_type == "event" and not _is_low_priority_item(item)]
    if not events:
        summaries = [item for item in items if item.item_type == "note_summary" and not _is_low_priority_item(item)]
        if summaries:
            snippets = "、".join(item.title for item in summaries[:4] if item.title)
            return (
                "この月のevent抽出はまだ少ないため、出来事の把握はnote summaryに寄っています。"
                f"関連する記録として、{snippets} などがあります。"
            )
        return "この月のevent抽出はまだ少ないため、出来事の把握はまだ不十分です。analyze-all後に再生成すると改善します。"
    top = sorted(events, key=lambda item: item.importance + item.confidence, reverse=True)[:5]
    snippets = "、".join(item.title for item in top if item.title)
    return f"この月には、{snippets} などの出来事や進展が記録されています。"


def _important_timeline_changes(items: list[MonthTimelineItem]) -> list[str]:
    changes = []
    material = _main_material_items(items) or [item for item in items if not _is_low_priority_item(item)]
    for item in sorted(material, key=lambda row: row.importance, reverse=True):
        if item.importance < 0.55:
            continue
        phrase = "考えの変化" if item.item_type == "thought" else "出来事・進展"
        changes.append(f"{item.title}: {phrase}として後で見返す価値があります。")
        if len(changes) >= 5:
            break
    return changes


def _key_themes(sources: dict[str, Any], items: list[MonthTimelineItem]) -> list[str]:
    counter: Counter[str] = Counter()
    for item in items:
        if _is_low_priority_item(item):
            continue
        counter.update(theme for theme in item.themes if not str(theme).startswith("quality:"))
        counter.update(category for category in item.categories if category != LOW_PRIORITY_TAG and not str(category).startswith("quality:"))
    for category in sources.get("categories") or []:
        if category.get("name"):
            counter[str(category["name"])] += 1
    return [name for name, _ in counter.most_common(8)]


def _dominant_categories(sources: dict[str, Any]) -> list[str]:
    counter: Counter[str] = Counter()
    for row in sources.get("categories") or []:
        name = row.get("name")
        if name:
            counter[str(name)] += 1
    return [name for name, _ in counter.most_common(6)]


def _timeline_rediscovery_points(items: list[MonthTimelineItem]) -> list[str]:
    points = []
    material = _main_material_items(items) or [item for item in items if not _is_low_priority_item(item)]
    for item in sorted(material, key=lambda row: row.importance + row.confidence, reverse=True)[:5]:
        points.append(f"{item.title}: {_short(item.summary or item.detail, 110)}")
    return points


def _revisit_reasons(sources: dict[str, Any], items: list[MonthTimelineItem]) -> list[str]:
    values = []
    for row in sources.get("summaries") or []:
        title = str(row.get("generated_title") or row.get("title") or "")
        summary = str(row.get("one_line_summary") or row.get("detailed_summary") or "")
        evidence = _evidence(row.get("summary_evidence_json"), row.get("id", ""))
        if _low_priority_reasons(title, summary, str(row.get("revisit_reason") or ""), evidence, [], row.get("source_relative_path")):
            continue
        reason = str(row.get("revisit_reason") or "").strip()
        if reason:
            values.append(_short(reason, 140))
    for item in _main_material_items(items) or [item for item in items if not _is_low_priority_item(item)]:
        if item.detail and item.item_type in {"thought", "suggestion"}:
            values.append(_short(item.detail, 140))
    deduped = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped[:6]


def _timeline_evidence(items: list[MonthTimelineItem]) -> list[dict[str, str]]:
    evidence = []
    material = _main_material_items(items) or [item for item in items if not _is_low_priority_item(item)] or items
    for item in sorted(material, key=lambda row: row.importance + row.confidence, reverse=True):
        for row in item.evidence:
            quote = str(row.get("quote") or "").strip()
            note_id = str(row.get("note_id") or item.source_note_id)
            if note_id and quote:
                evidence.append({"note_id": note_id, "quote": _short(quote, 160)})
                break
        if len(evidence) >= 8:
            break
    return evidence


def _timeline_quality(sources: dict[str, Any], items: list[MonthTimelineItem]) -> dict[str, Any]:
    counts = _source_counts_from_sources(sources)
    evidence_items = [item for item in items if any((row.get("quote") or "").strip() for row in item.evidence)]
    warnings = []
    if not evidence_items:
        warnings.append("evidence_missing")
    if any(_is_title_only_evidence(row, item.title) for item in items for row in item.evidence):
        warnings.append("title_only_evidence")
    if any(item.confidence <= 0.45 for item in items):
        warnings.append("low_confidence")
    unknown_dates = sum(1 for item in items if not item.date_start)
    if items and unknown_dates / len(items) > 0.4:
        warnings.extend(["unknown_date", "date_attribution_uncertain"])
    if any(item.date_confidence <= 0.35 for item in items):
        warnings.append("date_attribution_uncertain")
    direct_items = _direct_items(items)
    main_items = _main_material_items(items)
    suggestion_items = [item for item in items if item.item_type == "suggestion"]
    active_suggestion_items = [item for item in suggestion_items if not _is_low_priority_item(item)]
    low_priority_items = [item for item in items if _is_low_priority_item(item)]
    fallback_heavy = counts["thoughts"] < 2 and counts["events"] < 2 and counts["summaries"] > 0
    if fallback_heavy:
        warnings.append("fallback_heavy")
    if counts["thoughts"] < 2:
        warnings.extend(["too_few_thoughts", "low_direct_thoughts"])
    if counts["events"] < 2:
        warnings.extend(["too_few_events", "low_direct_events"])
    suggestions_dominated = bool(active_suggestion_items and len(active_suggestion_items) >= max(2, len(main_items)))
    if suggestions_dominated:
        warnings.append("suggestions_dominated")
    if any(any(str(value).startswith("quality:") for value in item.categories) for item in low_priority_items):
        warnings.append("noisy_items_present")
    if low_priority_items:
        warnings.append("low_value_items_present")
    if any("quality:generated_date_used_for_suggestion" in item.categories for item in suggestion_items):
        warnings.append("generated_date_used_for_suggestion")
    if _is_unknown_month(str(sources.get("month") or "")):
        warnings.append("invalid_month_1900")
    source_count = max(len(items), 1)
    quality_score = 1.0
    quality_score -= 0.1 * len(set(warnings))
    if suggestions_dominated:
        quality_score -= 0.2
    if len(main_items) >= 3:
        quality_score += 0.18
    elif direct_items:
        quality_score += 0.08
    quality_score += min(0.2, len(evidence_items) / source_count * 0.2)
    quality_score = max(0.0, min(1.0, quality_score))
    return {
        "source_counts": counts,
        "item_counts": dict(Counter(item.item_type for item in items)),
        "main_item_count": len(main_items),
        "low_priority_item_count": len(low_priority_items),
        "evidence_coverage": len(evidence_items) / source_count,
        "thought_coverage": counts["thoughts"] / max(counts["notes"], 1),
        "event_coverage": counts["events"] / max(counts["notes"], 1),
        "is_fallback_heavy": fallback_heavy,
        "is_suggestions_dominated": suggestions_dominated,
        "warnings": sorted(set(warnings)),
        "quality_score": quality_score,
    }


def _timeline_confidence(items: list[MonthTimelineItem], quality: dict[str, Any]) -> float:
    base = _average([item.confidence for item in items], default=0.25)
    return max(0.0, min(1.0, base * 0.75 + float(quality.get("quality_score") or 0.0) * 0.25))


def _timeline_importance(items: list[MonthTimelineItem]) -> float:
    if not items:
        return 0.2
    material = _main_material_items(items) or [item for item in items if not _is_low_priority_item(item)] or items
    top = sorted((item.importance for item in material), reverse=True)[:5]
    return max(0.0, min(1.0, sum(top) / len(top)))


def _store_month_timeline_snapshot(snapshot: MonthTimelineSnapshot, *, db_path: str | Path | None, force: bool) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        if force:
            conn.execute("DELETE FROM monthly_timeline_items WHERE month = ?", (snapshot.month,))
            conn.execute("DELETE FROM monthly_timeline_snapshots WHERE month = ?", (snapshot.month,))
        conn.execute(
            """
            INSERT INTO monthly_timeline_snapshots(
                id, month, title, overview, thought_summary, event_summary,
                important_changes_json, key_themes_json, dominant_categories_json,
                rediscovery_points_json, revisit_reasons_json, evidence_json,
                quality_json, source_counts_json, source_hash, model_name,
                generated_by, confidence, importance, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(month, source_hash) DO UPDATE SET
                title = excluded.title,
                overview = excluded.overview,
                thought_summary = excluded.thought_summary,
                event_summary = excluded.event_summary,
                important_changes_json = excluded.important_changes_json,
                key_themes_json = excluded.key_themes_json,
                dominant_categories_json = excluded.dominant_categories_json,
                rediscovery_points_json = excluded.rediscovery_points_json,
                revisit_reasons_json = excluded.revisit_reasons_json,
                evidence_json = excluded.evidence_json,
                quality_json = excluded.quality_json,
                source_counts_json = excluded.source_counts_json,
                model_name = excluded.model_name,
                generated_by = excluded.generated_by,
                confidence = excluded.confidence,
                importance = excluded.importance,
                updated_at = excluded.updated_at
            """,
            (
                snapshot.id,
                snapshot.month,
                snapshot.title,
                snapshot.overview,
                snapshot.thought_summary,
                snapshot.event_summary,
                json.dumps(snapshot.important_changes, ensure_ascii=False),
                json.dumps(snapshot.key_themes, ensure_ascii=False),
                json.dumps(snapshot.dominant_categories, ensure_ascii=False),
                json.dumps(snapshot.rediscovery_points, ensure_ascii=False),
                json.dumps(snapshot.revisit_reasons, ensure_ascii=False),
                json.dumps(snapshot.evidence, ensure_ascii=False),
                json.dumps(snapshot.quality, ensure_ascii=False),
                json.dumps(snapshot.source_counts, ensure_ascii=False),
                snapshot.source_hash,
                snapshot.model_name,
                snapshot.generated_by,
                snapshot.confidence,
                snapshot.importance,
                snapshot.created_at,
                snapshot.updated_at,
            ),
        )
        conn.execute("DELETE FROM monthly_timeline_items WHERE month = ?", (snapshot.month,))
        for item in snapshot.items:
            conn.execute(
                """
                INSERT INTO monthly_timeline_items(
                    id, month, date_start, date_end, date_label, item_type,
                    title, summary, detail, themes_json, categories_json,
                    emotion_json, evidence_json, source_table, source_id,
                    source_note_id, confidence, importance, date_confidence,
                    sort_key, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.month,
                    item.date_start,
                    item.date_end,
                    item.date_label,
                    item.item_type,
                    item.title,
                    item.summary,
                    item.detail,
                    json.dumps(item.themes, ensure_ascii=False),
                    json.dumps(item.categories, ensure_ascii=False),
                    json.dumps(item.emotion, ensure_ascii=False),
                    json.dumps(item.evidence, ensure_ascii=False),
                    item.source_table,
                    item.source_id,
                    item.source_note_id,
                    item.confidence,
                    item.importance,
                    item.date_confidence,
                    item.sort_key,
                    item.created_at,
                ),
            )


def _stored_month_items(conn, month: str) -> list[MonthTimelineItem]:
    rows = conn.execute(
        "SELECT * FROM monthly_timeline_items WHERE month = ? ORDER BY sort_key ASC, importance DESC",
        (month,),
    ).fetchall()
    return [_month_item_from_row(row) for row in rows]


def _snapshot_from_row(row, items: list[MonthTimelineItem]) -> MonthTimelineSnapshot:
    return MonthTimelineSnapshot(
        id=row["id"],
        month=row["month"],
        title=row["title"] or "",
        overview=row["overview"] or "",
        thought_summary=row["thought_summary"] or "",
        event_summary=row["event_summary"] or "",
        important_changes=_json_list(row["important_changes_json"]),
        key_themes=_json_list(row["key_themes_json"]),
        dominant_categories=_json_list(row["dominant_categories_json"]),
        rediscovery_points=_json_list(row["rediscovery_points_json"]),
        revisit_reasons=_json_list(row["revisit_reasons_json"]),
        evidence=_evidence(row["evidence_json"], ""),
        quality=_json_obj(row["quality_json"]),
        source_counts={key: int(value) for key, value in _json_obj(row["source_counts_json"]).items()},
        source_hash=row["source_hash"] or "",
        model_name=row["model_name"] or "",
        generated_by=row["generated_by"] or "",
        confidence=float(row["confidence"] or 0.0),
        importance=float(row["importance"] or 0.0),
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        items=items,
    )


def _month_item_from_row(row) -> MonthTimelineItem:
    return MonthTimelineItem(
        id=row["id"],
        month=row["month"],
        date_start=row["date_start"] or "",
        date_end=row["date_end"] or "",
        date_label=row["date_label"] or "",
        item_type=row["item_type"],
        title=row["title"] or "",
        summary=row["summary"] or "",
        detail=row["detail"] or "",
        themes=_json_list(row["themes_json"]),
        categories=_json_list(row["categories_json"]),
        emotion=_json_obj(row["emotion_json"]),
        evidence=_evidence(row["evidence_json"], row["source_note_id"] or ""),
        source_table=row["source_table"] or "",
        source_id=row["source_id"] or "",
        source_note_id=row["source_note_id"] or "",
        confidence=float(row["confidence"] or 0.0),
        importance=float(row["importance"] or 0.0),
        date_confidence=float(row["date_confidence"] or 0.0),
        sort_key=row["sort_key"] or "",
        created_at=row["created_at"] or "",
    )


def _timeline_qa_action(warnings: list[str]) -> str:
    if not warnings:
        return "ready"
    if (
        "fallback_heavy" in warnings
        or "too_few_thoughts" in warnings
        or "too_few_events" in warnings
        or "low_direct_thoughts" in warnings
        or "low_direct_events" in warnings
    ):
        return "analyze-all後にgenerate-timeline --forceを実行してください。"
    if "suggestions_dominated" in warnings:
        return "suggestions以外のthought/event/note summaryを増やしてから再生成してください。"
    if "evidence_missing" in warnings or "title_only_evidence" in warnings:
        return "元メモを確認し、evidenceの弱い抽出をレビューしてください。"
    return "Timeline detailで元メモを確認してください。"


def _json_list(payload: str | None) -> list[Any]:
    try:
        value = json.loads(payload or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _json_obj(payload: str | None) -> dict[str, Any]:
    try:
        value = json.loads(payload or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stable_id(*parts: Any) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _date_start(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    return text[:10] if len(text) >= 10 and text[4:5] == "-" else ""


def _sort_key(date_value: str | None, item_type: str, source_id: Any) -> str:
    date = _date_start(date_value) or "9999-99-99"
    return f"{date}:{item_type}:{source_id or ''}"


def _date_confidence_score(value: Any) -> float:
    text = str(value or "").lower()
    if text in {"high", "1", "1.0"}:
        return 0.9
    if text in {"medium", "mid", "0.5"}:
        return 0.6
    if text in {"low", "0.25"}:
        return 0.35
    return 0.2


def _float_safe(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_importance(
    value: Any,
    *,
    evidence: list[dict[str, str]],
    categories: list[str],
    has_reminder: bool = False,
    date_value: str | None = None,
) -> float:
    score = _float_safe(value, 0.35)
    if any((item.get("quote") or "").strip() for item in evidence):
        score += 0.05
    if has_reminder:
        score += 0.08
    if date_value and _date_start(date_value):
        score += 0.03
    important_categories = {"研究", "就職活動", "就活", "感情・内省", "人間関係", "アプリ開発", "AIエージェント"}
    if any(category in important_categories for category in categories):
        score += 0.06
    return max(0.0, min(1.0, score))


def _score_confidence(value: Any, *, evidence: list[dict[str, str]], date_value: str | None) -> float:
    score = _float_safe(value, 0.35)
    if any((item.get("quote") or "").strip() for item in evidence):
        score += 0.08
    else:
        score -= 0.08
    if date_value and _date_start(date_value):
        score += 0.04
    return max(0.05, min(1.0, score))


def _is_title_only_evidence(evidence: dict[str, str], title: str) -> bool:
    quote = " ".join(str(evidence.get("quote") or "").lower().split())
    normalized_title = " ".join(str(title or "").lower().split())
    return bool(quote and normalized_title and (quote == normalized_title or quote in normalized_title or normalized_title in quote))


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
    normalized = _normalize_month_key(value)
    if normalized:
        return normalized
    return value[:7] if len(value) >= 7 else value


def _normalize_month_key(value: str | None) -> str | None:
    text = str(value or "").strip()
    match = re.match(r"^(\d{4})[-/\.](\d{1,2})", text)
    if match:
        month_value = int(match.group(2))
        return f"{match.group(1)}-{month_value:02d}" if 1 <= month_value <= 12 else None
    match = re.match(r"^(\d{4})年\s*(\d{1,2})月", text)
    if match:
        month_value = int(match.group(2))
        return f"{match.group(1)}-{month_value:02d}" if 1 <= month_value <= 12 else None
    return None


def _is_unknown_month(month: str) -> bool:
    value = str(month or "").strip().lower()
    return value in UNKNOWN_MONTHS or value.startswith("1900-")


def _month_patterns(month: str) -> list[str]:
    normalized = _normalize_month_key(month) or month
    if not re.match(r"^\d{4}-\d{2}$", normalized):
        return [f"{month}%"]
    year, month_value = normalized.split("-", 1)
    month_int = int(month_value)
    return [
        f"{year}-{month_value}%",
        f"{year}/{month_value}%",
        f"{year}.{month_value}%",
        f"{year}年{month_int}月%",
        f"{year}年{month_value}月%",
    ]


def _month_filter_sql(expression: str, month: str) -> tuple[str, tuple[str, ...]]:
    patterns = _month_patterns(month)
    return "(" + " OR ".join(f"{expression} LIKE ?" for _ in patterns) + ")", tuple(patterns)


def _short(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"
