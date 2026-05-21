from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

from notes_lifelog_rag.analysis.service import PROMPT_VERSION, analyze_all
from notes_lifelog_rag.config import database_path, raw_notes_path
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.embeddings.engines import get_embedding_backend
from notes_lifelog_rag.embeddings.repository import build_chunk_embeddings
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.models.status import model_statuses
from notes_lifelog_rag.runtime.cuda import collect_cuda_status
from notes_lifelog_rag.runtime.device import effective_dtype, resolve_device
from notes_lifelog_rag.search.hybrid import hybrid_search_notes
from notes_lifelog_rag.timeline.service import (
    MonthTimelineSnapshot,
    ReflectionReport,
    TimelineItem,
    build_monthly_reflection,
    build_timeline,
    format_reflection_markdown,
    format_timeline_markdown,
    generate_month_timeline_snapshot,
    list_month_timeline_snapshots,
    list_timeline_months,
    timeline_qa,
)

DEFAULT_NOTE_LIMIT = 80


def get_db_stats(db_path: str | Path | None = None) -> dict[str, int]:
    init_db(db_path)
    names = [
        "notes",
        "note_chunks",
        "chunk_embeddings",
        "categories",
        "note_categories",
        "note_summaries",
        "events",
        "thoughts",
        "suggestions",
        "monthly_reflections",
        "monthly_timeline_snapshots",
        "monthly_timeline_items",
        "model_runs",
        "import_errors",
    ]
    with connect(db_path) as conn:
        return {name: table_count(conn, name) for name in names}


def db_stats_rows(db_path: str | Path | None = None) -> list[list[Any]]:
    stats = get_db_stats(db_path)
    return [[name, count] for name, count in stats.items()]


def import_error_rows(limit: int = 50, db_path: str | Path | None = None) -> list[list[Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT source_path, parser_name, error_message, created_at FROM import_errors ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [[row["created_at"], row["parser_name"], row["source_path"], row["error_message"]] for row in rows]


def get_sidebar_state(db_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "stats": get_db_stats(db_path),
        "categories": list_categories_with_counts(db_path),
        "months": list_months_with_counts(db_path),
        "health": get_analysis_health(db_path),
        "running_jobs": get_running_jobs(),
    }


def list_categories_with_counts(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                categories.name AS name,
                COUNT(DISTINCT note_categories.note_id) AS note_count
            FROM categories
            LEFT JOIN note_categories ON note_categories.category_id = categories.id
            GROUP BY categories.id, categories.name
            ORDER BY note_count DESC, categories.name ASC
            """
        ).fetchall()
    return [{"name": row["name"], "note_count": int(row["note_count"] or 0)} for row in rows]


def list_months_with_counts(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT SUBSTR(COALESCE(note_date, modified_at, created_at, imported_at), 1, 7) AS month,
                   COUNT(*) AS note_count
            FROM notes
            WHERE COALESCE(note_date, modified_at, created_at, imported_at) IS NOT NULL
              AND LENGTH(COALESCE(note_date, modified_at, created_at, imported_at)) >= 7
            GROUP BY month
            ORDER BY month DESC
            LIMIT 48
            """
        ).fetchall()
    return [
        {"month": row["month"], "note_count": int(row["note_count"] or 0)}
        for row in rows
        if row["month"]
    ]


def list_notes(
    *,
    category: str | None = None,
    month: str | None = None,
    query: str | None = None,
    filter_name: str | None = None,
    sort: str = "updated_desc",
    limit: int = DEFAULT_NOTE_LIMIT,
    has_summary: bool = False,
    has_events: bool = False,
    has_thoughts: bool = False,
    low_confidence: bool = False,
    evidence_missing: bool = False,
    favorite: bool = False,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
    filter_key = (filter_name or "").strip().lower().replace(" ", "_")
    if filter_key in {"important", "today_rediscovery"}:
        favorite = True
    if filter_key in {"low_confidence", "needs_review"}:
        low_confidence = True
    if filter_key in {"evidence_missing", "evidence_review"}:
        evidence_missing = True
    where: list[str] = []
    params: list[Any] = []
    if category:
        where.append(
            """
            EXISTS (
                SELECT 1 FROM note_categories nc
                JOIN categories c ON c.id = nc.category_id
                WHERE nc.note_id = notes.id AND c.name = ?
            )
            """
        )
        params.append(category)
    if month:
        where.append("COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at, '') LIKE ?")
        params.append(f"{month}%")
    if query and query.strip():
        like = f"%{query.strip()}%"
        where.append(
            """
            (
                notes.title LIKE ?
                OR notes.source_relative_path LIKE ?
                OR notes.body LIKE ?
                OR note_summaries.generated_title LIKE ?
                OR note_summaries.one_line_summary LIKE ?
            )
            """
        )
        params.extend([like, like, like, like, like])
    if has_summary:
        where.append("note_summaries.note_id IS NOT NULL")
    if has_events:
        where.append("EXISTS (SELECT 1 FROM events WHERE events.note_id = notes.id)")
    if has_thoughts:
        where.append("EXISTS (SELECT 1 FROM thoughts WHERE thoughts.note_id = notes.id)")
    if low_confidence:
        where.append(
            """
            (
                COALESCE(note_summaries.confidence, 1.0) <= 0.45
                OR EXISTS (SELECT 1 FROM events WHERE events.note_id = notes.id AND COALESCE(events.confidence, 1.0) <= 0.45)
                OR EXISTS (SELECT 1 FROM thoughts WHERE thoughts.note_id = notes.id AND COALESCE(thoughts.confidence, 1.0) <= 0.45)
            )
            """
        )
    if evidence_missing:
        where.append(
            """
            (
                note_summaries.note_id IS NULL
                OR note_summaries.evidence_json IS NULL
                OR note_summaries.evidence_json = ''
                OR note_summaries.evidence_json = '[]'
                OR EXISTS (SELECT 1 FROM events WHERE events.note_id = notes.id AND (events.evidence_json IS NULL OR events.evidence_json = '' OR events.evidence_json = '[]'))
                OR EXISTS (SELECT 1 FROM thoughts WHERE thoughts.note_id = notes.id AND (thoughts.evidence_json IS NULL OR thoughts.evidence_json = '' OR thoughts.evidence_json = '[]'))
            )
            """
        )
    if favorite:
        where.append(
            """
            (
                COALESCE(note_summaries.importance, 0.0) >= 0.75
                OR EXISTS (SELECT 1 FROM events WHERE events.note_id = notes.id AND COALESCE(events.importance, 0.0) >= 0.75)
                OR EXISTS (SELECT 1 FROM thoughts WHERE thoughts.note_id = notes.id AND COALESCE(thoughts.importance, 0.0) >= 0.75)
            )
            """
        )
    order_by = _sort_clause(sort)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT
            notes.id,
            notes.title,
            notes.source_relative_path,
            COALESCE(notes.folder, '') AS folder,
            COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at, '') AS date_value,
            notes.content_hash,
            note_summaries.generated_title,
            note_summaries.one_line_summary,
            note_summaries.confidence AS summary_confidence,
            note_summaries.importance AS summary_importance,
            note_summaries.evidence_json AS summary_evidence_json,
            GROUP_CONCAT(DISTINCT categories.name) AS category_names,
            COUNT(DISTINCT events.id) AS event_count,
            COUNT(DISTINCT thoughts.id) AS thought_count
        FROM notes
        LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
        LEFT JOIN note_categories ON note_categories.note_id = notes.id
        LEFT JOIN categories ON categories.id = note_categories.category_id
        LEFT JOIN events ON events.note_id = notes.id
        LEFT JOIN thoughts ON thoughts.note_id = notes.id
        {where_sql}
        GROUP BY notes.id
        ORDER BY {order_by}
        LIMIT ?
    """
    params.append(max(1, int(limit)))
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_note_list_item(row) for row in rows]


def get_analysis_health(db_path: str | Path | None = None) -> dict[str, Any]:
    stats = get_db_stats(db_path)
    notes = int(stats.get("notes", 0))
    chunks = int(stats.get("note_chunks", 0))
    with connect(db_path) as conn:
        unique_category_notes = int(
            conn.execute("SELECT COUNT(DISTINCT note_id) AS count FROM note_categories").fetchone()["count"] or 0
        )
        unique_event_notes = int(
            conn.execute("SELECT COUNT(DISTINCT note_id) AS count FROM events").fetchone()["count"] or 0
        )
        unique_thought_notes = int(
            conn.execute("SELECT COUNT(DISTINCT note_id) AS count FROM thoughts").fetchone()["count"] or 0
        )
        embedding_chunks = int(
            conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) AS count FROM chunk_embeddings WHERE status = 'success'"
            ).fetchone()["count"]
            or 0
        )
        embedding_notes = int(
            conn.execute(
                "SELECT COUNT(DISTINCT note_id) AS count FROM chunk_embeddings WHERE status = 'success'"
            ).fetchone()["count"]
            or 0
        )
        failures = int(conn.execute("SELECT COUNT(*) AS count FROM model_runs WHERE success = 0").fetchone()["count"] or 0)
        low_confidence_items = int(
            conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM note_summaries WHERE COALESCE(confidence, 1.0) <= 0.45)
                  + (SELECT COUNT(*) FROM events WHERE COALESCE(confidence, 1.0) <= 0.45)
                  + (SELECT COUNT(*) FROM thoughts WHERE COALESCE(confidence, 1.0) <= 0.45) AS count
                """
            ).fetchone()["count"]
            or 0
        )
        evidence_warnings = len(get_quality_warnings(limit=500, db_path=db_path, warning_types={"evidence_missing", "evidence_title_only"}))
    summary_coverage = _ratio(stats.get("note_summaries", 0), notes)
    category_coverage = _ratio(unique_category_notes, notes)
    event_coverage = _ratio(unique_event_notes, notes)
    thought_coverage = _ratio(unique_thought_notes, notes)
    embedding_coverage = _ratio(embedding_chunks, chunks)
    warnings: list[str] = []
    if stats.get("suggestions", 0) == 0:
        warnings.append("Suggestions are not generated yet. Run generate-suggestions.")
    if thought_coverage < 0.5:
        warnings.append("Thought extraction is still incomplete. Run analyze-all or extract-thoughts.")
    if evidence_warnings:
        warnings.append(f"{evidence_warnings} items need evidence review.")
    status = "Good"
    if summary_coverage < 0.6 or thought_coverage < 0.3:
        status = "Sparse"
    if failures or low_confidence_items or evidence_warnings:
        status = "Needs Review"
    if get_running_jobs().get("analyze_all"):
        status = "Running Job"
    return {
        **stats,
        "notes": notes,
        "chunks": chunks,
        "summaries": stats.get("note_summaries", 0),
        "events": stats.get("events", 0),
        "thoughts": stats.get("thoughts", 0),
        "embeddings": stats.get("chunk_embeddings", 0),
        "embedding_notes": embedding_notes,
        "embedding_chunks": embedding_chunks,
        "unique_category_notes": unique_category_notes,
        "unique_event_notes": unique_event_notes,
        "unique_thought_notes": unique_thought_notes,
        "summary_coverage": summary_coverage,
        "category_coverage": category_coverage,
        "event_coverage": event_coverage,
        "thought_coverage": thought_coverage,
        "embedding_coverage": embedding_coverage,
        "summary_missing": max(notes - int(stats.get("note_summaries", 0)), 0),
        "event_missing": max(notes - unique_event_notes, 0),
        "thought_missing": max(notes - unique_thought_notes, 0),
        "embedding_missing": max(chunks - embedding_chunks, 0),
        "model_runs_failures": failures,
        "low_confidence_items": low_confidence_items,
        "evidence_warnings": evidence_warnings,
        "health_status": status,
        "warnings": warnings,
    }


def search_notes(
    query: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    device_info = resolve_device(device)
    results = hybrid_search_notes(
        query,
        limit=int(limit),
        db_path=db_path,
        embedding_backend=backend,
        reranker_backend="none",
        device_info=device_info,
    )
    with connect(db_path) as conn:
        output: list[dict[str, Any]] = []
        for result in results:
            row = conn.execute(
                """
                SELECT
                    notes.id,
                    notes.title,
                    notes.source_relative_path,
                    COALESCE(notes.folder, '') AS folder,
                    COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at, '') AS date_value,
                    notes.content_hash,
                    note_summaries.generated_title,
                    note_summaries.one_line_summary,
                    note_summaries.confidence AS summary_confidence,
                    note_summaries.importance AS summary_importance,
                    note_summaries.evidence_json AS summary_evidence_json,
                    GROUP_CONCAT(DISTINCT categories.name) AS category_names,
                    COUNT(DISTINCT events.id) AS event_count,
                    COUNT(DISTINCT thoughts.id) AS thought_count
                FROM notes
                LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
                LEFT JOIN note_categories ON note_categories.note_id = notes.id
                LEFT JOIN categories ON categories.id = note_categories.category_id
                LEFT JOIN events ON events.note_id = notes.id
                LEFT JOIN thoughts ON thoughts.note_id = notes.id
                WHERE notes.id = ?
                GROUP BY notes.id
                """,
                (result.note_id,),
            ).fetchone()
            if row is None:
                continue
            item = _note_list_item(row)
            item.update(
                {
                    "search_score": float(result.score),
                    "search_source": result.source,
                    "search_snippet": _short(result.snippet, 150),
                }
            )
            output.append(item)
    return output


def ask_with_evidence(
    query: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    limit: int = 8,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    notes = search_notes(query, backend=backend, device=device, limit=limit, db_path=db_path)
    if not notes:
        return {
            "answer": "十分な根拠が見つかりませんでした。検索語を変えるか、先に分析を実行してください。",
            "evidence_notes": [],
            "confidence": 0.15,
        }
    top = notes[0]
    answer = (
        f"「{top['display_title']}」などのメモに、質問に関係する記録が残っている可能性があります。"
        "これは検索された根拠断片にもとづく暫定回答で、未分析のメモが多い場合は控えめに読んでください。"
    )
    return {"answer": answer, "evidence_notes": notes, "confidence": min(0.75, 0.35 + len(notes) * 0.05)}


def get_note_detail(note_id_or_choice: str | None, db_path: str | Path | None = None) -> dict[str, Any]:
    note_id = extract_note_id(note_id_or_choice)
    if not note_id:
        return {"found": False, "message": "メモを選択してください。"}
    init_db(db_path)
    with connect(db_path) as conn:
        note = conn.execute("SELECT * FROM notes WHERE id LIKE ? LIMIT 1", (f"{note_id}%",)).fetchone()
        if note is None:
            return {"found": False, "message": "メモが見つかりません。"}
        summary = _row_to_dict(conn.execute("SELECT * FROM note_summaries WHERE note_id = ?", (note["id"],)).fetchone())
        categories = get_note_ai_outputs(note["id"], db_path=db_path)["categories"]
        events = get_note_events(note["id"], db_path=db_path)
        thoughts = get_note_thoughts(note["id"], db_path=db_path)
        model_runs = _model_runs_for_note(conn, note)
        reflection = _reflection_for_note_month(conn, note)
    detail = {
        "found": True,
        "note": dict(note),
        "summary": _decode_summary(summary) if summary else None,
        "categories": categories,
        "events": events,
        "thoughts": thoughts,
        "model_runs": model_runs,
        "reflection": reflection,
    }
    detail["warnings"] = _detail_warnings(detail)
    return detail


def get_note_ai_outputs(note_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        summary = _row_to_dict(conn.execute("SELECT * FROM note_summaries WHERE note_id = ?", (note_id,)).fetchone())
        categories = [
            {
                "name": row["name"],
                "confidence": float(row["confidence"] or 0.0),
                "importance": float(row["importance"] or 0.0),
                "evidence": _parse_evidence(row["evidence_json"]),
            }
            for row in conn.execute(
                """
                SELECT categories.name, note_categories.confidence, note_categories.importance, note_categories.evidence_json
                FROM note_categories
                JOIN categories ON categories.id = note_categories.category_id
                WHERE note_categories.note_id = ?
                ORDER BY note_categories.confidence DESC
                """,
                (note_id,),
            ).fetchall()
        ]
    return {"summary": _decode_summary(summary) if summary else None, "categories": categories}


def get_note_events(note_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM events WHERE note_id = ? ORDER BY importance DESC, created_at DESC", (note_id,)).fetchall()
    return [_decode_event(row) for row in rows]


def get_note_thoughts(note_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM thoughts WHERE note_id = ? ORDER BY importance DESC, created_at DESC", (note_id,)).fetchall()
    return [_decode_thought(row) for row in rows]


def get_timeline(
    month: str | None = None,
    *,
    sort: str = "date",
    limit: int = 100,
    db_path: str | Path | None = None,
) -> list[TimelineItem]:
    items = build_timeline(month or None, db_path=db_path, limit=int(limit))
    if sort == "importance":
        return sorted(items, key=lambda item: item.importance, reverse=True)
    return items


def list_timeline_years(db_path: str | Path | None = None) -> list[str]:
    months = list_timeline_months(db_path=db_path, order="desc")
    years = sorted({item.month[:4] for item in months if item.month}, reverse=True)
    return years


def get_timeline_months(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    return [item.to_dict() for item in list_timeline_months(db_path=db_path, order="desc")]


def get_timeline_month_snapshots(
    *,
    year: str | None = None,
    category: str | None = None,
    theme: str | None = None,
    item_type: str = "all",
    sort: str = "chronological_desc",
    limit: int = 24,
    db_path: str | Path | None = None,
) -> list[MonthTimelineSnapshot]:
    order = "asc" if sort == "chronological_asc" else "desc"
    snapshots = list_month_timeline_snapshots(year=year or None, db_path=db_path, order=order, limit=int(limit))
    if category:
        snapshots = [snapshot for snapshot in snapshots if category in snapshot.dominant_categories or category in snapshot.key_themes]
    if theme:
        snapshots = [
            snapshot
            for snapshot in snapshots
            if any(theme in value for value in snapshot.key_themes + snapshot.dominant_categories)
        ]
    if item_type and item_type != "all":
        key = "thought" if item_type == "thoughts" else "event" if item_type == "events" else item_type.rstrip("s")
        snapshots = [snapshot for snapshot in snapshots if any(item.item_type == key for item in snapshot.items)]
    if sort == "importance_desc":
        snapshots = sorted(snapshots, key=lambda snapshot: snapshot.importance, reverse=True)
    return snapshots[: int(limit)]


def generate_timeline_snapshot_ui(
    *,
    month: str | None,
    force: bool = False,
    dry_run: bool = True,
    backend: str = "rule",
    db_path: str | Path | None = None,
) -> tuple[str, MonthTimelineSnapshot | None]:
    if not month:
        return "Select a month before generating timeline.", None
    if get_running_jobs().get("analyze_all"):
        return "analyze-all is running. Timeline generation is disabled until it finishes.", None
    snapshot = generate_month_timeline_snapshot(
        month,
        db_path=db_path,
        backend=backend,
        force=force,
        dry_run=dry_run,
    )
    mode = "dry-run" if dry_run else "saved"
    return f"timeline {mode}: {snapshot.month} items={len(snapshot.items)} warnings={len(snapshot.quality.get('warnings') or [])}", snapshot


def get_timeline_qa(month: str | None = None, all_months: bool = False, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    return timeline_qa(month=month, all_months=all_months, db_path=db_path)


def get_reflection(month: str | None = None, db_path: str | Path | None = None) -> ReflectionReport:
    return build_monthly_reflection(month or None, db_path=db_path)


def get_model_status() -> list[dict[str, Any]]:
    return [status.to_dict() for status in model_statuses()]


def get_cuda_status_summary() -> dict[str, Any]:
    status = collect_cuda_status()
    return {
        "torch_version": status.torch_version,
        "torch_cuda_version": status.torch_cuda_version,
        "cuda_available": status.cuda_available,
        "device_count": status.device_count,
        "likely_reason": status.likely_reason,
        "gpus": [device.__dict__ for device in status.devices],
        "nvidia_smi_driver": status.nvidia_smi_driver_version,
        "nvidia_smi_cuda": status.nvidia_smi_cuda_version,
    }


def get_missing_analysis_counts(db_path: str | Path | None = None) -> dict[str, Any]:
    return get_analysis_health(db_path)


def get_running_jobs() -> dict[str, Any]:
    pattern = "notes_lifelog_rag.cli analyze-all"
    try:
        result = subprocess.run(["ps", "aux"], check=False, capture_output=True, text=True, timeout=2)
    except Exception as exc:  # pragma: no cover - platform fallback.
        return {"analyze_all": False, "matches": [], "error": str(exc)}
    matches = [
        line
        for line in result.stdout.splitlines()
        if pattern in line and "grep" not in line and "pytest" not in line
    ]
    return {"analyze_all": bool(matches), "matches": matches}


def get_note_summary(note_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    return get_note_ai_outputs(note_id, db_path=db_path)["summary"]


def get_note_categories(note_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    return get_note_ai_outputs(note_id, db_path=db_path)["categories"]


def get_note_model_runs(note_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    detail = get_note_detail(note_id, db_path=db_path)
    return detail.get("model_runs", []) if detail.get("found") else []


def today_rediscovery(limit: int = 20, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    month_day = datetime.now().strftime("%m-%d")
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                notes.id,
                notes.title,
                notes.source_relative_path,
                COALESCE(notes.folder, '') AS folder,
                COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at, '') AS date_value,
                notes.content_hash,
                note_summaries.generated_title,
                note_summaries.one_line_summary,
                note_summaries.confidence AS summary_confidence,
                note_summaries.importance AS summary_importance,
                note_summaries.evidence_json AS summary_evidence_json,
                GROUP_CONCAT(DISTINCT categories.name) AS category_names,
                COUNT(DISTINCT events.id) AS event_count,
                COUNT(DISTINCT thoughts.id) AS thought_count
            FROM notes
            LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
            LEFT JOIN note_categories ON note_categories.note_id = notes.id
            LEFT JOIN categories ON categories.id = note_categories.category_id
            LEFT JOIN events ON events.note_id = notes.id
            LEFT JOIN thoughts ON thoughts.note_id = notes.id
            WHERE SUBSTR(COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at, ''), 6, 5) = ?
               OR COALESCE(note_summaries.importance, 0.0) >= 0.75
               OR EXISTS (SELECT 1 FROM thoughts WHERE thoughts.note_id = notes.id AND COALESCE(thoughts.importance, 0.0) >= 0.65)
            GROUP BY notes.id
            ORDER BY COALESCE(note_summaries.importance, 0.0) DESC,
                     COALESCE(notes.note_date, notes.modified_at, notes.created_at, notes.imported_at) DESC
            LIMIT ?
            """,
            (month_day, int(limit)),
        ).fetchall()
    return [_note_list_item(row, rediscovery=True) for row in rows]


def import_notes(input_path: str | None = None) -> tuple[str, list[list[Any]]]:
    path = raw_notes_path(input_path or None)
    summary = ingest_directory(path)
    message = (
        f"Imported {summary.imported_notes} notes. "
        f"Duplicates {summary.skipped_duplicates}, unsupported {summary.skipped_unsupported}, parser errors {summary.parser_errors}."
    )
    return message, db_stats_rows()


def initialize_database() -> tuple[str, list[list[Any]]]:
    path = init_db()
    return f"Initialized {path}", db_stats_rows()


def run_build_embeddings(
    limit: int | str | None = 10,
    backend: str = "mock",
    *,
    dry_run: bool = False,
    only_missing: bool = True,
    force: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 16,
) -> str:
    running = get_running_jobs()
    if running.get("analyze_all"):
        return "analyze-all is already running. Build embeddings is disabled from UI until it finishes."
    limit_value = _optional_int(limit)
    device_info = resolve_device(device)
    selected = get_embedding_backend(backend, device_info=device_info, dtype=dtype, batch_size=int(batch_size))
    summary = build_chunk_embeddings(
        selected,
        limit=limit_value,
        force=force,
        only_missing=only_missing,
        dry_run=dry_run,
        batch_size=int(batch_size),
    )
    if summary.disabled_reason:
        return f"disabled: {summary.disabled_reason}"
    return (
        f"{summary.model_name}: device={device_info.resolved_device}, dtype={effective_dtype(dtype, device_info)}, "
        f"scanned={summary.scanned_chunks}, selected={summary.selected_chunks}, "
        f"would_embed={summary.would_embed_chunks}, embedded={summary.embedded_chunks}, failed={summary.failed_chunks}, "
        f"dry_run={summary.dry_run}"
    )


def run_analyze(
    limit: int | str | None = 10,
    backend: str = "mock",
    *,
    dry_run: bool = False,
    only_missing: bool = True,
    force: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
) -> str:
    running = get_running_jobs()
    if running.get("analyze_all"):
        return "analyze-all is already running. UI analysis jobs are disabled until it finishes."
    limit_value = _optional_int(limit)
    device_info = resolve_device(device)
    summaries = analyze_all(
        limit=limit_value,
        backend_name=backend,
        dry_run=dry_run,
        only_missing=only_missing,
        force=force,
        device_info=device_info,
        dtype=dtype,
        batch_size=int(batch_size),
        max_new_tokens=int(max_new_tokens),
    )
    lines = [f"device={device_info.resolved_device}, dtype={effective_dtype(dtype, device_info)}, dry_run={dry_run}"]
    lines.extend(
        f"{s.task_name}: scanned={s.scanned_notes}, eligible={s.eligible_notes}, selected={s.selected_notes}, "
        f"processed={s.processed_notes}, cached={s.cached_notes}, failed={s.failed_notes}, items={s.created_items}"
        for s in summaries
    )
    return "\n".join(lines)


def run_generate_suggestions(limit: int | str | None = 50, month: str | None = None, today: bool = False, force: bool = False) -> str:
    if get_running_jobs().get("analyze_all"):
        return "analyze-all is already running. Suggestion generation is disabled until it finishes."
    result = generate_suggestions(limit=_optional_int(limit) or 50, month=month or None, today=today, force=force)
    return f"generate-suggestions: created={result['created']}, skipped={result['skipped']}, candidates={result['candidates']}"


def run_generate_reflections(month: str | None = None, all_months: bool = False, force: bool = False) -> str:
    if get_running_jobs().get("analyze_all"):
        return "analyze-all is already running. Reflection generation is disabled until it finishes."
    reports = generate_reflections(month=month or None, all_months=all_months, force=force)
    return "generate-reflections: " + ", ".join(f"{report.month} warnings={len(report.quality_warnings)}" for report in reports)


def qa_report(month: str | None = None, limit: int = 50, db_path: str | Path | None = None) -> str:
    month_filter = month.strip() if month else None
    warnings = get_quality_warnings(month=month, limit=limit, db_path=db_path)
    failures = [item for item in warnings if item["warning_type"] == "model_run_failure"]
    import_errors = [item for item in warnings if item["warning_type"] == "import_error"]
    low_rows = [item for item in warnings if item["warning_type"] == "low_confidence"]
    lines = ["## QA Report", "", f"- month: `{month_filter or 'all'}`", ""]
    lines.append("### Low Confidence Summaries")
    lines.extend(
        [
            f"- `{row['note_id'][:12]}` {row['title']} confidence={float(row.get('confidence') or 0.0):.2f} importance={float(row.get('importance') or 0.0):.2f}"
            for row in low_rows[:limit]
        ]
        or ["- なし"]
    )
    lines.append("\n### Model Run Failures")
    lines.extend(
        [f"- {row['created_at']} {row['issue']}" for row in failures[:limit]]
        or ["- なし"]
    )
    lines.append("\n### Import Errors")
    lines.extend(
        [f"- {row['created_at']} {row['source_path']} - {row['issue']}" for row in import_errors[:limit]]
        or ["- なし"]
    )
    return "\n".join(lines)


def get_quality_warnings(
    *,
    month: str | None = None,
    limit: int = 100,
    warning_types: set[str] | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
    month_filter = (month or "").strip()
    warnings: list[dict[str, Any]] = []

    def wanted(kind: str) -> bool:
        return warning_types is None or kind in warning_types

    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT notes.id AS note_id, notes.title, notes.source_relative_path, note_summaries.generated_title,
                   note_summaries.one_line_summary, note_summaries.detailed_summary,
                   note_summaries.confidence, note_summaries.importance, note_summaries.evidence_json,
                   COALESCE(notes.note_date, notes.modified_at, notes.imported_at, '') AS date_value
            FROM notes
            LEFT JOIN note_summaries ON note_summaries.note_id = notes.id
            WHERE (? = '' OR COALESCE(notes.note_date, notes.modified_at, notes.imported_at, '') LIKE ?)
            LIMIT ?
            """,
            (month_filter, f"{month_filter}%", max(int(limit) * 4, 50)),
        ).fetchall()
        for row in rows:
            evidence = _parse_evidence(row["evidence_json"])
            confidence = _float(row["confidence"], None)
            importance = _float(row["importance"], None)
            one_line = row["one_line_summary"] or ""
            detailed = row["detailed_summary"] or ""
            if confidence is not None and confidence <= 0.45 and wanted("low_confidence"):
                warnings.append(_quality_warning("low_confidence", row, "summary confidence is low", confidence, importance, evidence))
            if _evidence_missing(row["evidence_json"]) and wanted("evidence_missing"):
                warnings.append(_quality_warning("evidence_missing", row, "summary evidence is missing or too short", confidence, importance, evidence))
            if _evidence_title_only(evidence, row["title"]) and wanted("evidence_title_only"):
                warnings.append(_quality_warning("evidence_title_only", row, "evidence quote looks like title only", confidence, importance, evidence))
            if (not one_line.strip() and not detailed.strip()) and wanted("empty_summary"):
                warnings.append(_quality_warning("empty_summary", row, "summary is empty", confidence, importance, evidence))
            if 0 < len(one_line.strip()) <= 12 and wanted("very_short_summary"):
                warnings.append(_quality_warning("very_short_summary", row, "summary is very short", confidence, importance, evidence))

        event_rows = conn.execute(
            """
            SELECT events.id, events.note_id, notes.title, notes.source_relative_path, events.title AS item_title,
                   events.summary, events.event_date, events.date_label, events.confidence, events.importance,
                   events.evidence_json
            FROM events
            JOIN notes ON notes.id = events.note_id
            WHERE (? = '' OR COALESCE(events.event_date, events.date_label, notes.note_date, notes.modified_at, '') LIKE ?)
            ORDER BY COALESCE(events.confidence, 1.0) ASC, COALESCE(events.importance, 0.0) DESC
            LIMIT ?
            """,
            (month_filter, f"{month_filter}%", max(int(limit) * 4, 50)),
        ).fetchall()
        for row in event_rows:
            evidence = _parse_evidence(row["evidence_json"])
            confidence = _float(row["confidence"], None)
            importance = _float(row["importance"], None)
            if confidence is not None and confidence <= 0.45 and wanted("low_confidence"):
                warnings.append(_quality_warning("low_confidence", row, "event confidence is low", confidence, importance, evidence))
            if _evidence_missing(row["evidence_json"]) and wanted("evidence_missing"):
                warnings.append(_quality_warning("evidence_missing", row, "event evidence is missing", confidence, importance, evidence))
            if not str(row["event_date"] or row["date_label"] or "").strip() and wanted("unknown_event_date"):
                warnings.append(_quality_warning("unknown_event_date", row, "event date is unknown", confidence, importance, evidence))

        thought_rows = conn.execute(
            """
            SELECT thoughts.id, thoughts.note_id, notes.title, notes.source_relative_path, thoughts.title AS item_title,
                   thoughts.summary, thoughts.confidence, thoughts.importance, thoughts.evidence_json
            FROM thoughts
            JOIN notes ON notes.id = thoughts.note_id
            WHERE (? = '' OR COALESCE(thoughts.date_label, notes.note_date, notes.modified_at, '') LIKE ?)
            ORDER BY COALESCE(thoughts.confidence, 1.0) ASC, COALESCE(thoughts.importance, 0.0) DESC
            LIMIT ?
            """,
            (month_filter, f"{month_filter}%", max(int(limit) * 4, 50)),
        ).fetchall()
        for row in thought_rows:
            evidence = _parse_evidence(row["evidence_json"])
            confidence = _float(row["confidence"], None)
            importance = _float(row["importance"], None)
            if confidence is not None and confidence <= 0.45 and wanted("low_confidence"):
                warnings.append(_quality_warning("low_confidence", row, "thought confidence is low", confidence, importance, evidence))
            if _evidence_missing(row["evidence_json"]) and wanted("evidence_missing"):
                warnings.append(_quality_warning("evidence_missing", row, "thought evidence is missing", confidence, importance, evidence))

        if wanted("model_run_failure"):
            failures = conn.execute(
                """
                SELECT model_runs.task_name, model_runs.model_name, model_runs.note_id,
                       notes.title, notes.source_relative_path,
                       COALESCE(NULLIF(model_runs.error_type, ''), 'legacy_unknown_failure') AS error_type,
                       COALESCE(NULLIF(model_runs.error_message, ''), 'model run failed') AS error_message,
                       model_runs.created_at
                FROM model_runs
                LEFT JOIN notes ON notes.id = model_runs.note_id
                WHERE model_runs.success = 0
                ORDER BY model_runs.created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            for row in failures:
                warnings.append(
                    {
                        "warning_type": "model_run_failure",
                        "note_id": row["note_id"] or "",
                        "title": row["title"] or row["task_name"],
                        "source_path": row["source_relative_path"] or row["model_name"],
                        "issue": f"{row['error_type']}: {row['error_message']}",
                        "confidence": None,
                        "importance": None,
                        "evidence": [],
                        "created_at": row["created_at"],
                    }
                )
        if wanted("import_error"):
            errors = conn.execute(
                "SELECT source_path, parser_name, error_message, created_at FROM import_errors ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            for row in errors:
                warnings.append(
                    {
                        "warning_type": "import_error",
                        "note_id": "",
                        "title": row["parser_name"] or "import error",
                        "source_path": row["source_path"],
                        "issue": row["error_message"],
                        "confidence": None,
                        "importance": None,
                        "evidence": [],
                        "created_at": row["created_at"],
                    }
                )
    return warnings[: int(limit)]


def list_suggestions(limit: int = 50, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT suggestions.*, notes.title AS note_title, notes.source_relative_path
            FROM suggestions
            LEFT JOIN notes ON notes.id = suggestions.note_id
            ORDER BY COALESCE(suggestions.importance, 0.0) DESC, suggestions.created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_decode_suggestion(row) for row in rows]


def generate_suggestions(
    *,
    limit: int = 100,
    month: str | None = None,
    today: bool = False,
    force: bool = False,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    target_date = datetime.now().date().isoformat()
    month_filter = (month or "").strip()
    if force:
        with connect(db_path) as conn:
            if month_filter:
                conn.execute("DELETE FROM suggestions WHERE target_date LIKE ?", (f"{month_filter}%",))
            elif today:
                conn.execute("DELETE FROM suggestions WHERE target_date = ?", (target_date,))
            else:
                conn.execute("DELETE FROM suggestions")
    candidates = _suggestion_candidates(limit=limit, month=month_filter or None, today=today, db_path=db_path)
    inserted = 0
    skipped = 0
    with connect(db_path) as conn:
        for candidate in candidates[: int(limit)]:
            exists = conn.execute(
                """
                SELECT 1 FROM suggestions
                WHERE suggestion_type = ?
                  AND COALESCE(note_id, '') = COALESCE(?, '')
                  AND title = ?
                  AND COALESCE(target_date, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (candidate["suggestion_type"], candidate.get("note_id"), candidate["title"], candidate.get("target_date")),
            ).fetchone()
            if exists and not force:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO suggestions(
                    note_id, suggestion_type, title, message, target_date,
                    importance, confidence, evidence_json, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.get("note_id"),
                    candidate["suggestion_type"],
                    candidate["title"],
                    candidate["message"],
                    candidate.get("target_date") or target_date,
                    candidate.get("importance"),
                    candidate.get("confidence"),
                    json.dumps(candidate.get("evidence") or [], ensure_ascii=False),
                    "new",
                    now,
                ),
            )
            inserted += 1
    return {"created": inserted, "skipped": skipped, "candidates": len(candidates), "limit": int(limit)}


def list_reflections(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM monthly_reflections ORDER BY month DESC").fetchall()
    output = []
    for row in rows:
        value = dict(row)
        try:
            value["summary"] = json.loads(row["summary_json"])
        except json.JSONDecodeError:
            value["summary"] = {}
        value["evidence"] = _parse_evidence(row["evidence_json"])
        output.append(value)
    return output


def generate_reflections(
    *,
    month: str | None = None,
    all_months: bool = False,
    force: bool = False,
    db_path: str | Path | None = None,
) -> list[ReflectionReport]:
    months = [month] if month else []
    if all_months or not months:
        months = [row["month"] for row in list_months_with_counts(db_path)]
    reports = []
    for value in months:
        if value:
            reports.append(build_monthly_reflection(value, db_path=db_path, force=force))
    return reports


def timeline_ui(month: str, limit: int = 100) -> tuple[str, list[list[Any]]]:
    items = get_timeline(month or None, limit=int(limit))
    rows = [
        [
            item.date_label,
            item.item_type,
            item.title,
            f"{item.confidence:.2f}",
            f"{item.importance:.2f}",
            (item.evidence[0]["quote"] if item.evidence else ""),
            item.note_id[:12],
        ]
        for item in items
    ]
    return format_timeline_markdown(items, month=month or None), rows


def reflection_ui(month: str) -> str:
    report = get_reflection(month or None)
    return format_reflection_markdown(report)


def model_rows() -> list[list[Any]]:
    return [
        [
            s["purpose"],
            s["name"],
            "exists" if s["path_exists"] else "missing",
            s["runtime_status"] if not s["runtime_available"] else "ready",
            s["cuda_status"],
            "yes" if s["enabled"] else "disabled",
            s["reason"],
        ]
        for s in get_model_status()
    ]


def settings_markdown() -> str:
    cuda = get_cuda_status_summary()
    return (
        "## Local Settings\n\n"
        f"- Database: `{database_path()}`\n"
        f"- Raw notes: `{raw_notes_path()}`\n"
        "- Default host: `127.0.0.1`\n"
        "- External APIs: disabled\n"
        "- Model downloads: never automatic\n"
        f"- CUDA: `{'available' if cuda['cuda_available'] else 'unavailable'}` / `{cuda['likely_reason']}`\n"
    )


def note_detail(choice_or_id: str | None) -> tuple[str, str, str, str]:
    detail = get_note_detail(choice_or_id)
    if not detail.get("found"):
        return detail.get("message", "メモを選択してください。"), "", "", ""
    note = detail["note"]
    summary = detail.get("summary")
    meta = (
        f"# {note['title']}\n\n"
        f"- note_id: `{note['id']}`\n"
        f"- source: `{note['source_relative_path']}`\n"
        f"- date: `{note['note_date'] or note['date_label'] or 'unknown'}` "
        f"({note['date_confidence'] or 'unknown'})\n"
    )
    analysis = ["## AI Outputs"]
    if summary:
        analysis.append(f"### {summary['generated_title'] or note['title']}")
        analysis.append(summary.get("one_line_summary") or "")
        analysis.append(
            f"confidence `{summary.get('confidence', 0.0):.2f}` / importance `{summary.get('importance', 0.0):.2f}`"
        )
    for title, items in [("Categories", detail["categories"]), ("Events", detail["events"]), ("Thoughts", detail["thoughts"])]:
        if items:
            analysis.append(f"### {title}")
            for item in items:
                label = item.get("name") or item.get("title")
                analysis.append(
                    f"- {label}: confidence `{float(item.get('confidence') or 0.0):.2f}` / "
                    f"importance `{float(item.get('importance') or 0.0):.2f}`"
                )
    return meta, note["body"], "\n".join(analysis), note["id"]


def legacy_list_notes(limit: int = 100, query: str = "") -> tuple[list[list[Any]], list[str]]:
    notes = list_notes(limit=limit, query=query)
    rows = [
        [
            item["id"][:12],
            item["display_title"],
            item["date_label"],
            item["source_short"],
            item["one_line_summary"],
        ]
        for item in notes
    ]
    return rows, note_choices(notes)


def note_choices(notes: list[dict[str, Any]]) -> list[str]:
    return [_choice(item["id"], item["display_title"]) for item in notes]


def extract_note_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.split("·", 1)[0].strip()


def _note_list_item(row: sqlite3.Row, *, rediscovery: bool = False) -> dict[str, Any]:
    category_names = [value for value in str(row["category_names"] or "").split(",") if value]
    confidence = _float(row["summary_confidence"], None)
    importance = _float(row["summary_importance"], None)
    generated_title = row["generated_title"] or ""
    title = row["title"] or "Untitled"
    one_line = row["one_line_summary"] or ""
    item = {
        "id": row["id"],
        "title": title,
        "generated_title": generated_title,
        "display_title": generated_title or title,
        "one_line_summary": one_line,
        "date_label": _short(str(row["date_value"] or ""), 10),
        "source_path": row["source_relative_path"] or "",
        "source_short": _short_path(row["source_relative_path"] or ""),
        "folder": row["folder"] or "",
        "categories": category_names,
        "confidence": confidence,
        "importance": importance,
        "event_count": int(row["event_count"] or 0),
        "thought_count": int(row["thought_count"] or 0),
        "content_hash": row["content_hash"],
        "has_summary": bool(generated_title or one_line),
        "low_confidence": confidence is not None and confidence <= 0.45,
        "important": importance is not None and importance >= 0.75,
        "evidence_missing": _evidence_missing(row["summary_evidence_json"]),
        "rediscovery": rediscovery,
    }
    return item


def _decode_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    row = dict(row)
    row["important_points"] = _parse_json_list(row.get("important_points_json"))
    row["evidence"] = _parse_evidence(row.get("evidence_json"))
    row["confidence"] = _float(row.get("confidence"), 0.0)
    row["importance"] = _float(row.get("importance"), 0.0)
    row["low_confidence"] = row["confidence"] <= 0.45
    row["evidence_missing"] = _evidence_missing(row.get("evidence_json"))
    return row


def _decode_event(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["confidence"] = _float(item.get("confidence"), 0.0)
    item["importance"] = _float(item.get("importance"), 0.0)
    item["evidence"] = _parse_evidence(item.get("evidence_json"))
    item["low_confidence"] = item["confidence"] <= 0.45
    item["evidence_missing"] = _evidence_missing(item.get("evidence_json"))
    return item


def _decode_thought(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["themes"] = _parse_json_list(item.get("themes_json"))
    item["confidence"] = _float(item.get("confidence"), 0.0)
    item["importance"] = _float(item.get("importance"), 0.0)
    item["evidence"] = _parse_evidence(item.get("evidence_json"))
    item["low_confidence"] = item["confidence"] <= 0.45
    item["evidence_missing"] = _evidence_missing(item.get("evidence_json"))
    return item


def _model_runs_for_note(conn: sqlite3.Connection, note: sqlite3.Row) -> list[dict[str, Any]]:
    hashes = [_analysis_input_hash(task, note) for task in ["summary", "categories", "events", "thoughts"]]
    placeholders = ",".join("?" for _ in hashes)
    rows = conn.execute(
        f"""
        SELECT task_name, model_name, success, error_type, error_message, prompt_version,
               empty_result, retry_count, fallback_used, created_at
        FROM model_runs
        WHERE input_hash IN ({placeholders})
        ORDER BY created_at DESC
        """,
        hashes,
    ).fetchall()
    return [dict(row) for row in rows]


def _analysis_input_hash(task_name: str, note: sqlite3.Row) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode("utf-8"))
    digest.update(task_name.encode("utf-8"))
    digest.update(str(note["id"]).encode("utf-8"))
    digest.update(str(note["content_hash"]).encode("utf-8"))
    return digest.hexdigest()


def _reflection_for_note_month(conn: sqlite3.Connection, note: sqlite3.Row) -> dict[str, Any] | None:
    date_value = note["note_date"] or note["modified_at"] or note["created_at"] or note["imported_at"]
    if not date_value or len(str(date_value)) < 7:
        return None
    month = str(date_value)[:7]
    row = conn.execute("SELECT * FROM monthly_reflections WHERE month = ?", (month,)).fetchone()
    if not row:
        return None
    value = dict(row)
    try:
        value["summary"] = json.loads(row["summary_json"])
    except json.JSONDecodeError:
        value["summary"] = {}
    value["evidence"] = _parse_evidence(row["evidence_json"])
    return value


def _detail_warnings(detail: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not detail.get("summary"):
        warnings.append("summary_missing")
    if not detail.get("events"):
        warnings.append("events_sparse")
    if not detail.get("thoughts"):
        warnings.append("thoughts_sparse")
    items: list[dict[str, Any]] = []
    if detail.get("summary"):
        items.append(detail["summary"])
    items.extend(detail.get("events") or [])
    items.extend(detail.get("thoughts") or [])
    if any(item.get("low_confidence") for item in items):
        warnings.append("low_confidence")
    if any(item.get("evidence_missing") for item in items):
        warnings.append("evidence_missing")
    return warnings


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _sort_clause(sort: str) -> str:
    return {
        "created_desc": "COALESCE(notes.created_at, notes.note_date, notes.imported_at) DESC",
        "importance_desc": "COALESCE(note_summaries.importance, 0.0) DESC, COALESCE(notes.modified_at, notes.imported_at) DESC",
        "confidence_asc": "COALESCE(note_summaries.confidence, 1.0) ASC, COALESCE(notes.modified_at, notes.imported_at) DESC",
        "title": "notes.title COLLATE NOCASE ASC",
        "updated_desc": "COALESCE(notes.modified_at, notes.note_date, notes.imported_at) DESC",
    }.get(sort, "COALESCE(notes.modified_at, notes.note_date, notes.imported_at) DESC")


def _parse_json_list(payload: str | None) -> list[Any]:
    try:
        value = json.loads(payload or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _parse_evidence(payload: str | None) -> list[dict[str, str]]:
    values = _parse_json_list(payload)
    evidence: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            evidence.append(
                {
                    "note_id": str(item.get("note_id") or ""),
                    "quote": _short(str(item.get("quote") or ""), 160),
                }
            )
    return evidence


def _quality_warning(
    kind: str,
    row: sqlite3.Row,
    issue: str,
    confidence: float | None,
    importance: float | None,
    evidence: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "warning_type": kind,
        "note_id": row["note_id"],
        "title": row["title"],
        "source_path": row["source_relative_path"],
        "issue": issue,
        "confidence": confidence,
        "importance": importance,
        "evidence": evidence,
        "created_at": "",
    }


def _evidence_missing(payload: str | None) -> bool:
    evidence = _parse_evidence(payload)
    if not evidence:
        return True
    quote = evidence[0].get("quote", "").strip()
    return len(quote) < 6


def _evidence_title_only(evidence: list[dict[str, str]], title: str) -> bool:
    normalized_title = " ".join(str(title or "").strip().lower().split())
    if not normalized_title:
        return False
    for item in evidence:
        quote = " ".join(str(item.get("quote") or "").strip().lower().split())
        if quote and (quote == normalized_title or quote in normalized_title or normalized_title in quote):
            return True
    return False


def _decode_suggestion(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["evidence"] = _parse_evidence(item.get("evidence_json"))
    item["importance"] = _float(item.get("importance"), 0.0)
    item["confidence"] = _float(item.get("confidence"), 0.0)
    return item


def _suggestion_candidates(
    *,
    limit: int,
    month: str | None,
    today: bool,
    db_path: str | Path | None,
) -> list[dict[str, Any]]:
    target_date = datetime.now().date().isoformat()
    month_day = datetime.now().strftime("%m-%d")
    candidates: list[dict[str, Any]] = []
    with connect(db_path) as conn:
        thoughts = conn.execute(
            """
            SELECT thoughts.note_id, thoughts.title, thoughts.summary, thoughts.remember_reason,
                   thoughts.importance, thoughts.confidence, thoughts.evidence_json,
                   notes.title AS note_title, notes.note_date, notes.modified_at
            FROM thoughts
            JOIN notes ON notes.id = thoughts.note_id
            WHERE (? IS NULL OR COALESCE(thoughts.date_label, notes.note_date, notes.modified_at, '') LIKE ?)
            ORDER BY COALESCE(thoughts.importance, 0.0) DESC, COALESCE(thoughts.confidence, 0.0) DESC
            LIMIT ?
            """,
            (month, f"{month}%" if month else None, max(limit, 20)),
        ).fetchall()
        for row in thoughts:
            if _float(row["importance"], 0.0) >= 0.65:
                candidates.append(
                    {
                        "suggestion_type": "important_thought",
                        "note_id": row["note_id"],
                        "title": row["title"] or row["note_title"],
                        "message": row["remember_reason"] or row["summary"] or "重要度の高い思考として見返す価値があります。",
                        "target_date": target_date,
                        "importance": _float(row["importance"], 0.0),
                        "confidence": _float(row["confidence"], 0.0),
                        "evidence": _parse_evidence(row["evidence_json"]),
                    }
                )
        events = conn.execute(
            """
            SELECT events.note_id, events.title, events.summary, events.importance, events.confidence,
                   events.evidence_json, notes.title AS note_title
            FROM events
            JOIN notes ON notes.id = events.note_id
            WHERE (? IS NULL OR COALESCE(events.event_date, events.date_label, notes.note_date, notes.modified_at, '') LIKE ?)
            ORDER BY COALESCE(events.importance, 0.0) DESC, COALESCE(events.confidence, 0.0) DESC
            LIMIT ?
            """,
            (month, f"{month}%" if month else None, max(limit, 20)),
        ).fetchall()
        for row in events:
            if _float(row["importance"], 0.0) >= 0.7:
                candidates.append(
                    {
                        "suggestion_type": "important_event",
                        "note_id": row["note_id"],
                        "title": row["title"] or row["note_title"],
                        "message": row["summary"] or "重要度の高い出来事として見返す価値があります。",
                        "target_date": target_date,
                        "importance": _float(row["importance"], 0.0),
                        "confidence": _float(row["confidence"], 0.0),
                        "evidence": _parse_evidence(row["evidence_json"]),
                    }
                )
        summaries = conn.execute(
            """
            SELECT notes.id AS note_id, notes.title AS note_title, notes.note_date, notes.modified_at,
                   note_summaries.generated_title, note_summaries.one_line_summary, note_summaries.revisit_reason,
                   note_summaries.importance, note_summaries.confidence, note_summaries.evidence_json
            FROM note_summaries
            JOIN notes ON notes.id = note_summaries.note_id
            WHERE (? IS NULL OR COALESCE(notes.note_date, notes.modified_at, '') LIKE ?)
            ORDER BY COALESCE(note_summaries.importance, 0.0) DESC, COALESCE(note_summaries.confidence, 0.0) DESC
            LIMIT ?
            """,
            (month, f"{month}%" if month else None, max(limit, 20)),
        ).fetchall()
        for row in summaries:
            note_date = str(row["note_date"] or row["modified_at"] or "")
            same_day = today and len(note_date) >= 10 and note_date[5:10] == month_day
            if same_day:
                suggestion_type = "today_rediscovery"
                message = row["revisit_reason"] or row["one_line_summary"] or "今日と同じ日付に近い過去のメモです。"
            elif row["revisit_reason"]:
                suggestion_type = "revisit_note"
                message = row["revisit_reason"]
            else:
                continue
            candidates.append(
                {
                    "suggestion_type": suggestion_type,
                    "note_id": row["note_id"],
                    "title": row["generated_title"] or row["note_title"],
                    "message": message,
                    "target_date": target_date,
                    "importance": _float(row["importance"], 0.0),
                    "confidence": _float(row["confidence"], 0.0),
                    "evidence": _parse_evidence(row["evidence_json"]),
                }
            )
    for warning in get_quality_warnings(limit=max(10, limit // 2), db_path=db_path):
        if warning["warning_type"] in {"low_confidence", "evidence_missing", "evidence_title_only"} and warning.get("note_id"):
            candidates.append(
                {
                    "suggestion_type": "low_confidence_review"
                    if warning["warning_type"] == "low_confidence"
                    else "evidence_review",
                    "note_id": warning["note_id"],
                    "title": warning["title"],
                    "message": warning["issue"],
                    "target_date": target_date,
                    "importance": warning.get("importance") or 0.45,
                    "confidence": warning.get("confidence") or 0.35,
                    "evidence": warning.get("evidence") or [],
                }
            )
    if month:
        candidates.append(
            {
                "suggestion_type": "monthly_reflection",
                "note_id": None,
                "title": f"{month} のふり返り",
                "message": "この月のevents/thoughts/summariesをもとにReflectionを確認してください。",
                "target_date": f"{month}-01",
                "importance": 0.6,
                "confidence": 0.7,
                "evidence": [],
            }
        )
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda value: float(value.get("importance") or 0.0), reverse=True):
        key = (str(item.get("suggestion_type")), str(item.get("note_id") or ""), str(item.get("title") or ""))
        dedup.setdefault(key, item)
    return list(dedup.values())


def _ratio(value: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return round(float(value) / float(total), 4)


def _choice(note_id: str, title: str) -> str:
    return f"{note_id[:12]} · {title}"


def _short_path(path: str, limit: int = 42) -> str:
    if len(path) <= limit:
        return path
    parts = path.split("/")
    if len(parts) >= 2:
        compact = f"{parts[0]}/.../{parts[-1]}"
        if len(compact) <= limit:
            return compact
    return "..." + path[-(limit - 3) :]


def _short(value: str, limit: int) -> str:
    clean = " ".join(str(value or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    stripped = str(value).strip()
    if not stripped:
        return None
    parsed = int(float(stripped))
    return parsed if parsed > 0 else None
