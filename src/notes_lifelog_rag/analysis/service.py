from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from notes_lifelog_rag.config import database_path, load_categories
from notes_lifelog_rag.db.schema import connect, init_db
from notes_lifelog_rag.llm.backends import LLMBackend, get_llm_backend
from notes_lifelog_rag.runtime.device import DeviceInfo

ProgressCallback = Callable[[int, int, str], None]
StoreFunc = Callable[[sqlite3.Connection, sqlite3.Row, dict[str, Any], str], int]
PROMPT_VERSION = "analysis-v2"


@dataclass
class AnalysisSummary:
    task_name: str
    model_name: str
    backend_name: str = "auto"
    device: str | None = None
    total_notes: int = 0
    scanned_notes: int = 0
    eligible_notes: int = 0
    selected_notes: int = 0
    would_process_notes: int = 0
    processed_notes: int = 0
    cached_notes: int = 0
    cache_hits: int = 0
    skipped_existing: int = 0
    skipped_cached_empty: int = 0
    failed_notes: int = 0
    created_items: int = 0
    limit: int | None = None
    force: bool = False
    dry_run: bool = False
    would_process_note_ids: list[str] | None = None
    disabled_reason: str | None = None


def summarize_notes(
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    all_notes: bool = False,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    backend_name: str = "auto",
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisSummary:
    backend = get_llm_backend(
        backend_name,
        allow_mock_fallback=True,
        device_info=device_info,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    return _run_note_task(
        "summary",
        backend,
        db_path=db_path,
        limit=limit,
        all_notes=all_notes,
        force=force,
        only_missing=only_missing,
        dry_run=dry_run,
        store=_store_summary,
        device_info=device_info,
        progress_callback=progress_callback,
    )


def categorize_notes(
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    all_notes: bool = False,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    backend_name: str = "auto",
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisSummary:
    backend = get_llm_backend(
        backend_name,
        allow_mock_fallback=True,
        device_info=device_info,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    return _run_note_task(
        "categories",
        backend,
        db_path=db_path,
        limit=limit,
        all_notes=all_notes,
        force=force,
        only_missing=only_missing,
        dry_run=dry_run,
        store=_store_categories,
        device_info=device_info,
        progress_callback=progress_callback,
    )


def extract_events(
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    all_notes: bool = False,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    backend_name: str = "auto",
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisSummary:
    backend = get_llm_backend(
        backend_name,
        allow_mock_fallback=True,
        device_info=device_info,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    return _run_note_task(
        "events",
        backend,
        db_path=db_path,
        limit=limit,
        all_notes=all_notes,
        force=force,
        only_missing=only_missing,
        dry_run=dry_run,
        store=_store_events,
        device_info=device_info,
        progress_callback=progress_callback,
    )


def extract_thoughts(
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    all_notes: bool = False,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    backend_name: str = "auto",
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisSummary:
    backend = get_llm_backend(
        backend_name,
        allow_mock_fallback=True,
        device_info=device_info,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    return _run_note_task(
        "thoughts",
        backend,
        db_path=db_path,
        limit=limit,
        all_notes=all_notes,
        force=force,
        only_missing=only_missing,
        dry_run=dry_run,
        store=_store_thoughts,
        device_info=device_info,
        progress_callback=progress_callback,
    )


def analyze_all(
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    all_notes: bool = False,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    backend_name: str = "auto",
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> list[AnalysisSummary]:
    return [
        summarize_notes(
            db_path=db_path,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend_name,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        ),
        categorize_notes(
            db_path=db_path,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend_name,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        ),
        extract_events(
            db_path=db_path,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend_name,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        ),
        extract_thoughts(
            db_path=db_path,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend_name,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        ),
    ]


def _run_note_task(
    task_name: str,
    backend: LLMBackend,
    *,
    db_path: str | Path | None,
    limit: int | None,
    all_notes: bool,
    force: bool,
    only_missing: bool,
    dry_run: bool,
    store: StoreFunc,
    device_info: DeviceInfo | None,
    progress_callback: ProgressCallback | None,
) -> AnalysisSummary:
    if not dry_run:
        init_db(db_path)
    summary = AnalysisSummary(
        task_name=task_name,
        model_name=backend.model_name,
        backend_name=backend.__class__.__name__,
        device=device_info.resolved_device if device_info else None,
        limit=limit,
        force=force,
        dry_run=dry_run,
        would_process_note_ids=[],
    )
    if not backend.is_available() and not dry_run:
        summary.disabled_reason = backend.availability_error()
        return summary
    if not backend.is_available():
        summary.disabled_reason = backend.availability_error()
    selected_limit = limit
    if dry_run and not database_path(db_path).exists():
        summary.disabled_reason = f"database does not exist: {database_path(db_path)}"
        return summary
    with connect(db_path) as conn:
        notes = _select_notes(conn, None)
        summary.total_notes = len(notes)
        notes_to_process = _notes_to_process(
            conn,
            task_name,
            notes,
            model_name=backend.model_name,
            limit=selected_limit,
            only_missing=only_missing,
            force=force,
            summary=summary,
        )
        summary.selected_notes = len(notes_to_process)
        summary.would_process_notes = len(notes_to_process)
        summary.would_process_note_ids = [str(note["id"]) for note in notes_to_process]
        if progress_callback:
            progress_callback(0, len(notes_to_process), task_name)
        categories = load_categories()
        for index, note in enumerate(notes_to_process, start=1):
            input_hash = _input_hash(task_name, note)
            cached = None if force else _cached_model_run(conn, task_name, backend.model_name, input_hash)
            if dry_run:
                if cached is not None:
                    summary.cached_notes += 1
                if progress_callback:
                    progress_callback(index, len(notes_to_process), task_name)
                continue
            if cached is not None:
                payload = cached
                summary.cached_notes += 1
            else:
                try:
                    payload = backend.generate_json(task_name, dict(note), categories=categories)
                    _record_model_run(conn, task_name, backend.model_name, input_hash, payload, True, None)
                except Exception as exc:  # local model JSON/runtime failures should not stop the whole batch.
                    summary.failed_notes += 1
                    _record_model_run(conn, task_name, backend.model_name, input_hash, {}, False, str(exc))
                    conn.commit()
                    if progress_callback:
                        progress_callback(index, len(notes_to_process), task_name)
                    continue
            created = store(conn, note, payload, backend.model_name)
            summary.created_items += created
            summary.processed_notes += 1
            conn.commit()
            if progress_callback:
                progress_callback(index, len(notes_to_process), task_name)
    return summary


def _select_notes(conn: sqlite3.Connection, limit: int | None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM notes ORDER BY COALESCE(note_date, modified_at, imported_at) DESC, title ASC"
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (limit,)).fetchall()
    return conn.execute(sql).fetchall()


def _notes_to_process(
    conn: sqlite3.Connection,
    task_name: str,
    notes: list[sqlite3.Row],
    *,
    model_name: str,
    limit: int | None,
    only_missing: bool,
    force: bool,
    summary: AnalysisSummary,
) -> list[sqlite3.Row]:
    eligible: list[sqlite3.Row] = []
    for note in notes:
        summary.scanned_notes += 1
        if only_missing and not force:
            input_hash = _input_hash(task_name, note)
            cached = _cached_model_run(conn, task_name, model_name, input_hash)
            output_exists = _analysis_output_exists(conn, task_name, note["id"])
            if cached is not None:
                summary.cache_hits += 1
                if not _payload_has_storable_output(task_name, cached):
                    summary.skipped_cached_empty += 1
                    continue
                if output_exists:
                    summary.skipped_existing += 1
                    continue
            elif output_exists:
                # Existing rows without a current model_runs cache are considered stale.
                # They remain in place until a non-dry-run refresh writes current output.
                pass
        eligible.append(note)
    summary.eligible_notes = len(eligible)
    return eligible[:limit] if limit is not None else eligible


def _analysis_output_exists(conn: sqlite3.Connection, task_name: str, note_id: str) -> bool:
    table_by_task = {
        "summary": "note_summaries",
        "categories": "note_categories",
        "events": "events",
        "thoughts": "thoughts",
    }
    table = table_by_task.get(task_name)
    if table is None:
        return False
    row = conn.execute(f"SELECT 1 FROM {table} WHERE note_id = ? LIMIT 1", (note_id,)).fetchone()
    return row is not None


def _payload_has_storable_output(task_name: str, payload: dict[str, Any]) -> bool:
    if task_name == "summary":
        return True
    key_by_task = {
        "categories": ("categories", "name"),
        "events": ("events", "title"),
        "thoughts": ("thoughts", "title"),
    }
    key_pair = key_by_task.get(task_name)
    if key_pair is None:
        return bool(payload)
    list_key, required_key = key_pair
    return any(
        isinstance(item, dict) and bool(_str(item.get(required_key), ""))
        for item in payload.get(list_key) or []
    )


def _store_summary(conn: sqlite3.Connection, note: sqlite3.Row, payload: dict[str, Any], model_name: str) -> int:
    evidence = _normalize_evidence(payload.get("evidence"), note["id"], note["body"])
    conn.execute(
        """
        INSERT INTO note_summaries(
            note_id, model_name, generated_title, one_line_summary, detailed_summary,
            important_points_json, revisit_reason, confidence, importance, evidence_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(note_id) DO UPDATE SET
            model_name = excluded.model_name,
            generated_title = excluded.generated_title,
            one_line_summary = excluded.one_line_summary,
            detailed_summary = excluded.detailed_summary,
            important_points_json = excluded.important_points_json,
            revisit_reason = excluded.revisit_reason,
            confidence = excluded.confidence,
            importance = excluded.importance,
            evidence_json = excluded.evidence_json,
            created_at = excluded.created_at
        """,
        (
            note["id"],
            model_name,
            _str(payload.get("generated_title"), note["title"]),
            _str(payload.get("one_line_summary"), ""),
            _str(payload.get("detailed_summary"), ""),
            json.dumps(payload.get("important_points") or [], ensure_ascii=False),
            _str(payload.get("revisit_reason"), ""),
            _float(payload.get("confidence"), 0.0),
            _float(payload.get("importance"), 0.5),
            json.dumps(evidence, ensure_ascii=False),
            _now(),
        ),
    )
    return 1


def _store_categories(conn: sqlite3.Connection, note: sqlite3.Row, payload: dict[str, Any], model_name: str) -> int:
    items = [
        item
        for item in payload.get("categories") or []
        if isinstance(item, dict) and _str(item.get("name"), "")
    ]
    if not items:
        return 0
    conn.execute("DELETE FROM note_categories WHERE note_id = ?", (note["id"],))
    count = 0
    for item in items:
        name = _str(item.get("name"), "")
        conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        if row is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO note_categories(note_id, category_id, confidence, evidence_json, created_at, importance)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                note["id"],
                row["id"],
                _float(item.get("confidence"), 0.0),
                json.dumps(_normalize_evidence(item.get("evidence"), note["id"], note["body"]), ensure_ascii=False),
                _now(),
                _float(item.get("importance"), 0.45),
            ),
        )
        count += 1
    return count


def _store_events(conn: sqlite3.Connection, note: sqlite3.Row, payload: dict[str, Any], model_name: str) -> int:
    items = [
        item
        for item in payload.get("events") or []
        if isinstance(item, dict) and _str(item.get("title"), "")
    ]
    if not items:
        return 0
    conn.execute("DELETE FROM events WHERE note_id = ?", (note["id"],))
    count = 0
    for item in items:
        title = _str(item.get("title"), "")
        conn.execute(
            """
            INSERT INTO events(
                note_id, title, summary, event_type, event_date, date_label,
                date_confidence, importance, confidence, evidence_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note["id"],
                title,
                _str(item.get("summary"), ""),
                _str(item.get("event_type"), ""),
                _str(item.get("event_date"), None),
                _str(item.get("date_label"), ""),
                _str(item.get("date_confidence"), "unknown"),
                _float(item.get("importance"), 0.0),
                _float(item.get("confidence"), 0.0),
                json.dumps(_normalize_evidence(item.get("evidence"), note["id"], note["body"]), ensure_ascii=False),
                _now(),
            ),
        )
        count += 1
    return count


def _store_thoughts(conn: sqlite3.Connection, note: sqlite3.Row, payload: dict[str, Any], model_name: str) -> int:
    items = [
        item
        for item in payload.get("thoughts") or []
        if isinstance(item, dict) and _str(item.get("title"), "")
    ]
    if not items:
        return 0
    conn.execute("DELETE FROM thoughts WHERE note_id = ?", (note["id"],))
    count = 0
    for item in items:
        title = _str(item.get("title"), "")
        conn.execute(
            """
            INSERT INTO thoughts(
                note_id, title, summary, thought_type, themes_json, emotion_label,
                emotion_intensity, date_label, importance, confidence,
                remember_reason, evidence_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note["id"],
                title,
                _str(item.get("summary"), ""),
                _str(item.get("thought_type"), ""),
                json.dumps(item.get("themes") or [], ensure_ascii=False),
                _str(item.get("emotion_label"), None),
                _float(item.get("emotion_intensity"), None),
                _str(item.get("date_label"), ""),
                _float(item.get("importance"), 0.0),
                _float(item.get("confidence"), 0.0),
                _str(item.get("remember_reason"), ""),
                json.dumps(_normalize_evidence(item.get("evidence"), note["id"], note["body"]), ensure_ascii=False),
                _now(),
            ),
        )
        count += 1
    return count


def _cached_model_run(conn: sqlite3.Connection, task_name: str, model_name: str, input_hash: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT output_json
        FROM model_runs
        WHERE task_name = ? AND model_name = ? AND input_hash = ? AND success = 1
        LIMIT 1
        """,
        (task_name, model_name, input_hash),
    ).fetchone()
    return json.loads(row["output_json"]) if row and row["output_json"] else None


def _record_model_run(
    conn: sqlite3.Connection,
    task_name: str,
    model_name: str,
    input_hash: str,
    payload: dict[str, Any],
    success: bool,
    error_message: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO model_runs(
            task_name, model_name, input_hash, output_json, success, error_message,
            created_at, prompt_version, empty_result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_name, model_name, input_hash) DO UPDATE SET
            output_json = excluded.output_json,
            success = excluded.success,
            error_message = excluded.error_message,
            created_at = excluded.created_at,
            prompt_version = excluded.prompt_version,
            empty_result = excluded.empty_result
        """,
        (
            task_name,
            model_name,
            input_hash,
            json.dumps(payload, ensure_ascii=False),
            1 if success else 0,
            error_message,
            _now(),
            PROMPT_VERSION,
            1 if success and not _payload_has_storable_output(task_name, payload) else 0,
        ),
    )


def _input_hash(task_name: str, note: sqlite3.Row) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode("utf-8"))
    digest.update(task_name.encode("utf-8"))
    digest.update(str(note["id"]).encode("utf-8"))
    digest.update(str(note["content_hash"]).encode("utf-8"))
    return digest.hexdigest()


def _normalize_evidence(value: Any, note_id: str, body: str) -> list[dict[str, str]]:
    if isinstance(value, list) and value:
        normalized = []
        for item in value[:3]:
            if not isinstance(item, dict):
                continue
            quote = _str(item.get("quote"), "")[:120]
            normalized.append({"note_id": _str(item.get("note_id"), note_id), "quote": quote})
        if normalized:
            return normalized
    quote = next((line.strip() for line in body.splitlines() if line.strip()), "")[:80]
    return [{"note_id": note_id, "quote": quote}]


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _str(value: Any, default: str | None = "") -> str | None:
    if value is None:
        return default
    return str(value)


def _float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
