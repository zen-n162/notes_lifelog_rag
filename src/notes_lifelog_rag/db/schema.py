from __future__ import annotations

import sqlite3
from pathlib import Path

from notes_lifelog_rag.config import database_path, load_categories


CORE_TABLES_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    folder TEXT,
    file_type TEXT NOT NULL,
    created_at TEXT,
    modified_at TEXT,
    imported_at TEXT NOT NULL,
    note_date TEXT,
    date_label TEXT,
    date_confidence TEXT,
    parser_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    parser_name TEXT,
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS note_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(note_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id INTEGER NOT NULL REFERENCES note_chunks(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    vector_json TEXT,
    dimension INTEGER,
    text_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model_name)
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS note_categories (
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    confidence REAL,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (note_id, category_id)
);

CREATE TABLE IF NOT EXISTS note_summaries (
    note_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    generated_title TEXT,
    one_line_summary TEXT,
    detailed_summary TEXT,
    important_points_json TEXT,
    revisit_reason TEXT,
    confidence REAL,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT,
    event_type TEXT,
    event_date TEXT,
    date_label TEXT,
    date_confidence TEXT,
    importance REAL,
    confidence REAL,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT,
    thought_type TEXT,
    themes_json TEXT,
    emotion_label TEXT,
    emotion_intensity REAL,
    date_label TEXT,
    importance REAL,
    confidence REAL,
    remember_reason TEXT,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT REFERENCES notes(id) ON DELETE SET NULL,
    suggestion_type TEXT NOT NULL DEFAULT 'revisit_note',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    target_date TEXT,
    importance REAL,
    confidence REAL,
    evidence_json TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_reflections (
    month TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL,
    importance REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_timeline_snapshots (
    id TEXT PRIMARY KEY,
    month TEXT NOT NULL,
    title TEXT,
    overview TEXT,
    thought_summary TEXT,
    event_summary TEXT,
    important_changes_json TEXT,
    key_themes_json TEXT,
    dominant_categories_json TEXT,
    rediscovery_points_json TEXT,
    revisit_reasons_json TEXT,
    evidence_json TEXT,
    quality_json TEXT,
    source_counts_json TEXT,
    source_hash TEXT,
    model_name TEXT,
    generated_by TEXT,
    confidence REAL,
    importance REAL,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(month, source_hash)
);

CREATE TABLE IF NOT EXISTS monthly_timeline_items (
    id TEXT PRIMARY KEY,
    month TEXT NOT NULL,
    date_start TEXT,
    date_end TEXT,
    date_label TEXT,
    item_type TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    detail TEXT,
    themes_json TEXT,
    categories_json TEXT,
    emotion_json TEXT,
    evidence_json TEXT,
    source_table TEXT,
    source_id TEXT,
    source_note_id TEXT,
    confidence REAL,
    importance REAL,
    date_confidence REAL,
    date_source TEXT,
    date_quality TEXT,
    evidence_enriched INTEGER DEFAULT 0,
    quality_flags_json TEXT,
    grouped_item_ids_json TEXT,
    sub_items_json TEXT,
    sort_key TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    note_id TEXT,
    input_hash TEXT NOT NULL,
    body_hash TEXT,
    output_json TEXT,
    raw_output TEXT,
    success INTEGER NOT NULL,
    error_type TEXT,
    error_message TEXT,
    prompt_version TEXT,
    empty_result INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    fallback_used INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(task_name, model_name, input_hash)
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = database_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | Path | None = None) -> Path:
    db_path = database_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(CORE_TABLES_SQL)
        _ensure_columns(conn)
        _backfill_model_run_failures(conn)
        _ensure_fts_table(conn)
        _ensure_indexes(conn)
        _seed_categories(conn, load_categories())
    return db_path


def _ensure_fts_table(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                note_id UNINDEXED,
                title,
                body,
                source_path UNINDEXED,
                folder UNINDEXED,
                tokenize = 'trigram'
            )
            """
        )
    except sqlite3.OperationalError:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                note_id UNINDEXED,
                title,
                body,
                source_path UNINDEXED,
                folder UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )


def _seed_categories(conn: sqlite3.Connection, categories: list[str]) -> None:
    for name in categories:
        conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,))


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_note_chunks_note_id ON note_chunks(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model_name, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_note_id ON chunk_embeddings(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_event_date ON events(event_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_thoughts_date_label ON thoughts(date_label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_type_status ON suggestions(suggestion_type, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_note ON suggestions(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monthly_timeline_snapshots_month ON monthly_timeline_snapshots(month)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monthly_timeline_items_month_sort ON monthly_timeline_items(month, sort_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monthly_timeline_items_source_note ON monthly_timeline_items(source_note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_task_success_error ON model_runs(task_name, success, error_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_note_id ON model_runs(note_id)")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "note_summaries", "importance", "REAL")
    _add_column_if_missing(conn, "note_categories", "importance", "REAL")
    _add_column_if_missing(conn, "model_runs", "note_id", "TEXT")
    _add_column_if_missing(conn, "model_runs", "body_hash", "TEXT")
    _add_column_if_missing(conn, "model_runs", "raw_output", "TEXT")
    _add_column_if_missing(conn, "model_runs", "error_type", "TEXT")
    _add_column_if_missing(conn, "model_runs", "error_message", "TEXT")
    _add_column_if_missing(conn, "model_runs", "prompt_version", "TEXT")
    _add_column_if_missing(conn, "model_runs", "empty_result", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "model_runs", "retry_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "model_runs", "fallback_used", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "suggestions", "suggestion_type", "TEXT NOT NULL DEFAULT 'revisit_note'")
    _add_column_if_missing(conn, "suggestions", "target_date", "TEXT")
    _add_column_if_missing(conn, "suggestions", "status", "TEXT NOT NULL DEFAULT 'new'")
    _add_column_if_missing(conn, "monthly_timeline_items", "date_source", "TEXT")
    _add_column_if_missing(conn, "monthly_timeline_items", "date_quality", "TEXT")
    _add_column_if_missing(conn, "monthly_timeline_items", "evidence_enriched", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "monthly_timeline_items", "quality_flags_json", "TEXT")
    _add_column_if_missing(conn, "monthly_timeline_items", "grouped_item_ids_json", "TEXT")
    _add_column_if_missing(conn, "monthly_timeline_items", "sub_items_json", "TEXT")


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _backfill_model_run_failures(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE model_runs
        SET error_type = CASE
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%cuda%out of memory%' THEN 'cuda_oom'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%json%' THEN 'json_parse_error'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%expecting%' THEN 'json_parse_error'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%unterminated%' THEN 'truncated_output'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%schema%' THEN 'schema_validation_error'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%validation%' THEN 'schema_validation_error'
            WHEN LOWER(COALESCE(error_message, '')) LIKE '%model%' THEN 'model_load_error'
            ELSE 'legacy_unknown_failure'
        END
        WHERE success = 0
          AND (error_type IS NULL OR error_type = '' OR error_type = 'legacy_unknown_failure')
          AND error_message IS NOT NULL
          AND error_message != ''
          AND error_message != 'Failure was recorded before error diagnostics were added.'
        """
    )
    conn.execute(
        """
        UPDATE model_runs
        SET
            error_type = COALESCE(error_type, 'legacy_unknown_failure'),
            error_message = COALESCE(
                error_message,
                'Failure was recorded before error diagnostics were added.'
            )
        WHERE success = 0
          AND (error_type IS NULL OR error_type = '' OR error_message IS NULL OR error_message = '')
        """
    )


def table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])
