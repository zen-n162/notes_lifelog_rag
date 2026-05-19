from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from notes_lifelog_rag.config import raw_notes_path
from notes_lifelog_rag.db.schema import connect, init_db
from notes_lifelog_rag.ingest.parsers import SUPPORTED_EXTENSIONS, ParsedNote, ParserError, parse_note_file
from notes_lifelog_rag.utils.dates import DateParseResult, parse_date_label


@dataclass
class IngestSummary:
    scanned_files: int = 0
    imported_notes: int = 0
    skipped_duplicates: int = 0
    skipped_unsupported: int = 0
    parser_errors: int = 0
    imported_ids: list[str] = field(default_factory=list)


def ingest_directory(input_path: str | Path | None = None, db_path: str | Path | None = None) -> IngestSummary:
    root = raw_notes_path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    init_db(db_path)
    summary = IngestSummary()
    with connect(db_path) as conn:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            summary.scanned_files += 1
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                summary.skipped_unsupported += 1
                continue
            try:
                parsed = parse_note_file(path)
                if not parsed.body.strip():
                    raise ParserError("Parsed note body is empty.")
                inserted_id = _insert_note(conn, root, path, parsed)
            except ParserError as exc:
                _record_import_error(conn, path, _parser_name_for(path), str(exc))
                summary.parser_errors += 1
                continue
            if inserted_id is None:
                summary.skipped_duplicates += 1
            else:
                summary.imported_notes += 1
                summary.imported_ids.append(inserted_id)
    return summary


def _insert_note(conn: sqlite3.Connection, root: Path, path: Path, parsed: ParsedNote) -> str | None:
    now = _utc_now()
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    source_relative_path = _relative_to(path, root)
    content_hash = _content_hash(parsed)
    note_id = content_hash
    date_result = parse_date_label(f"{parsed.title} {path.stem}", context_date=modified_at[:10])
    try:
        conn.execute(
            """
            INSERT INTO notes (
                id, content_hash, title, body, source_path, source_relative_path, folder,
                file_type, created_at, modified_at, imported_at, note_date, date_label,
                date_confidence, parser_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                content_hash,
                parsed.title,
                parsed.body,
                str(path.resolve()),
                source_relative_path,
                str(Path(source_relative_path).parent),
                path.suffix.lower(),
                created_at,
                modified_at,
                now,
                date_result.iso_date,
                date_result.date_label,
                date_result.confidence,
                parsed.parser_name,
            ),
        )
    except sqlite3.IntegrityError:
        return None

    conn.execute(
        "INSERT INTO notes_fts(note_id, title, body, source_path, folder) VALUES (?, ?, ?, ?, ?)",
        (note_id, parsed.title, parsed.body, source_relative_path, str(Path(source_relative_path).parent)),
    )
    _insert_chunks(conn, note_id, parsed.body, now)
    return note_id


def _insert_chunks(conn: sqlite3.Connection, note_id: str, body: str, created_at: str) -> None:
    chunk_size = 1200
    overlap = 200
    start = 0
    chunk_index = 0
    body_len = len(body)
    while start < body_len:
        end = min(start + chunk_size, body_len)
        text = body[start:end].strip()
        if text:
            conn.execute(
                """
                INSERT OR IGNORE INTO note_chunks(note_id, chunk_index, text, start_char, end_char, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (note_id, chunk_index, text, start, end, created_at),
            )
            chunk_index += 1
        if end == body_len:
            break
        start = max(end - overlap, start + 1)


def _record_import_error(
    conn: sqlite3.Connection, path: Path, parser_name: str | None, error_message: str
) -> None:
    existing = conn.execute(
        """
        SELECT 1
        FROM import_errors
        WHERE source_path = ? AND error_message = ?
        LIMIT 1
        """,
        (str(path.resolve()), error_message),
    ).fetchone()
    if existing:
        return
    conn.execute(
        """
        INSERT INTO import_errors(source_path, parser_name, error_message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (str(path.resolve()), parser_name, error_message, _utc_now()),
    )


def _parser_name_for(path: Path) -> str | None:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or None


def _content_hash(parsed: ParsedNote) -> str:
    digest = hashlib.sha256()
    digest.update(parsed.title.strip().encode("utf-8"))
    digest.update(b"\n\n")
    digest.update(parsed.body.strip().encode("utf-8"))
    return digest.hexdigest()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
