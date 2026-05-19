from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Callable

from notes_lifelog_rag.db.schema import connect, init_db
from notes_lifelog_rag.embeddings.engines import EmbeddingBackend
from notes_lifelog_rag.embeddings.vector import cosine_similarity, text_hash, vector_from_json, vector_to_json
from notes_lifelog_rag.search.keyword import make_snippet

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class BuildEmbeddingSummary:
    backend: str
    model_name: str
    scanned_chunks: int = 0
    selected_chunks: int = 0
    would_embed_chunks: int = 0
    embedded_chunks: int = 0
    skipped_existing: int = 0
    failed_chunks: int = 0
    dry_run: bool = False
    disabled_reason: str | None = None


@dataclass(frozen=True)
class VectorSearchResult:
    note_id: str
    chunk_id: int
    title: str
    source_relative_path: str
    folder: str
    snippet: str
    score: float
    model_name: str


def build_chunk_embeddings(
    backend: EmbeddingBackend,
    *,
    db_path: str | Path | None = None,
    limit: int | None = None,
    force: bool = False,
    only_missing: bool = True,
    dry_run: bool = False,
    batch_size: int = 16,
    progress_callback: ProgressCallback | None = None,
) -> BuildEmbeddingSummary:
    init_db(db_path)
    summary = BuildEmbeddingSummary(
        backend=backend.__class__.__name__,
        model_name=backend.model_name,
        dry_run=dry_run,
    )
    if not backend.is_available() and not dry_run:
        summary.disabled_reason = backend.availability_error()
        return summary
    if not backend.is_available():
        summary.disabled_reason = backend.availability_error()

    with connect(db_path) as conn:
        chunks, scanned_chunks, skipped_existing = _chunks_to_embed(
            conn,
            backend.model_name,
            limit=limit,
            force=force,
            only_missing=only_missing,
        )
        summary.scanned_chunks = scanned_chunks
        summary.selected_chunks = len(chunks)
        summary.skipped_existing = skipped_existing
        summary.would_embed_chunks = len(chunks)
        if progress_callback:
            progress_callback(0, len(chunks), "embedding chunks")
        if not chunks:
            return summary
        if dry_run:
            if progress_callback:
                progress_callback(len(chunks), len(chunks), "embedding chunks")
            return summary
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            results = backend.embed_texts([row["text"] for row in batch])
            for row, result in zip(batch, results):
                if result.status == "success" and result.vector:
                    summary.embedded_chunks += 1
                else:
                    summary.failed_chunks += 1
                _upsert_embedding(
                    conn,
                    row,
                    result.status,
                    result.error_message,
                    result.dimension,
                    result.vector,
                    backend.model_name,
                )
            conn.commit()
            if progress_callback:
                done = min(start + len(batch), len(chunks))
                progress_callback(done, len(chunks), "embedding chunks")
    return summary


def vector_search(
    query: str,
    backend: EmbeddingBackend,
    *,
    db_path: str | Path | None = None,
    model_name: str | None = None,
    limit: int = 10,
) -> list[VectorSearchResult]:
    if not query.strip() or not backend.is_available():
        return []
    init_db(db_path)
    query_result = backend.embed_texts([query])[0]
    if query_result.status != "success" or not query_result.vector:
        return []
    target_model = model_name or query_result.model_name
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                chunk_embeddings.chunk_id,
                chunk_embeddings.note_id,
                chunk_embeddings.vector_json,
                chunk_embeddings.model_name,
                note_chunks.text,
                notes.title,
                notes.source_relative_path,
                COALESCE(notes.folder, '') AS folder
            FROM chunk_embeddings
            JOIN note_chunks ON note_chunks.id = chunk_embeddings.chunk_id
            JOIN notes ON notes.id = chunk_embeddings.note_id
            WHERE chunk_embeddings.status = 'success'
              AND chunk_embeddings.model_name = ?
              AND chunk_embeddings.vector_json IS NOT NULL
            """,
            (target_model,),
        ).fetchall()
    scored: list[VectorSearchResult] = []
    for row in rows:
        score = cosine_similarity(query_result.vector, vector_from_json(row["vector_json"]))
        if score <= 0:
            continue
        scored.append(
            VectorSearchResult(
                note_id=row["note_id"],
                chunk_id=int(row["chunk_id"]),
                title=row["title"],
                source_relative_path=row["source_relative_path"],
                folder=row["folder"],
                snippet=make_snippet(row["text"], query),
                score=score,
                model_name=row["model_name"],
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def best_embedding_model_in_db(db_path: str | Path | None = None, *, include_mock: bool = False) -> str | None:
    init_db(db_path)
    mock_clause = "" if include_mock else "AND model_name NOT LIKE 'mock-%'"
    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT model_name, COUNT(*) AS count
            FROM chunk_embeddings
            WHERE status = 'success'
              {mock_clause}
            GROUP BY model_name
            ORDER BY count DESC, model_name ASC
            LIMIT 1
            """
        ).fetchone()
    return str(row["model_name"]) if row else None


def _chunks_to_embed(
    conn: sqlite3.Connection,
    model_name: str,
    *,
    limit: int | None,
    force: bool,
    only_missing: bool,
) -> tuple[list[sqlite3.Row], int, int]:
    rows = conn.execute("SELECT id, note_id, text FROM note_chunks ORDER BY id").fetchall()
    selected: list[sqlite3.Row] = []
    scanned = 0
    skipped_existing = 0
    for row in rows:
        scanned += 1
        if only_missing and not force:
            if _successful_embedding_exists(conn, row, model_name):
                skipped_existing += 1
                continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected, scanned, skipped_existing


def _successful_embedding_exists(conn: sqlite3.Connection, row: sqlite3.Row, model_name: str) -> bool:
    existing = conn.execute(
        """
        SELECT 1 FROM chunk_embeddings
        WHERE chunk_id = ? AND model_name = ? AND text_hash = ? AND status = 'success'
        LIMIT 1
        """,
        (row["id"], model_name, text_hash(row["text"])),
    ).fetchone()
    return existing is not None


def _upsert_embedding(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    status: str,
    error_message: str | None,
    dimension: int,
    vector: list[float],
    model_name: str,
) -> None:
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO chunk_embeddings(
            chunk_id, note_id, model_name, vector_json, dimension, text_hash,
            status, error_message, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id, model_name) DO UPDATE SET
            vector_json = excluded.vector_json,
            dimension = excluded.dimension,
            text_hash = excluded.text_hash,
            status = excluded.status,
            error_message = excluded.error_message,
            updated_at = excluded.updated_at
        """,
        (
            row["id"],
            row["note_id"],
            model_name,
            vector_to_json(vector) if vector else None,
            dimension,
            text_hash(row["text"]),
            status,
            error_message,
            now,
            now,
        ),
    )
