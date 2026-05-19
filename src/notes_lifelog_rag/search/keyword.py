from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from notes_lifelog_rag.db.schema import connect, init_db


@dataclass(frozen=True)
class SearchResult:
    note_id: str
    title: str
    source_relative_path: str
    folder: str
    snippet: str
    score: float
    source: str


def search_notes(query: str, limit: int = 10, db_path: str | Path | None = None) -> list[SearchResult]:
    if not query.strip():
        return []
    init_db(db_path)
    with connect(db_path) as conn:
        results: dict[str, SearchResult] = {}
        for result in _fts_results(conn, query, max(limit * 2, 10)):
            results[result.note_id] = result
        for result in _like_results(conn, query, max(limit * 2, 10)):
            previous = results.get(result.note_id)
            if previous is None or result.score > previous.score:
                results[result.note_id] = result
        return sorted(results.values(), key=lambda item: item.score, reverse=True)[:limit]


def _fts_results(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchResult]:
    phrase = _fts_phrase(query)
    try:
        rows = conn.execute(
            """
            SELECT
                notes.id AS note_id,
                notes.title AS title,
                notes.source_relative_path AS source_relative_path,
                COALESCE(notes.folder, '') AS folder,
                snippet(notes_fts, 2, '', '', '...', 16) AS snippet,
                bm25(notes_fts) AS rank
            FROM notes_fts
            JOIN notes ON notes.id = notes_fts.note_id
            WHERE notes_fts MATCH ?
            LIMIT ?
            """,
            (phrase, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        SearchResult(
            note_id=row["note_id"],
            title=row["title"],
            source_relative_path=row["source_relative_path"],
            folder=row["folder"],
            snippet=_clean_snippet(row["snippet"]),
            score=max(0.0, 1.0 - float(row["rank"])),
            source="fts5",
        )
        for row in rows
    ]


def _like_results(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchResult]:
    like = f"%{query}%"
    rows = conn.execute(
        """
        SELECT id, title, body, source_relative_path, COALESCE(folder, '') AS folder
        FROM notes
        WHERE title LIKE ? OR body LIKE ? OR source_relative_path LIKE ?
        ORDER BY modified_at DESC
        LIMIT ?
        """,
        (like, like, like, limit),
    ).fetchall()
    results: list[SearchResult] = []
    for row in rows:
        score = 0.75
        if query in row["title"]:
            score += 0.15
        if query in row["source_relative_path"]:
            score += 0.05
        results.append(
            SearchResult(
                note_id=row["id"],
                title=row["title"],
                source_relative_path=row["source_relative_path"],
                folder=row["folder"],
                snippet=make_snippet(row["body"], query),
                score=score,
                source="like",
            )
        )
    return results


def make_snippet(text: str, query: str, max_chars: int = 120) -> str:
    collapsed = _clean_snippet(text)
    if not collapsed:
        return ""
    index = collapsed.casefold().find(query.casefold())
    if index == -1:
        return _truncate(collapsed, max_chars)
    radius = max_chars // 2
    start = max(0, index - radius)
    end = min(len(collapsed), index + len(query) + radius)
    snippet = collapsed[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(collapsed):
        snippet = snippet + "..."
    return snippet


def _fts_phrase(query: str) -> str:
    escaped = query.replace('"', '""').strip()
    return f'"{escaped}"'


def _clean_snippet(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

