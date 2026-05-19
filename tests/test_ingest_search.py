from pathlib import Path

from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.search.keyword import search_notes


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"


def test_ingest_is_idempotent_and_searchable(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)

    first = ingest_directory(FIXTURES, db_path)
    second = ingest_directory(FIXTURES, db_path)

    assert first.imported_notes == 4
    assert first.parser_errors == 0
    assert second.imported_notes == 0
    assert second.skipped_duplicates == 4

    with connect(db_path) as conn:
        assert table_count(conn, "notes") == 4
        assert table_count(conn, "note_chunks") >= 4
        assert table_count(conn, "chunk_embeddings") == 0
        assert table_count(conn, "categories") >= 1

    results = search_notes("研究", db_path=db_path)
    assert results
    assert results[0].title == "研究メモ"
    assert "研究" in results[0].snippet
