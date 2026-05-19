from pathlib import Path

from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.embeddings.engines import get_embedding_backend
from notes_lifelog_rag.embeddings.repository import build_chunk_embeddings
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.search.hybrid import hybrid_search_notes


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"


def test_build_mock_embeddings_and_hybrid_search(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    backend = get_embedding_backend("mock")
    summary = build_chunk_embeddings(backend, db_path=db_path)

    assert summary.embedded_chunks >= 4
    with connect(db_path) as conn:
        assert table_count(conn, "chunk_embeddings") >= 4
        embedding_count = table_count(conn, "chunk_embeddings")

    skipped = build_chunk_embeddings(backend, db_path=db_path, limit=2)
    assert skipped.embedded_chunks == 0
    assert skipped.selected_chunks == 0
    assert skipped.skipped_existing >= 4

    dry_run = build_chunk_embeddings(backend, db_path=db_path, limit=1, force=True, dry_run=True)
    assert dry_run.dry_run is True
    assert dry_run.would_embed_chunks == 1
    assert dry_run.embedded_chunks == 0
    with connect(db_path) as conn:
        assert table_count(conn, "chunk_embeddings") == embedding_count

    results = hybrid_search_notes(
        "月面探査 研究",
        db_path=db_path,
        embedding_backend="mock",
        reranker_backend="mock",
    )
    assert results
    assert results[0].title == "研究メモ"
    assert "embedding" in results[0].source or "rerank" in results[0].source


def test_embedding_dry_run_does_not_call_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    class RaisingEmbeddingBackend:
        model_name = "raising-embedding"

        def is_available(self) -> bool:
            return True

        def availability_error(self) -> str | None:
            return None

        def embed_texts(self, texts: list[str]):
            raise AssertionError("dry-run should not call embed_texts")

    summary = build_chunk_embeddings(RaisingEmbeddingBackend(), db_path=db_path, limit=1, dry_run=True)
    assert summary.dry_run is True
    assert summary.would_embed_chunks == 1
    assert summary.embedded_chunks == 0
