from pathlib import Path
import hashlib

from notes_lifelog_rag.analysis.service import analyze_all, extract_events, extract_thoughts, summarize_notes
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.llm.mock import MockLLMBackend
from notes_lifelog_rag.runtime.device import resolve_device


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"
ANALYSIS_TABLES = ["model_runs", "note_summaries", "note_categories", "events", "thoughts"]


def test_mock_analysis_populates_phase4_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    summaries = analyze_all(db_path=db_path, limit=2, backend_name="mock")

    assert [summary.task_name for summary in summaries] == ["summary", "categories", "events", "thoughts"]
    assert all(summary.failed_notes == 0 for summary in summaries)
    with connect(db_path) as conn:
        assert table_count(conn, "note_summaries") == 2
        assert table_count(conn, "model_runs") >= 8
        assert table_count(conn, "note_categories") >= 1
        assert table_count(conn, "events") >= 1


def test_summary_only_missing_and_dry_run_are_resume_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    with connect(db_path) as conn:
        note_count = table_count(conn, "notes")

    first = summarize_notes(db_path=db_path, all_notes=True, backend_name="mock")
    assert first.processed_notes == note_count

    second = summarize_notes(db_path=db_path, all_notes=True, backend_name="mock")
    assert second.processed_notes == 0
    assert second.skipped_existing == note_count

    with connect(db_path) as conn:
        summary_count = table_count(conn, "note_summaries")
        model_run_count = table_count(conn, "model_runs")

    dry_run = summarize_notes(db_path=db_path, all_notes=True, force=True, dry_run=True, backend_name="mock")
    assert dry_run.dry_run is True
    assert dry_run.would_process_notes == note_count
    assert dry_run.processed_notes == 0

    with connect(db_path) as conn:
        assert table_count(conn, "note_summaries") == summary_count
        assert table_count(conn, "model_runs") == model_run_count


def test_model_runs_cache_restores_missing_event_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    with connect(db_path) as conn:
        note_count = table_count(conn, "notes")

    first = extract_events(db_path=db_path, all_notes=True, backend_name="mock")
    assert first.processed_notes == note_count

    with connect(db_path) as conn:
        event_count = table_count(conn, "events")
        model_run_count = table_count(conn, "model_runs")
        note_id = conn.execute("SELECT note_id FROM events LIMIT 1").fetchone()["note_id"]
        conn.execute("DELETE FROM events WHERE note_id = ?", (note_id,))
        conn.commit()

    resumed = extract_events(db_path=db_path, all_notes=True, backend_name="mock")
    assert resumed.processed_notes == 1
    assert resumed.cached_notes == 1
    assert resumed.skipped_existing == note_count - 1

    with connect(db_path) as conn:
        assert table_count(conn, "events") == event_count
        assert table_count(conn, "model_runs") == model_run_count


def test_empty_cached_analysis_outputs_do_not_block_only_missing_batches(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    with connect(db_path) as conn:
        note_count = table_count(conn, "notes")

    first = extract_thoughts(db_path=db_path, all_notes=True, backend_name="mock")
    assert first.processed_notes == note_count

    second = extract_thoughts(db_path=db_path, all_notes=True, backend_name="mock")
    assert second.processed_notes == 0
    assert second.skipped_existing + second.skipped_cached_empty == note_count


def test_cached_empty_events_are_skipped_as_processed(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    _insert_notes(db_path, 1, body_prefix="")

    first = extract_events(db_path=db_path, backend_name="mock")
    assert first.processed_notes == 1
    assert first.created_items == 0

    second = extract_events(db_path=db_path, backend_name="mock")
    assert second.processed_notes == 0
    assert second.skipped_cached_empty == 1
    with connect(db_path) as conn:
        row = conn.execute("SELECT output_json, success, empty_result FROM model_runs").fetchone()
        assert row["success"] == 1
        assert row["empty_result"] == 1
        assert row["output_json"] == '{"events": []}'


def test_analysis_dry_run_does_not_generate(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    class RaisingBackend(MockLLMBackend):
        def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
            raise AssertionError("dry-run should not call generate_json")

    monkeypatch.setattr("notes_lifelog_rag.analysis.service.get_llm_backend", lambda *args, **kwargs: RaisingBackend())
    summary = summarize_notes(db_path=db_path, limit=1, dry_run=True, backend_name="local")
    assert summary.would_process_notes == 1
    assert summary.processed_notes == 0


def test_analysis_cache_hit_does_not_generate_again(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    first_backend = MockLLMBackend()
    monkeypatch.setattr("notes_lifelog_rag.analysis.service.get_llm_backend", lambda *args, **kwargs: first_backend)
    first = summarize_notes(db_path=db_path, limit=1, backend_name="local")
    assert first.processed_notes == 1
    with connect(db_path) as conn:
        note_id = conn.execute("SELECT note_id FROM note_summaries LIMIT 1").fetchone()["note_id"]
        conn.execute("DELETE FROM note_summaries WHERE note_id = ?", (note_id,))
        conn.commit()

    class RaisingBackend(MockLLMBackend):
        def generate_json(self, task_name: str, note: dict, *, categories: list[str] | None = None) -> dict:
            raise AssertionError("cache hit should not call generate_json")

    monkeypatch.setattr("notes_lifelog_rag.analysis.service.get_llm_backend", lambda *args, **kwargs: RaisingBackend())
    second = summarize_notes(db_path=db_path, limit=1, backend_name="local")
    assert second.cached_notes == 1
    assert second.processed_notes == 1


def test_analysis_passes_device_info_to_local_backend(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    device_info = resolve_device("cpu")
    captured = {}

    def fake_get_llm_backend(*args, **kwargs):
        captured["device_info"] = kwargs.get("device_info")
        return MockLLMBackend()

    monkeypatch.setattr("notes_lifelog_rag.analysis.service.get_llm_backend", fake_get_llm_backend)
    summarize_notes(db_path=db_path, limit=1, backend_name="local", device_info=device_info)
    assert captured["device_info"].resolved_device == "cpu"


def test_default_limit_none_selects_all_eligible_notes(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    _insert_notes(db_path, 12)

    summary = summarize_notes(db_path=db_path, dry_run=True, backend_name="mock")

    assert summary.total_notes == 12
    assert summary.scanned_notes == 12
    assert summary.eligible_notes == 12
    assert summary.selected_notes == 12


def test_explicit_limit_restricts_selected_notes(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    _insert_notes(db_path, 12)

    summary = summarize_notes(db_path=db_path, limit=10, dry_run=True, backend_name="mock")

    assert summary.total_notes == 12
    assert summary.eligible_notes == 12
    assert summary.selected_notes == 10


def test_analyze_all_dry_run_does_not_modify_analysis_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    before = _table_counts(db_path, ANALYSIS_TABLES)
    summaries = analyze_all(db_path=db_path, dry_run=True, backend_name="mock")
    after = _table_counts(db_path, ANALYSIS_TABLES)

    assert after == before
    assert all(summary.processed_notes == 0 for summary in summaries)
    assert all(summary.created_items == 0 for summary in summaries)


def test_dry_run_does_not_insert_model_runs_or_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    summarize_notes(db_path=db_path, dry_run=True, backend_name="mock")

    counts = _table_counts(db_path, ANALYSIS_TABLES)
    assert counts == {name: 0 for name in ANALYSIS_TABLES}


def test_body_hash_change_makes_existing_summary_eligible_again(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    first = summarize_notes(db_path=db_path, limit=1, backend_name="mock")
    assert first.processed_notes == 1

    with connect(db_path) as conn:
        note_id = conn.execute("SELECT note_id FROM note_summaries LIMIT 1").fetchone()["note_id"]
        new_body = "本文が更新されたメモです。"
        conn.execute(
            "UPDATE notes SET body = ?, content_hash = ? WHERE id = ?",
            (new_body, hashlib.sha256(new_body.encode("utf-8")).hexdigest(), note_id),
        )
        conn.commit()

    dry_run = summarize_notes(db_path=db_path, dry_run=True, backend_name="mock")

    assert note_id in (dry_run.would_process_note_ids or [])


def _table_counts(db_path: Path, table_names: list[str]) -> dict[str, int]:
    with connect(db_path) as conn:
        return {name: table_count(conn, name) for name in table_names}


def _insert_notes(db_path: Path, count: int, *, body_prefix: str = "テスト本文 ") -> None:
    with connect(db_path) as conn:
        for index in range(count):
            body = f"{body_prefix}{index}" if body_prefix else ""
            note_id = f"note-{index:03d}"
            conn.execute(
                """
                INSERT INTO notes(
                    id, content_hash, title, body, source_path, source_relative_path,
                    folder, file_type, imported_at, parser_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note_id,
                    hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    f"テストメモ {index}",
                    body,
                    f"/tmp/test-{index}.md",
                    f"test-{index}.md",
                    "tests",
                    ".md",
                    "2026-01-01T00:00:00+00:00",
                    "test",
                ),
            )
        conn.commit()
