from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from notes_lifelog_rag.analysis.service import summarize_notes
from notes_lifelog_rag.cli import app
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.llm.json_utils import LenientJSONError, classify_json_error, parse_json_lenient, strip_code_fences


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"
runner = CliRunner()


def test_model_runs_migration_adds_error_columns_and_backfills_legacy_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(model_runs)").fetchall()}
        for column in [
            "note_id",
            "error_type",
            "error_message",
            "raw_output",
            "prompt_version",
            "body_hash",
            "empty_result",
            "retry_count",
            "fallback_used",
        ]:
            assert column in columns
        conn.execute(
            """
            INSERT INTO model_runs(task_name, model_name, input_hash, success, created_at)
            VALUES ('summary', 'legacy', 'hash-legacy', 0, '2026-01-01')
            """
        )
        conn.commit()

    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT error_type, error_message FROM model_runs WHERE input_hash = 'hash-legacy'").fetchone()
        assert row["error_type"] == "legacy_unknown_failure"
        assert "before error diagnostics" in row["error_message"]


def test_failed_analysis_records_error_type_and_raw_output(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    class BrokenBackend:
        model_name = "broken-local"

        def is_available(self) -> bool:
            return True

        def availability_error(self) -> str | None:
            return None

        def generate_with_raw(self, task_name: str, note: dict, *, categories: list[str] | None = None):
            raise LenientJSONError("json_parse_error", "could not parse JSON", raw_output="```json\n{\"bad\":\n```")

    monkeypatch.setattr("notes_lifelog_rag.analysis.service.get_llm_backend", lambda *args, **kwargs: BrokenBackend())
    summary = summarize_notes(db_path=db_path, limit=1, backend_name="local")
    assert summary.failed_notes == 1

    with connect(db_path) as conn:
        row = conn.execute("SELECT success, error_type, raw_output, note_id, body_hash FROM model_runs").fetchone()
        assert row["success"] == 0
        assert row["error_type"] == "json_parse_error"
        assert "bad" in row["raw_output"]
        assert row["note_id"]
        assert row["body_hash"]


def test_analysis_failures_and_db_schema_cli(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_runs(task_name, model_name, input_hash, success, error_type, error_message, created_at)
            VALUES ('events', 'mock', 'failure-hash', 0, 'empty_output', 'empty', '2026-01-01')
            """
        )
        conn.commit()

    result = runner.invoke(app, ["analysis-failures", "--db", str(db_path), "--group-by-error"])
    assert result.exit_code == 0
    assert "empty_output" in result.output

    result = runner.invoke(app, ["analysis-failures", "--db", str(db_path), "--task", "events", "--limit", "20"])
    assert result.exit_code == 0
    assert "empty_output" in result.output
    assert "events" in result.output

    result = runner.invoke(app, ["db-schema", "--db", str(db_path), "--table", "model_runs"])
    assert result.exit_code == 0
    assert "error_type" in result.output


def test_json_utils_handle_fences_and_malformed_output() -> None:
    assert strip_code_fences("```json\n{\"ok\": true}\n```") == '{"ok": true}'
    assert parse_json_lenient("prefix {\"ok\": true,} suffix") == {"ok": True}
    assert classify_json_error("", None) == "empty_output"
    assert classify_json_error('{"unfinished": ', None) == "truncated_output"


def test_analyze_sample_does_not_save_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    result = runner.invoke(
        app,
        ["analyze-sample", "--task", "summary", "--backend", "mock", "--device", "cpu", "--db", str(db_path)],
    )
    assert result.exit_code == 0
    assert "Analyze Sample" in result.output
    with connect(db_path) as conn:
        assert table_count(conn, "model_runs") == 0
        assert table_count(conn, "note_summaries") == 0
