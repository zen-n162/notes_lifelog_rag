import json
from pathlib import Path

from typer.testing import CliRunner

from notes_lifelog_rag.cli import app
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.timeline.review import (
    list_review_actions,
    reanalysis_plan_rows,
    revert_review_action,
    upsert_review_action,
)
from notes_lifelog_rag.timeline.service import (
    format_timeline_qa_markdown,
    generate_month_timeline_snapshot,
    get_month_timeline_snapshot,
    timeline_qa,
)
from notes_lifelog_rag.ui import services


RUNNER = CliRunner()


def test_create_review_action_updates_duplicate_instead_of_inserting(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "review-note", "研究メモ")
    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    item_id = snapshot.items[0].id

    first = upsert_review_action(action_type="hide", item_id=item_id, reason="noisy", db_path=db_path)
    second = upsert_review_action(action_type="hide", item_id=item_id, reason="still noisy", db_path=db_path)

    assert first.id == second.id
    assert second.reason == "still noisy"
    with connect(db_path) as conn:
        assert table_count(conn, "timeline_review_actions") == 1


def test_hide_item_removes_default_timeline_and_include_hidden_restores(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "hide-note", "スキャンPDF", source_path="scan.pdf", summary="｜｜｜ ノイズ断片")
    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    item_id = snapshot.items[0].id
    upsert_review_action(action_type="hide", item_id=item_id, reason="noisy pdf", db_path=db_path)

    default_snapshot = get_month_timeline_snapshot("2026-05", db_path=db_path)
    hidden_snapshot = get_month_timeline_snapshot("2026-05", db_path=db_path, include_hidden=True)

    assert default_snapshot is not None and not any(item.id == item_id for item in default_snapshot.items)
    assert hidden_snapshot is not None and any(item.id == item_id for item in hidden_snapshot.items)


def test_verify_and_dismiss_reduce_qa_problem_impact(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "weak-note", "弱い根拠", summary="短い")
    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    item_id = snapshot.items[0].id

    before = timeline_qa(month="2026-05", db_path=db_path, show_items=True)[0]
    upsert_review_action(action_type="verify", item_id=item_id, reason="human checked", db_path=db_path)
    upsert_review_action(action_type="dismiss_warning", item_id=item_id, reason="low_value_items_present", db_path=db_path)
    after = timeline_qa(month="2026-05", db_path=db_path, show_items=True)[0]

    assert before["problem_items"]
    assert after["verified_items_count"] == 1
    assert len(after["review_warnings"]) <= len(before["review_warnings"])


def test_exclude_source_note_removes_items_from_generation_and_revert_restores(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "exclude-note", "Scanner noise", source_path="scan.pdf", summary="｜｜｜ ノイズ断片")
    upsert_review_action(action_type="exclude_source_note", source_note_id="exclude-note", reason="scanner noise", db_path=db_path)

    excluded = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    action = list_review_actions(db_path=db_path)[0]
    revert_review_action(action.id, db_path=db_path)
    restored = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)

    assert not any(item.source_note_id == "exclude-note" for item in excluded.items)
    assert any(item.source_note_id == "exclude-note" for item in restored.items)


def test_timeline_qa_markdown_includes_review_commands_and_json_review_fields(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "cmd-note", "スキャンPDF", source_path="scan.pdf", summary="｜｜｜ ノイズ断片")
    generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    rows = timeline_qa(month="2026-05", db_path=db_path, show_items=True)
    markdown = format_timeline_qa_markdown(rows)

    assert "timeline-review hide --item-id" in markdown
    assert "item_id:" in markdown
    first_problem = next(iter(rows[0]["problem_items"].values()))[0]
    assert first_problem["item_id"]
    assert first_problem["source_note_id"] == "cmd-note"
    assert first_problem["quality_flags"]
    assert "review_actions_available" in first_problem


def test_timeline_review_cli_and_reanalysis_plan(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "fix-note", "根拠が弱いメモ", summary="短い")
    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    item_id = snapshot.items[0].id

    result = RUNNER.invoke(
        app,
        ["timeline-review", "needs-fix", "--item-id", item_id, "--reason", "weak evidence", "--db", str(db_path)],
    )
    plan_result = RUNNER.invoke(
        app,
        ["timeline-review", "reanalysis-plan", "--month", "2026-05", "--db", str(db_path)],
    )
    rows = reanalysis_plan_rows(month="2026-05", db_path=db_path)

    assert result.exit_code == 0, result.output
    assert plan_result.exit_code == 0, plan_result.output
    assert rows and rows[0]["note_id"] == "fix-note"
    assert "python -m notes_lifelog_rag.cli" in plan_result.output


def test_ui_service_returns_review_status(tmp_path: Path) -> None:
    db_path = _db_with_summary(tmp_path, "ui-note", "UI review note")
    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    action = services.save_timeline_review_action(
        action_type="verify",
        item_id=snapshot.items[0].id,
        source_note_id="ui-note",
        month="2026-05",
        comment="important",
        db_path=db_path,
    )

    actions = services.timeline_review_actions("2026-05", db_path=db_path)
    assert action["status"] == "active"
    assert actions and actions[0]["action_type"] == "verify"


def _db_with_summary(
    tmp_path: Path,
    note_id: str,
    title: str,
    *,
    source_path: str = "src.md",
    summary: str = "月面探査と機械学習に関する研究方針を整理した。",
) -> Path:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO notes(
                id, content_hash, title, body, source_path, source_relative_path, file_type,
                created_at, modified_at, imported_at, note_date, parser_name
            )
            VALUES (?, ?, ?, ?, ?, ?, 'md', '2026-05-01', '2026-05-01', '2026-05-01', '2026-05-01', 'test')
            """,
            (note_id, f"hash-{note_id}", title, f"Body for {title}. {summary}", source_path, source_path),
        )
        conn.execute(
            """
            INSERT INTO note_summaries(
                note_id, model_name, generated_title, one_line_summary, detailed_summary,
                important_points_json, revisit_reason, confidence, importance, evidence_json, created_at
            )
            VALUES (?, 'mock', ?, ?, '', '[]', 'あとで見返す', 0.4, 0.8, ?, '2026-05-01')
            """,
            (note_id, title, summary, json.dumps([{"note_id": note_id, "quote": summary[:60]}])),
        )
        conn.commit()
    return db_path
