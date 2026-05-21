from pathlib import Path

from notes_lifelog_rag.analysis.service import analyze_all
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.timeline.service import (
    build_monthly_reflection,
    build_month_timeline_items,
    build_timeline,
    format_month_timeline_markdown,
    format_reflection_markdown,
    format_timeline_report,
    format_timeline_markdown,
    generate_month_timeline_snapshot,
    generate_timeline_snapshots,
    get_month_sources,
    list_timeline_months,
    timeline_qa,
)


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"


def test_timeline_and_reflection_include_evidence_confidence_importance(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=2, backend_name="mock")

    items = build_timeline(db_path=db_path)
    assert items
    assert items[0].evidence
    assert items[0].confidence >= 0
    assert items[0].importance >= 0

    timeline_md = format_timeline_markdown(items)
    assert "confidence" in timeline_md
    assert "importance" in timeline_md
    assert "evidence" in timeline_md

    report = build_monthly_reflection(db_path=db_path)
    reflection_md = format_reflection_markdown(report)
    assert report.evidence
    assert "coverage" in report.to_dict()
    assert isinstance(report.quality_warnings, list)
    assert "confidence" in reflection_md
    assert "importance" in reflection_md
    assert "Evidence" in reflection_md

    with connect(db_path) as conn:
        assert table_count(conn, "monthly_reflections") == 1

    build_monthly_reflection(db_path=db_path, force=True)
    with connect(db_path) as conn:
        assert table_count(conn, "monthly_reflections") == 1


def test_month_timeline_snapshot_rule_backend_and_quality(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=3, backend_name="mock")

    months = list_timeline_months(db_path=db_path, order="asc")
    assert months == sorted(months, key=lambda item: item.month)
    month = months[0].month
    sources = get_month_sources(month, db_path=db_path)
    assert sources["notes"]
    assert "events" in sources and "thoughts" in sources and "summaries" in sources

    items = build_month_timeline_items(month, db_path=db_path)
    assert items
    assert any(item.item_type in {"thought", "event", "note_summary"} for item in items)
    assert all("body" not in item.to_dict() for item in items)

    dry_snapshot = generate_month_timeline_snapshot(month, db_path=db_path, backend="rule", dry_run=True)
    assert dry_snapshot.thought_summary
    assert dry_snapshot.evidence
    with connect(db_path) as conn:
        assert table_count(conn, "monthly_timeline_snapshots") == 0
        assert table_count(conn, "monthly_timeline_items") == 0

    stored = generate_month_timeline_snapshot(month, db_path=db_path, backend="rule")
    assert stored.title
    with connect(db_path) as conn:
        assert table_count(conn, "monthly_timeline_snapshots") == 1
        first_item_count = table_count(conn, "monthly_timeline_items")
        assert first_item_count > 0

    generate_month_timeline_snapshot(month, db_path=db_path, backend="rule", force=True)
    with connect(db_path) as conn:
        assert table_count(conn, "monthly_timeline_snapshots") == 1
        assert table_count(conn, "monthly_timeline_items") == first_item_count

    markdown = format_month_timeline_markdown(stored)
    report = format_timeline_report([stored], title="Test Timeline")
    qa = timeline_qa(month=month, db_path=db_path)
    assert "この月の概要" in markdown
    assert "Test Timeline" in report
    assert qa and "quality_score" in qa[0]

    with connect(db_path) as conn:
        snapshot_count = table_count(conn, "monthly_timeline_snapshots")
    dry_rows = generate_timeline_snapshots(all_months=True, db_path=db_path, backend="rule", dry_run=True)
    with connect(db_path) as conn:
        assert dry_rows
        assert table_count(conn, "monthly_timeline_snapshots") == snapshot_count

    output = tmp_path / "timeline.md"
    output.write_text(format_timeline_report([stored], title="Timeline 2026"), encoding="utf-8")
    assert "Timeline 2026" in output.read_text(encoding="utf-8")


def test_month_timeline_quality_warnings_for_title_only_and_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO notes(
                id, content_hash, title, body, source_path, source_relative_path, file_type,
                imported_at, note_date, parser_name
            )
            VALUES ('note-title-only', 'hash-title-only', 'Title Only', 'Body text', 'src.md', 'src.md', 'md', '2026-05-01', '2026-05-01', 'test')
            """
        )
        conn.execute(
            """
            INSERT INTO note_summaries(
                note_id, model_name, generated_title, one_line_summary, detailed_summary,
                important_points_json, revisit_reason, confidence, importance, evidence_json, created_at
            )
            VALUES (
                'note-title-only', 'mock', 'Title Only', 'Title Only', '',
                '[]', 'あとで見返す', 0.2, 0.8,
                '[{"note_id":"note-title-only","quote":"Title Only"}]',
                '2026-05-01'
            )
            """
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    warnings = set(snapshot.quality["warnings"])
    assert "title_only_evidence" in warnings
    assert "fallback_heavy" in warnings
    assert "low_confidence" in warnings
