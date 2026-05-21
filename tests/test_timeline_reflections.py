import json
from pathlib import Path

from notes_lifelog_rag.analysis.service import analyze_all
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.timeline.service import (
    TimelineBuildLimits,
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


def test_suggestion_created_at_is_not_used_for_month_attribution(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "old-note", "Old thinking", created_at="2024-01-10")
        conn.execute(
            """
            INSERT INTO suggestions(
                note_id, suggestion_type, title, message, target_date, importance,
                confidence, evidence_json, status, created_at
            )
            VALUES (?, 'today_rediscovery', 'Generated today', 'Look back',
                    '2026-05-21', 0.9, 0.9, ?, 'new', '2026-05-21T00:00:00')
            """,
            ("old-note", json.dumps([{"note_id": "old-note", "quote": "Old thinking"}])),
        )
        conn.commit()

    may_sources = get_month_sources("2026-05", db_path=db_path)
    old_sources = get_month_sources("2024-01", db_path=db_path)

    assert may_sources["suggestions"] == []
    assert len(old_sources["suggestions"]) == 1


def test_import_timestamp_created_at_does_not_collapse_timeline_months(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO notes(
                id, content_hash, title, body, source_path, source_relative_path, file_type,
                created_at, modified_at, imported_at, note_date, parser_name
            )
            VALUES (
                'imported-created-note', 'hash-imported-created-note', '2024/03/27',
                'Body text', 'src.md', 'src.md', 'md',
                '2026-05-18T08:31:26+00:00', '2024-03-27T04:48:19+00:00',
                '2026-05-18T08:34:34+00:00', '2024-03-27', 'test'
            )
            """
        )
        _insert_summary(
            conn,
            "imported-created-note",
            generated_title="2024年3月のメモ",
            one_line_summary="作成日時がimport日になっているメモ。",
        )
        conn.commit()

    months = [row.month for row in list_timeline_months(db_path=db_path, order="asc")]

    assert "2024-03" in months
    assert "2026-05" not in months


def test_suggestions_are_capped_and_do_not_drive_overview(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "direct-note", "QST ES and research plan", created_at="2026-05-03")
        _insert_summary(
            conn,
            "direct-note",
            generated_title="QST ES と研究計画",
            one_line_summary="研究計画、QST ES、メモ整理アプリ設計を整理したメモ。",
            importance=0.85,
        )
        for index in range(8):
            conn.execute(
                """
                INSERT INTO suggestions(
                    note_id, suggestion_type, title, message, target_date, importance,
                    confidence, evidence_json, status, created_at
                )
                VALUES (?, 'revisit_note', ?, ?, '2026-05-12', 0.95, 0.9, ?, 'new', '2026-04-30')
                """,
                (
                    "direct-note",
                    f"風のゆくえ / ado suggestion {index}",
                    "歌詞メモを見返す",
                    json.dumps([{"note_id": "direct-note", "quote": "研究計画、QST ES"}]),
                ),
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot(
        "2026-05",
        db_path=db_path,
        dry_run=True,
        limits=TimelineBuildLimits(max_suggestions=5),
    )

    assert sum(1 for item in snapshot.items if item.item_type == "suggestion") <= 5
    assert "QST ES" in snapshot.overview
    assert "風のゆくえ" not in snapshot.overview
    assert "suggestions_dominated" not in set(snapshot.quality["warnings"])


def test_low_value_content_is_marked_low_priority(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "lyric-note", "風のゆくえ / ado 歌詞", created_at="2026-05-04")
        _insert_summary(
            conn,
            "lyric-note",
            generated_title="風のゆくえ / ado 歌詞",
            one_line_summary="短い歌詞行\n短い歌詞行\n短い歌詞行\n短い歌詞行\n短い歌詞行\n短い歌詞行",
            source_quote="風のゆくえ / ado 歌詞",
            importance=0.9,
        )
        _insert_note(conn, "shopping-note", "買い物メモ", created_at="2026-05-05")
        _insert_summary(
            conn,
            "shopping-note",
            generated_title="買い物メモ",
            one_line_summary="スーパーで牛乳と卵を購入。合計1200円。",
            importance=0.8,
        )
        _insert_note(conn, "pdf-note", "garbage.pdf", created_at="2026-05-06", source_path="garbage.pdf")
        _insert_summary(
            conn,
            "pdf-note",
            generated_title="ごみ収集PDF",
            one_line_summary="｜｜｜ 断片  □□□",
            importance=0.8,
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    low_items = [item for item in snapshot.items if "low_priority" in item.categories]

    assert len(low_items) >= 3
    assert "low_value_items_present" in set(snapshot.quality["warnings"])
    assert "noisy_items_present" in set(snapshot.quality["warnings"])


def test_unknown_month_hidden_by_default_and_visible_when_requested(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "unknown-date", "Unknown date note", created_at="1900-01-01")
        conn.commit()

    assert "1900-01" not in [row.month for row in list_timeline_months(db_path=db_path)]
    assert "1900-01" in [
        row.month for row in list_timeline_months(db_path=db_path, include_unknown=True)
    ]


def test_generate_timeline_force_replaces_only_target_month(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "may-note", "May note", created_at="2026-05-01")
        _insert_summary(conn, "may-note", generated_title="May summary", one_line_summary="May summary body")
        _insert_note(conn, "jun-note", "June note", created_at="2026-06-01")
        _insert_summary(conn, "jun-note", generated_title="June summary", one_line_summary="June summary body")
        conn.commit()

    generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    generate_month_timeline_snapshot("2026-06", db_path=db_path, force=True)
    with connect(db_path) as conn:
        before_snapshots = table_count(conn, "monthly_timeline_snapshots")
        june_items = conn.execute(
            "SELECT COUNT(*) AS count FROM monthly_timeline_items WHERE month = '2026-06'"
        ).fetchone()["count"]

    generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)

    with connect(db_path) as conn:
        assert table_count(conn, "monthly_timeline_snapshots") == before_snapshots
        assert (
            conn.execute("SELECT COUNT(*) AS count FROM monthly_timeline_items WHERE month = '2026-06'").fetchone()["count"]
            == june_items
        )


def _insert_note(
    conn,
    note_id: str,
    title: str,
    *,
    created_at: str,
    source_path: str = "src.md",
) -> None:
    conn.execute(
        """
        INSERT INTO notes(
            id, content_hash, title, body, source_path, source_relative_path, file_type,
            created_at, modified_at, imported_at, note_date, parser_name
        )
        VALUES (?, ?, ?, ?, ?, ?, 'md', ?, ?, ?, ?, 'test')
        """,
        (
            note_id,
            f"hash-{note_id}",
            title,
            f"Body for {title}",
            source_path,
            source_path,
            created_at,
            created_at,
            created_at,
            created_at,
        ),
    )


def _insert_summary(
    conn,
    note_id: str,
    *,
    generated_title: str,
    one_line_summary: str,
    source_quote: str | None = None,
    importance: float = 0.8,
) -> None:
    conn.execute(
        """
        INSERT INTO note_summaries(
            note_id, model_name, generated_title, one_line_summary, detailed_summary,
            important_points_json, revisit_reason, confidence, importance, evidence_json, created_at
        )
        VALUES (?, 'mock', ?, ?, '', '[]', 'あとで見返す', 0.8, ?, ?, '2026-05-01')
        """,
        (
            note_id,
            generated_title,
            one_line_summary,
            importance,
            json.dumps([{"note_id": note_id, "quote": source_quote or one_line_summary[:60]}]),
        ),
    )
