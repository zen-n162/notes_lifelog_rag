import json
from pathlib import Path

from typer.testing import CliRunner

from notes_lifelog_rag.analysis.service import analyze_all
from notes_lifelog_rag.cli import app
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.timeline.service import (
    TimelineBuildLimits,
    build_monthly_reflection,
    build_month_timeline_items,
    build_timeline,
    format_month_timeline_markdown,
    format_reflection_markdown,
    format_timeline_qa_pretty,
    format_timeline_report,
    format_timeline_markdown,
    generate_month_timeline_snapshot,
    generate_timeline_snapshots,
    get_month_sources,
    list_timeline_months,
    timeline_qa,
    timeline_item_display_groups,
    visible_timeline_flags,
)


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"
RUNNER = CliRunner()


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


def test_direct_thoughts_events_and_enriched_evidence_drive_month_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    body = (
        "QST ESでは、宇宙データを対象にした機械学習経験を公共性の高い研究支援へ接続する方針を整理した。"
        "月面探査やハイパースペクトル解析も研究計画として見直している。"
    )
    with connect(db_path) as conn:
        _insert_note(conn, "qst-note", "QST ES", created_at="2026-05-03", body=body)
        _insert_summary(
            conn,
            "qst-note",
            generated_title="QST ES",
            one_line_summary="宇宙データと機械学習経験を公共性の高い研究支援に接続するメモ。",
            source_quote="# QST ES",
            importance=0.85,
        )
        conn.execute(
            """
            INSERT INTO thoughts(
                note_id, title, summary, thought_type, themes_json, emotion_label,
                emotion_intensity, date_label, importance, confidence, remember_reason,
                evidence_json, created_at
            )
            VALUES (
                'qst-note', '研究経験の接続', '宇宙データの機械学習経験を公共性の高い研究支援へ接続しようとしている。',
                'career', '["研究","QST"]', '', 0.0, '2026-05-03', 0.8, 0.8, '志望軸として再確認する',
                '[{"note_id":"qst-note","quote":"QST ES"}]', '2026-05-03'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO events(
                note_id, title, summary, event_type, event_date, date_label,
                date_confidence, importance, confidence, evidence_json, created_at
            )
            VALUES (
                'qst-note', 'QST ES整理', 'QST ESの志望動機と自己PRを整理した。',
                'career', '2026-05-03', '2026-05-03', 'high', 0.75, 0.8,
                '[{"note_id":"qst-note","quote":"QST ES"}]', '2026-05-03'
            )
            """
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    main_items = [item for item in snapshot.items if item.item_type != "suggestion" and "low_priority" not in item.categories]

    assert "thought抽出はまだ少ないため" not in snapshot.thought_summary
    assert "宇宙データ" in snapshot.thought_summary
    assert "QST ES" in snapshot.event_summary
    assert any(item.item_type == "thought" for item in main_items)
    assert any(item.item_type == "event" for item in main_items)
    assert any(item.evidence_enriched for item in snapshot.items)
    assert "title_only_evidence" not in set(snapshot.quality["warnings"])
    assert "date_attribution_uncertain" not in set(snapshot.quality["warnings"])


def test_grouped_timeline_merges_same_note_sections_and_preserves_sub_items(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(
            conn,
            "qst-note",
            "QST ES",
            created_at="2026-05-03",
            body="QST ESでは、研究経験と自己PR、志望動機を整理した。公共性の高い研究支援に関心がある。",
        )
        _insert_summary(
            conn,
            "qst-note",
            generated_title="QST ES",
            one_line_summary="研究経験をQSTでの公共性の高い研究支援に接続するメモ。",
            importance=0.9,
        )
        for title in ["自己PR", "志望動機", "趣味"]:
            conn.execute(
                """
                INSERT INTO events(
                    note_id, title, summary, event_type, event_date, date_label,
                    date_confidence, importance, confidence, evidence_json, created_at
                )
                VALUES (?, ?, ?, 'section', '2026-05-03', '2026-05-03', 'high', 0.15, 0.2, ?, '2026-05-03')
                """,
                (
                    "qst-note",
                    title,
                    f"QST ES内の{title}セクション。",
                    json.dumps([{"note_id": "qst-note", "quote": title}]),
                ),
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    groups = timeline_item_display_groups(snapshot.items)
    qst_group = next(item for item in groups["main"] if "QST ES" in item.title)

    assert qst_group.title == "QST ES・自己PR・志望動機の整理"
    assert len(qst_group.sub_items) >= 3
    assert "duplicate_same_note" in qst_group.quality_flags
    assert not any(item.title in {"自己PR", "志望動機", "趣味"} for item in groups["low_priority"])


def test_natural_narrative_avoids_long_raw_truncated_quotes(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    long_body = "QST ESでは、宇宙データを対象にした機械学習経験を公共性の高い研究支援へ接続する方針を整理した。" * 8
    with connect(db_path) as conn:
        _insert_note(conn, "qst-note", "QST ES", created_at="2026-05-03", body=long_body)
        _insert_summary(
            conn,
            "qst-note",
            generated_title="QST ES",
            one_line_summary=long_body,
            source_quote="# QST ES",
            importance=0.9,
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)

    assert "…" not in snapshot.overview
    assert "…" not in snapshot.thought_summary
    assert long_body[:180] not in snapshot.overview
    assert "社会的意義のあるAI活用" in snapshot.overview


def test_low_priority_reason_flags_and_default_limit_three(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        for index in range(6):
            note_id = f"scan-{index}"
            _insert_note(conn, note_id, f"スキャンPDF {index}", created_at="2026-05-01", source_path=f"scan-{index}.pdf")
            _insert_summary(
                conn,
                note_id,
                generated_title=f"スキャンPDF {index}",
                one_line_summary=f"｜｜｜ ノイズ断片 {index}",
                importance=0.8,
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    groups = timeline_item_display_groups(snapshot.items)
    assert groups["low_priority"]
    assert {"noisy_pdf", "scanned_document"} & set(groups["low_priority"][0].quality_flags)
    default_md = format_month_timeline_markdown(snapshot)
    full_md = format_month_timeline_markdown(snapshot, show_low_priority=True)

    assert "... (+" in default_md
    assert "スキャンPDF 5" not in default_md
    assert "スキャンPDF 5" in full_md


def test_timeline_report_uses_grouped_view_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "qst-note", "QST ES", created_at="2026-05-03")
        _insert_summary(
            conn,
            "qst-note",
            generated_title="QST ES",
            one_line_summary="研究経験を志望動機に整理するメモ。",
            importance=0.9,
        )
        conn.execute(
            """
            INSERT INTO events(
                note_id, title, summary, event_type, event_date, date_label,
                date_confidence, importance, confidence, evidence_json, created_at
            )
            VALUES ('qst-note', '自己PR', '自己PRを整理した。', 'section', '2026-05-03', '2026-05-03', 'high', 0.15, 0.2,
                    '[{"note_id":"qst-note","quote":"自己PR"}]', '2026-05-03')
            """
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    grouped_report = format_timeline_report([snapshot], title="Timeline 2026")
    ungrouped_report = format_timeline_report([snapshot], title="Timeline 2026", grouped=False, show_low_priority=True)

    assert "QST ES・自己PR・志望動機の整理" in grouped_report
    assert "sub_items:" in grouped_report
    assert "自己PR" in ungrouped_report


def test_unknown_month_triggers_date_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "unknown-note", "Unknown note", created_at="1900-01-01")
        _insert_summary(
            conn,
            "unknown-note",
            generated_title="Unknown",
            one_line_summary="日付が不明なメモ。",
            importance=0.8,
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("1900-01", db_path=db_path, dry_run=True)

    assert "invalid_month_1900" in set(snapshot.quality["warnings"])
    assert "date_attribution_uncertain" in set(snapshot.quality["warnings"])


def test_low_priority_hidden_by_default_and_visible_when_requested(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        for index in range(7):
            note_id = f"scan-{index}"
            _insert_note(conn, note_id, f"スキャンPDF {index}", created_at="2026-05-01", source_path=f"scan-{index}.pdf")
            _insert_summary(
                conn,
                note_id,
                generated_title=f"スキャンPDF {index}",
                one_line_summary=f"｜｜｜ ノイズ断片 {index}",
                importance=0.8,
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    default_md = format_month_timeline_markdown(snapshot)
    full_md = format_month_timeline_markdown(snapshot, show_low_priority=True)

    assert "... (+" in default_md
    assert "スキャンPDF 6" not in default_md
    assert "スキャンPDF 6" in full_md
    assert "スキャンPDF" not in snapshot.overview


def test_suggestions_do_not_contribute_to_overview(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(
            conn,
            "main-note",
            "研究計画",
            created_at="2026-05-01",
            body="月面探査と機械学習の研究計画を整理し、ハイパースペクトル解析の方向性を確認した。",
        )
        _insert_summary(
            conn,
            "main-note",
            generated_title="研究計画",
            one_line_summary="月面探査と機械学習の研究計画を整理した。",
            importance=0.85,
        )
        conn.execute(
            """
            INSERT INTO suggestions(
                note_id, suggestion_type, title, message, target_date, importance,
                confidence, evidence_json, status, created_at
            )
                VALUES (
                    'main-note', 'revisit_note', 'マスカラ / SixTONES', '歌詞を見返す',
                    '2026-05-02', 0.95, 0.9,
                    '[{"note_id":"main-note","quote":"月面探査と機械学習"}]', 'new', '2026-05-21'
                )
                """
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)

    assert "月面探査" in snapshot.overview
    assert "マスカラ" not in snapshot.overview


def test_source_note_id_is_short_ref_and_recovered_from_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    long_id = "dd2217df77ccbce3e922f698ba9dd7b4a0e6b2dacb5d658f5fd617e034efbaa9"
    with connect(db_path) as conn:
        _insert_note(conn, long_id, "QST ES", created_at="2026-05-03")
        _insert_summary(
            conn,
            long_id,
            generated_title="QST ES",
            one_line_summary="QST ESを整理した。",
            importance=0.9,
        )
        conn.execute(
            """
            INSERT INTO suggestions(
                note_id, suggestion_type, title, message, target_date, importance,
                confidence, evidence_json, status, created_at
            )
            VALUES (
                    NULL, 'important_thought', 'QST ES再確認', 'QST ESを見返す',
                '', 0.9, 0.8, ?, 'new', '2026-05-21'
            )
            """,
            (json.dumps([{"note_id": long_id, "quote": "QST ESを整理した。"}]),),
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)
    assert any(item.source_note_id == "dd2217df77cc" for item in snapshot.items)
    assert any(
        item.item_type == "suggestion" and item.source_note_id == "dd2217df77cc"
        for item in snapshot.items
    )
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT source_note_id, evidence_json, grouped_item_ids_json, sub_items_json
            FROM monthly_timeline_items
            WHERE month='2026-05' AND source_note_id='dd2217df77cc'
            """
        ).fetchall()
    assert rows
    assert any("dd2217df77cc" in row["evidence_json"] for row in rows)
    assert all(row["grouped_item_ids_json"] for row in rows)


def test_grouped_flags_are_filtered_but_ungrouped_keeps_duplicate_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "qst-note", "QST ES", created_at="2026-05-03")
        _insert_summary(
            conn,
            "qst-note",
            generated_title="QST ES",
            one_line_summary="QST ESで研究経験と志望動機、自己PRを整理した。",
            importance=0.9,
        )
        conn.execute(
            """
            INSERT INTO events(
                note_id, title, summary, event_type, event_date, date_label,
                date_confidence, importance, confidence, evidence_json, created_at
            )
            VALUES ('qst-note', '自己PR', '自己PRを整理した。', 'section', '2026-05-03', '2026-05-03', 'high', 0.2, 0.3,
                    '[{"note_id":"qst-note","quote":"自己PR"}]', '2026-05-03')
            """
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    grouped_item = next(item for item in timeline_item_display_groups(snapshot.items, grouped=True)["main"] if "QST ES" in item.title)
    ungrouped_item = next(item for item in timeline_item_display_groups(snapshot.items, grouped=False)["main"] if item.source_note_id == "qst-note")

    assert "duplicate_same_note" in grouped_item.quality_flags
    assert "duplicate_same_note" not in visible_timeline_flags(grouped_item)
    assert "duplicate_same_note" in visible_timeline_flags(ungrouped_item, grouped=False)


def test_hide_low_priority_summarizes_reasons_without_titles(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        for index in range(4):
            note_id = f"scan-{index}"
            _insert_note(conn, note_id, f"スキャンPDF {index}", created_at="2026-05-01", source_path=f"scan-{index}.pdf")
            _insert_summary(
                conn,
                note_id,
                generated_title=f"スキャンPDF {index}",
                one_line_summary=f"｜｜｜ ノイズ断片 {index}",
                importance=0.8,
            )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)
    hidden = format_month_timeline_markdown(snapshot, hide_low_priority=True)
    shown = format_month_timeline_markdown(snapshot, show_low_priority=True)

    assert "主な理由:" in hidden
    assert "表示するには --show-low-priority" in hidden
    assert "スキャンPDF 3" not in hidden
    assert "スキャンPDF 3" in shown


def test_date_modified_only_is_info_warning_and_qa_links_warning_items(tmp_path: Path) -> None:
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
                'modified-note', 'hash-modified', 'Modified only note', 'modified body',
                'src.md', 'src.md', 'md', '2026-05-21', '2026-04-10', '2026-05-21', '', 'test'
            )
            """
        )
        _insert_summary(
            conn,
            "modified-note",
            generated_title="Modified only note",
            one_line_summary="modified_atで月に帰属するメモ。",
            importance=0.8,
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-04", db_path=db_path, force=True)
    qa = timeline_qa(month="2026-04", db_path=db_path, show_items=True)[0]

    assert "date_modified_only" in snapshot.quality["info_warnings"]
    assert "date_modified_only" not in snapshot.quality["warnings"]
    assert "date_modified_only" in qa["info_warnings"]
    assert "date_modified_only" in qa["warning_items"]


def test_generated_date_suggestion_is_excluded_when_no_target_or_evidence_date(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "main-note", "研究計画", created_at="2026-04-01")
        _insert_summary(
            conn,
            "main-note",
            generated_title="研究計画",
            one_line_summary="研究計画を整理した。",
            importance=0.8,
        )
        conn.execute(
            """
            INSERT INTO suggestions(
                note_id, suggestion_type, title, message, target_date, importance,
                confidence, evidence_json, status, created_at
            )
            VALUES (NULL, 'revisit_note', '生成日だけのsuggestion', '生成日だけ', '', 0.9, 0.9, '[]', 'new', '2026-05-21')
            """
        )
        conn.commit()

    snapshot = generate_month_timeline_snapshot("2026-05", db_path=db_path, dry_run=True)

    assert not any(item.item_type == "suggestion" and item.title == "生成日だけのsuggestion" for item in snapshot.items)
    assert "generated_date_used_for_suggestion" not in snapshot.quality["warnings"]


def test_timeline_qa_markdown_and_json_outputs_include_problem_items(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "scan-note", "スキャンPDF", created_at="2026-05-01", source_path="scan.pdf")
        _insert_summary(
            conn,
            "scan-note",
            generated_title="スキャンPDF",
            one_line_summary="｜｜｜ ノイズ断片",
            importance=0.8,
        )
        conn.commit()
    generate_month_timeline_snapshot("2026-05", db_path=db_path, force=True)

    markdown_path = tmp_path / "timeline_qa.md"
    json_path = tmp_path / "timeline_qa.json"
    md_result = RUNNER.invoke(
        app,
        [
            "timeline-qa",
            "--month",
            "2026-05",
            "--show-items",
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
            "--db",
            str(db_path),
        ],
    )
    json_result = RUNNER.invoke(
        app,
        [
            "timeline-qa",
            "--month",
            "2026-05",
            "--format",
            "json",
            "--output",
            str(json_path),
            "--db",
            str(db_path),
        ],
    )

    assert md_result.exit_code == 0, md_result.output
    assert json_result.exit_code == 0, json_result.output
    markdown = markdown_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "# Timeline QA Report" in markdown
    assert "Problem items:" in markdown
    assert "source_note_id:" in markdown
    assert "flags:" in markdown
    assert '"warning_items"' not in markdown
    assert data and data[0]["month"] == "2026-05"
    assert data[0]["warning_items"]
    first_problem = next(iter(data[0]["warning_items"].values()))[0]
    assert first_problem["source_note_id"] == "scan-note"
    assert first_problem["quality_flags"]


def test_timeline_qa_pretty_output_is_not_inline_json(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "scan-note", "スキャンPDF", created_at="2026-05-01", source_path="scan.pdf")
        _insert_summary(
            conn,
            "scan-note",
            generated_title="スキャンPDF",
            one_line_summary="｜｜｜ ノイズ断片",
            importance=0.8,
        )
        conn.commit()
    qa_rows = timeline_qa(month="2026-05", db_path=db_path, show_items=True)
    pretty = format_timeline_qa_pretty(qa_rows)

    assert "## 2026-05" in pretty
    assert "Problem items:" in pretty
    assert "source_note_id:" in pretty
    assert '{"' not in pretty


def test_timeline_qa_only_problems_and_month_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    init_db(db_path)
    with connect(db_path) as conn:
        _insert_note(conn, "clean-note", "研究計画", created_at="2026-05-01")
        _insert_summary(
            conn,
            "clean-note",
            generated_title="研究計画",
            one_line_summary="月面探査と機械学習に関する研究計画を整理した。",
            source_quote="月面探査と機械学習に関する研究計画を整理した。",
            importance=0.9,
        )
        for index in range(2):
            _insert_event(conn, "clean-note", f"研究イベント{index}", confidence=0.9, importance=0.8)
            _insert_thought(conn, "clean-note", f"研究思考{index}", confidence=0.9, importance=0.8)
        _insert_note(conn, "problem-note", "スキャンPDF", created_at="2026-06-01", source_path="scan.pdf")
        _insert_summary(
            conn,
            "problem-note",
            generated_title="スキャンPDF",
            one_line_summary="｜｜｜ ノイズ断片",
            importance=0.8,
        )
        conn.commit()

    clean_result = RUNNER.invoke(
        app,
        ["timeline-qa", "--all-months", "--only-problems", "--format", "json", "--db", str(db_path)],
    )
    month_result = RUNNER.invoke(
        app,
        ["timeline-qa", "--month", "2026-06", "--format", "json", "--db", str(db_path)],
    )

    assert clean_result.exit_code == 0, clean_result.output
    data = json.loads(clean_result.output)
    assert [row["month"] for row in data] == ["2026-06"]
    assert month_result.exit_code == 0, month_result.output
    month_data = json.loads(month_result.output)
    assert [row["month"] for row in month_data] == ["2026-06"]


def _insert_note(
    conn,
    note_id: str,
    title: str,
    *,
    created_at: str,
    source_path: str = "src.md",
    body: str | None = None,
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
            body or f"Body for {title}",
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


def _insert_event(
    conn,
    note_id: str,
    title: str,
    *,
    confidence: float,
    importance: float,
) -> None:
    conn.execute(
        """
        INSERT INTO events(
            note_id, title, summary, event_type, event_date, date_label,
            date_confidence, importance, confidence, evidence_json, created_at
        )
        VALUES (?, ?, ?, 'research', '2026-05-01', '2026-05-01', 'high', ?, ?, ?, '2026-05-01')
        """,
        (
            note_id,
            title,
            f"{title}を進めた。",
            importance,
            confidence,
            json.dumps([{"note_id": note_id, "quote": f"{title}について、月面探査と機械学習の研究方針を整理した。"}]),
        ),
    )


def _insert_thought(
    conn,
    note_id: str,
    title: str,
    *,
    confidence: float,
    importance: float,
) -> None:
    conn.execute(
        """
        INSERT INTO thoughts(
            note_id, title, summary, thought_type, themes_json, emotion_label,
            emotion_intensity, importance, confidence, remember_reason, evidence_json, created_at
        )
        VALUES (?, ?, ?, 'research', '["研究"]', '', 0.0, ?, ?, '研究方針を見返す', ?, '2026-05-01')
        """,
        (
            note_id,
            title,
            f"{title}について考えた。",
            importance,
            confidence,
            json.dumps([{"note_id": note_id, "quote": f"{title}について、月面探査と機械学習の研究方針を整理した。"}]),
        ),
    )
