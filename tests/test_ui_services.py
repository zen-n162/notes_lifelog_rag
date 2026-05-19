from pathlib import Path

from notes_lifelog_rag.analysis.service import analyze_all
from notes_lifelog_rag.db import schema
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.timeline.service import TimelineItem
from notes_lifelog_rag.ui import renderers
from notes_lifelog_rag.ui import services


FIXTURES = Path(__file__).parent / "fixtures" / "notes_export"


def test_note_detail_shows_ai_metadata(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "notes.db"
    monkeypatch.setattr(services, "init_db", lambda db_path_arg=None: schema.init_db(db_path))
    monkeypatch.setattr(services, "connect", lambda db_path_arg=None: schema.connect(db_path))
    monkeypatch.setattr(services, "database_path", lambda explicit_path=None: db_path)
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=1, backend_name="mock")

    notes = services.list_notes(limit=1)
    assert notes
    detail = services.get_note_detail(notes[0]["id"])
    assert detail["found"]
    assert detail["note"]["body"]
    assert detail["summary"]["confidence"] is not None
    assert detail["summary"]["importance"] is not None
    assert detail["summary"]["evidence"]


def test_render_note_cards_escapes_html() -> None:
    html = renderers.render_note_cards(
        [
            {
                "id": "note-1",
                "display_title": "<script>alert(1)</script>",
                "title": "<b>bad</b>",
                "one_line_summary": "<img src=x onerror=alert(1)>",
                "date_label": "2026-05",
                "source_short": "src.md",
                "categories": ["研究<script>"],
                "confidence": 0.4,
                "importance": 0.8,
                "important": True,
                "low_confidence": True,
                "evidence_missing": True,
            }
        ]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img" not in html
    assert 'role="button"' in html
    assert 'tabindex="0"' in html
    assert 'data-note-choice=' in html
    assert 'data-note-choice="note-1 · &lt;script&gt;alert(1)&lt;/script&gt;"' in html


def test_render_suggestions_are_clickable_and_escape_html() -> None:
    html = renderers.render_suggestions(
        [
            {
                "note_id": "note-1",
                "title": "<b>Suggestion</b>",
                "message": "<script>alert(1)</script>",
                "suggestion_type": "revisit_note",
                "status": "new",
                "importance": 0.8,
                "confidence": 0.7,
                "target_date": "2026-05-19",
                "source_relative_path": "src.md",
                "evidence": [{"note_id": "note-1", "quote": "<img src=x>"}],
            }
        ],
        selected_note_id="note-1",
    )
    assert "<script>" not in html
    assert "<img" not in html
    assert "suggestion-list" in html
    assert 'data-note-choice=' in html
    assert "note-card suggestion-card selected" in html


def test_review_timeline_and_reflection_cards_are_clickable_and_escape_html() -> None:
    qa_html = renderers.render_quality_warnings(
        [
            {
                "warning_type": "low_confidence",
                "note_id": "note-1",
                "title": "<b>QA</b>",
                "source_path": "<script>src</script>",
                "issue": "<img src=x>",
                "confidence": 0.2,
                "importance": 0.9,
                "evidence": [{"note_id": "note-1", "quote": "<script>x</script>"}],
            }
        ],
        selected_note_id="note-1",
    )
    timeline_html = renderers.render_timeline_cards(
        [
            TimelineItem(
                item_type="event",
                note_id="note-2",
                title="<b>Event</b>",
                summary="<script>bad</script>",
                date_label="2026-05",
                date_confidence="high",
                source_title="Source <img>",
                source_path="src.md",
                confidence=0.8,
                importance=0.7,
                evidence=[{"note_id": "note-2", "quote": "<img src=x>"}],
            )
        ],
        selected_note_id="note-2",
    )
    reflection_html = renderers.render_reflection_list(
        [
            {
                "month": "2026-05",
                "summary": {"reminder_messages": ["<script>bad</script>"]},
                "confidence": 0.6,
                "importance": 0.8,
                "updated_at": "2026-05-19",
                "evidence": [{"note_id": "note-3", "quote": "<img src=x>"}],
            }
        ],
        selected_note_id="note-3",
    )

    for html in [qa_html, timeline_html, reflection_html]:
        assert "<script>" not in html
        assert "<img" not in html
        assert 'data-note-choice=' in html
        assert "note-card selected" in html


def test_render_note_detail_handles_missing_outputs() -> None:
    html = renderers.render_note_detail(
        {
            "found": True,
            "note": {
                "id": "note-1",
                "title": "Test note",
                "body": "body text",
                "source_relative_path": "folder/test.md",
                "content_hash": "abc123",
                "note_date": None,
                "modified_at": None,
                "imported_at": "2026-01-01",
            },
            "summary": None,
            "categories": [],
            "events": [],
            "thoughts": [],
            "model_runs": [],
            "reflection": None,
            "warnings": ["summary_missing", "events_sparse", "thoughts_sparse"],
        }
    )
    assert "summaryが未生成" in html
    assert "Original Note" in html
    assert "body text" in html


def test_sidebar_category_and_month_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=2, backend_name="mock")

    categories = services.list_categories_with_counts(db_path)
    months = services.list_months_with_counts(db_path)

    assert categories
    assert any("name" in row and "note_count" in row for row in categories)
    assert months
    assert any("month" in row and "note_count" in row for row in months)


def test_get_missing_analysis_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=1, backend_name="mock")

    counts = services.get_missing_analysis_counts(db_path)

    assert counts["notes"] >= 1
    assert counts["summaries"] >= 1
    assert "summary_missing" in counts
    assert "model_runs_failures" in counts

    health = services.get_analysis_health(db_path)
    assert "summary_coverage" in health
    assert "health_status" in health


def test_ui_service_list_does_not_expose_full_body_but_detail_does(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)

    notes = services.list_notes(limit=1, db_path=db_path)
    assert notes
    assert "body" not in notes[0]

    detail = services.get_note_detail(notes[0]["id"], db_path=db_path)
    assert detail["found"]
    assert detail["note"]["body"]


def test_low_confidence_and_evidence_missing_flags(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    with schema.connect(db_path) as conn:
        note_id = conn.execute("SELECT id FROM notes LIMIT 1").fetchone()["id"]
        conn.execute(
            """
            INSERT INTO note_summaries(
                note_id, model_name, generated_title, one_line_summary, detailed_summary,
                important_points_json, revisit_reason, confidence, importance, evidence_json, created_at
            )
            VALUES (?, 'test', 'low', 'low summary', '', '[]', '', 0.2, 0.9, '[]', '2026-01-01')
            """,
            (note_id,),
        )
        conn.commit()

    notes = services.list_notes(limit=10, low_confidence=True, db_path=db_path)
    item = next(row for row in notes if row["id"] == note_id)
    detail = services.get_note_detail(note_id, db_path=db_path)

    assert item["low_confidence"] is True
    assert item["evidence_missing"] is True
    assert "low_confidence" in detail["warnings"]
    assert "evidence_missing" in detail["warnings"]

    warnings = services.get_quality_warnings(limit=20, db_path=db_path)
    warning_types = {row["warning_type"] for row in warnings}
    assert "low_confidence" in warning_types
    assert "evidence_missing" in warning_types


def test_generate_suggestions_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.db"
    schema.init_db(db_path)
    ingest_directory(FIXTURES, db_path)
    analyze_all(db_path=db_path, limit=2, backend_name="mock")

    first = services.generate_suggestions(limit=10, db_path=db_path)
    second = services.generate_suggestions(limit=10, db_path=db_path)
    suggestions = services.list_suggestions(limit=20, db_path=db_path)

    assert first["created"] > 0
    assert second["created"] == 0
    assert suggestions
    assert {"suggestion_type", "status", "evidence"}.issubset(suggestions[0].keys())


def test_ui_service_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    schema.init_db(db_path)

    assert services.list_notes(db_path=db_path) == []
    health = services.get_analysis_health(db_path)
    assert health["notes"] == 0
    assert services.get_running_jobs()["analyze_all"] in {True, False}
