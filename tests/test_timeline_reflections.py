from pathlib import Path

from notes_lifelog_rag.analysis.service import analyze_all
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.timeline.service import (
    build_monthly_reflection,
    build_timeline,
    format_reflection_markdown,
    format_timeline_markdown,
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
