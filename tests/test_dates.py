from notes_lifelog_rag.utils.dates import parse_date_label


def test_parse_full_date() -> None:
    result = parse_date_label("2026-05-18 研究メモ")
    assert result.iso_date == "2026-05-18"
    assert result.confidence == "high"


def test_parse_month_day_with_context() -> None:
    result = parse_date_label("5月18日", context_date="2026-05-20")
    assert result.iso_date == "2026-05-18"
    assert result.confidence == "medium"


def test_parse_relative_date_with_context() -> None:
    result = parse_date_label("昨日のメモ", context_date="2026-05-18")
    assert result.iso_date == "2026-05-17"
    assert result.confidence == "medium"

