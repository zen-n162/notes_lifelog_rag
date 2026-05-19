from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class DateParseResult:
    iso_date: str | None
    date_label: str
    confidence: str


def parse_date_label(text: str, context_date: str | date | None = None) -> DateParseResult:
    context = _coerce_context_date(context_date)
    text = text.strip()

    full = re.search(r"(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if full:
        return _date_result(int(full.group(1)), int(full.group(2)), int(full.group(3)), full.group(0), "high")

    jp_full = re.search(r"(20\d{2}|19\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if jp_full:
        return _date_result(
            int(jp_full.group(1)), int(jp_full.group(2)), int(jp_full.group(3)), jp_full.group(0), "high"
        )

    month_day = re.search(r"(?<!年)(\d{1,2})月\s*(\d{1,2})日", text)
    if month_day:
        label = month_day.group(0)
        if context:
            return _date_result(context.year, int(month_day.group(1)), int(month_day.group(2)), label, "medium")
        return DateParseResult(None, label, "low")

    relative = _relative_date(text, context)
    if relative:
        return relative

    vague = _vague_date(text, context)
    if vague:
        return vague

    return DateParseResult(None, "", "unknown")


def _relative_date(text: str, context: date | None) -> DateParseResult | None:
    base = context or date.today()
    mapping = {
        "今日": (base, "medium" if context else "low"),
        "昨日": (base - timedelta(days=1), "medium" if context else "low"),
        "明日": (base + timedelta(days=1), "medium" if context else "low"),
        "先週": (base - timedelta(days=7), "low"),
    }
    for label, (value, confidence) in mapping.items():
        if label in text:
            return DateParseResult(value.isoformat(), label, confidence)
    if "先月" in text:
        year = base.year
        month = base.month - 1
        if month == 0:
            year -= 1
            month = 12
        return DateParseResult(f"{year:04d}-{month:02d}", "先月", "low")
    return None


def _vague_date(text: str, context: date | None) -> DateParseResult | None:
    year = context.year if context else None
    seasons = {
        "春頃": "spring",
        "夏頃": "summer",
        "秋頃": "autumn",
        "冬頃": "winter",
    }
    for label, season in seasons.items():
        if label in text:
            prefix = f"{year}-" if year else ""
            return DateParseResult(None, f"{prefix}{label}", "low")

    around_month = re.search(r"(\d{4}|何年)\s*年\s*(\d{1,2}|何)\s*月\s*ごろ", text)
    if around_month:
        return DateParseResult(None, around_month.group(0), "low")
    return None


def _date_result(year: int, month: int, day: int, label: str, confidence: str) -> DateParseResult:
    try:
        value = date(year, month, day)
    except ValueError:
        return DateParseResult(None, label, "low")
    return DateParseResult(value.isoformat(), label, confidence)


def _coerce_context_date(value: str | date | None) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None

