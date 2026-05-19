from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # pragma: no cover - dependency is declared.
    BeautifulSoup = None


SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".json", ".pdf"}


class ParserError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedNote:
    title: str
    body: str
    parser_name: str
    metadata: dict[str, Any]


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)

    def text(self) -> str:
        return "\n".join(self.parts)


def parse_note_file(path: Path) -> ParsedNote:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParserError(f"Unsupported file extension: {suffix or '(none)'}")
    if suffix == ".md":
        return _parse_markdown(path)
    if suffix == ".txt":
        return _parse_text(path)
    if suffix in {".html", ".htm"}:
        return _parse_html(path)
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ParserError(f"No parser registered for {suffix}")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ParserError("Could not decode text file as UTF-8 or CP932.")


def _parse_markdown(path: Path) -> ParsedNote:
    text = _read_text(path)
    text = _strip_frontmatter(text)
    title = _first_markdown_heading(text) or path.stem
    return ParsedNote(title=title, body=text.strip(), parser_name="markdown", metadata={})


def _parse_text(path: Path) -> ParsedNote:
    text = _read_text(path)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = first_line[:80] if first_line else path.stem
    return ParsedNote(title=title, body=text.strip(), parser_name="text", metadata={})


def _parse_html(path: Path) -> ParsedNote:
    html = _read_text(path)
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = soup.get_text("\n")
    else:
        parser = _HTMLTextParser()
        parser.feed(html)
        title = path.stem
        body = parser.text()
    return ParsedNote(title=title, body=_collapse_blank_lines(body), parser_name="html", metadata={})


def _parse_json(path: Path) -> ParsedNote:
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ParserError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        title = str(data.get("title") or data.get("name") or path.stem)
        body_value = data.get("body") or data.get("text") or data.get("content")
        if body_value is None:
            body = json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(body_value, str):
            body = body_value
        else:
            body = json.dumps(body_value, ensure_ascii=False, indent=2)
    else:
        title = path.stem
        body = json.dumps(data, ensure_ascii=False, indent=2)
    return ParsedNote(title=title, body=body.strip(), parser_name="json", metadata={})


def _parse_pdf(path: Path) -> ParsedNote:
    for logger_name in ("pypdf", "pypdf._cmap", "pypdf._reader", "pypdf.generic._base"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
    warnings.filterwarnings("ignore", module="pypdf")
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - covered by smoke env.
        raise ParserError("pypdf is not installed; PDF parsing is disabled.") from exc

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # pypdf raises several format-specific exceptions.
        raise ParserError(f"Could not parse PDF: {exc}") from exc
    body = _collapse_blank_lines("\n\n".join(page for page in pages if page))
    if not body:
        raise ParserError("PDF did not contain extractable text.")
    return ParsedNote(
        title=path.stem,
        body=body,
        parser_name="pdf",
        metadata={"page_count": len(reader.pages)},
    )


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :].lstrip()
    return text


def _first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def _collapse_blank_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed.strip()
