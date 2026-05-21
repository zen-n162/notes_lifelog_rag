from __future__ import annotations

import re
from typing import Any, Iterable


def build_month_overview(
    month: str,
    grouped_items: list[Any],
    themes: list[str],
    source_counts: dict[str, int],
    quality_warnings: list[str],
) -> str:
    label = _month_label(month)
    text = _joined_text(grouped_items, themes)
    theme_phrase = _theme_phrase(themes)
    sentences: list[str] = []
    if _has_qst(text) and _has_research(text, themes):
        sentences.append(
            f"{label}は、QST ESや研究メモを通じて、研究経験を社会的意義のあるAI活用へ接続しようとしていた月です。"
        )
    elif _has_research(text, themes):
        sentences.append(f"{label}は、{theme_phrase or '研究'}に関する整理や学習が中心に見える月です。")
    elif theme_phrase:
        sentences.append(f"{label}は、{theme_phrase}に関するメモが中心に残っている月です。")
    else:
        sentences.append(f"{label}は、複数の記録をもとに後から見返せる月です。")

    if _has_any(text, ["月面", "宇宙", "ハイパースペクトル", "機械学習", "EmerDiff", "拡散モデル"]):
        sentences.append(
            "宇宙データ、月面探査、機械学習に関する研究整理が続いており、技術を実用的な形へ整える関心が表れています。"
        )
    elif grouped_items:
        focus = _compact_item_titles(grouped_items, limit=3)
        if focus:
            sentences.append(f"主な手がかりとして、{focus} が残っています。")

    if (source_counts.get("thoughts", 0) or 0) > 0 or (source_counts.get("events", 0) or 0) > 0:
        sentences.append("直接抽出されたthought/eventもあるため、月の中心テーマはnote summaryだけには依存していません。")
    if any(warning in quality_warnings for warning in {"noisy_items_present", "low_value_items_present"}):
        sentences.append("一方で、PDFやスキャン由来のノイズを含む記録もあるため、重要項目は元メモ確認を前提に読むのがよさそうです。")
    return _limit_sentences(sentences, 4)


def build_month_thought_summary(
    grouped_items: list[Any],
    themes: list[str],
    source_counts: dict[str, int],
) -> str:
    text = _joined_text(grouped_items, themes)
    sentences: list[str] = []
    if _has_qst(text) and _has_research(text, themes):
        sentences.append(
            "この月は、宇宙データを対象にした機械学習経験を、自分の強みや志望動機として整理していた可能性があります。"
        )
        sentences.append(
            "単にモデル精度を高めるだけでなく、前処理・再現性・説明可能性を含めて、他者が使える技術にする意識が見られます。"
        )
    else:
        thought_like = _items_with_type(grouped_items, {"thought"})
        material = thought_like or grouped_items
        for item in material[:3]:
            phrase = _clean_summary(_item_summary(item) or _item_title(item))
            if phrase:
                sentences.append(f"{phrase}という考えが手がかりになります。")
    if not sentences:
        if source_counts.get("thoughts", 0):
            sentences.append("この月にはthought抽出があり、当時の考えを確認できる材料が残っています。")
        else:
            sentences.append("この月のthought抽出はまだ少ないため、何を考えていたかはnote summaryからの控えめな推定になります。")
    if 0 < (source_counts.get("thoughts", 0) or 0) < 3:
        sentences.append("ただし抽出数は少ないため、元メモとあわせて読むのが安全です。")
    return _limit_sentences(sentences, 4)


def build_month_event_summary(
    grouped_items: list[Any],
    themes: list[str],
    source_counts: dict[str, int],
) -> str:
    text = _joined_text(grouped_items, themes)
    event_like = _items_with_type(grouped_items, {"event", "note_summary", "reflection"})
    titles = _compact_item_titles(event_like or grouped_items, limit=4)
    if _has_qst(text):
        parts = ["QST ESの作成・整理"]
        if _has_any(text, ["研究プロンプト"]):
            parts.append("研究プロンプトの作成")
        if _has_any(text, ["EmerDiff", "拡散モデル"]):
            parts.append("EmerDiff論文メモ")
        if len(parts) == 1 and titles:
            parts.append(titles)
        return f"{'、'.join(_dedupe(parts))}など、研究と就職活動に関する記録が残っています。"
    if titles:
        suffix = "などの出来事や進展が記録されています。"
        if (source_counts.get("events", 0) or 0) == 0:
            suffix = "などがnote summary上の主な手がかりです。"
        return f"{titles} {suffix}"
    return "この月のevent抽出はまだ少ないため、出来事の把握はまだ不十分です。analyze-all後に再生成すると改善します。"


def build_revisit_reasons(grouped_items: list[Any]) -> list[str]:
    values = []
    for item in grouped_items:
        phrase = _clean_summary(_item_summary(item))
        title = _item_title(item)
        if title and phrase:
            values.append(f"{title}: {phrase}")
        elif title:
            values.append(f"{title}: 当時の判断材料を見返す手がかりになります。")
        if len(values) >= 6:
            break
    return _dedupe(values)


def build_important_changes(grouped_items: list[Any]) -> list[str]:
    values = []
    for item in grouped_items:
        title = _item_title(item)
        if not title:
            continue
        if _item_type(item) == "thought":
            values.append(f"{title}: 考え方や志望軸を見返す手がかりです。")
        else:
            values.append(f"{title}: その月の出来事・進展として後で見返す価値があります。")
        if len(values) >= 5:
            break
    return values


def summarize_grouped_item(items: list[Any], title: str) -> str:
    text = _joined_text(items, [title])
    if _has_qst(text):
        return (
            "宇宙データに機械学習を活用してきた研究経験を、QSTでの公共性の高い研究・医療支援に接続しようとしていた。"
            "自己PRや志望動機では、前処理・再現性・説明可能性まで整える姿勢が表れている。"
        )
    if _has_any(text, ["EmerDiff", "拡散モデル", "セグメンテーション"]):
        return "拡散モデル内部のセマンティックな情報を、追加学習なしのセグメンテーションに活用する研究メモです。"
    if _has_any(text, ["研究プロンプト", "Vit attention", "ディレクトリー整理"]):
        return "研究プロンプト、ViT attention、ディレクトリ整理など、研究作業の論点を短く整理したメモです。"
    phrases = [_clean_summary(_item_summary(item)) for item in items]
    phrases = [phrase for phrase in phrases if phrase]
    if phrases:
        return " / ".join(_dedupe(phrases)[:2])
    return _clean_summary(title) or title


def _joined_text(items: Iterable[Any], themes: Iterable[str]) -> str:
    parts = list(themes)
    for item in items:
        parts.extend([_item_title(item), _item_summary(item), str(getattr(item, "detail", "") or "")])
        for sub_item in getattr(item, "sub_items", []) or []:
            if isinstance(sub_item, dict):
                parts.extend([str(sub_item.get("title") or ""), str(sub_item.get("summary") or "")])
    return "\n".join(part for part in parts if part)


def _items_with_type(items: list[Any], types: set[str]) -> list[Any]:
    result = []
    for item in items:
        if _item_type(item) in types:
            result.append(item)
            continue
        sub_types = {str(sub.get("item_type") or "") for sub in getattr(item, "sub_items", []) or [] if isinstance(sub, dict)}
        if sub_types & types:
            result.append(item)
    return result


def _item_title(item: Any) -> str:
    return str(getattr(item, "title", "") or "")


def _item_summary(item: Any) -> str:
    return str(getattr(item, "summary", "") or "")


def _item_type(item: Any) -> str:
    return str(getattr(item, "item_type", "") or "")


def _has_qst(text: str) -> bool:
    return _has_any(text, ["QST", "ES", "志望動機", "自己PR"])


def _has_research(text: str, themes: list[str]) -> bool:
    return _has_any(text + "\n" + "\n".join(themes), ["研究", "月面", "宇宙", "機械学習", "ハイパースペクトル", "AI"])


def _has_any(text: str, needles: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _theme_phrase(themes: list[str]) -> str:
    useful = [theme for theme in themes if theme and not str(theme).startswith("quality:")]
    return "、".join(useful[:4])


def _compact_item_titles(items: list[Any], *, limit: int) -> str:
    titles = []
    for item in items:
        title = _clean_title(_item_title(item))
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    if not titles:
        return ""
    if len(titles) == 1:
        return titles[0]
    return "、".join(titles[:-1]) + "、" + titles[-1]


def _clean_title(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:80]


def _clean_summary(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[#>*\-\s・]+", "", text)
    if not text or _looks_like_raw_quote(text):
        return ""
    sentences = re.split(r"(?<=[。！？!?])\s*", text)
    cleaned = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > 130:
            continue
        cleaned = sentence
        break
    if not cleaned and len(text) <= 120:
        cleaned = text
    return cleaned.rstrip("…")


def _looks_like_raw_quote(text: str) -> bool:
    value = str(text or "")
    if len(value) > 160:
        return True
    if "…" in value or value.endswith("..."):
        return True
    if value.startswith("#") and len(value) < 60:
        return True
    return False


def _limit_sentences(sentences: list[str], limit: int) -> str:
    values = _dedupe([sentence.strip() for sentence in sentences if sentence.strip()])
    return "".join(values[:limit])


def _dedupe(values: list[str]) -> list[str]:
    output = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return output


def _month_label(month: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(month or ""))
    if not match:
        return str(month or "この月")
    return f"{int(match.group(1))}年{int(match.group(2))}月"
