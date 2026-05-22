from __future__ import annotations

from html import escape
from typing import Any

from notes_lifelog_rag.timeline.service import (
    MonthTimelineSnapshot,
    ReflectionReport,
    TimelineItem,
    timeline_qa_problem_items,
    timeline_qa_severity_for_snapshot,
    timeline_item_display_groups,
    visible_timeline_flags,
)


def render_sidebar(state: dict[str, Any], *, active_scope: str = "全ノート") -> str:
    stats = state.get("stats", {})
    health = state.get("health", {})
    categories = state.get("categories", [])
    months = state.get("months", [])
    nav_items = [
        "All Notes",
        "Today Rediscovery",
        "Important",
        "Low Confidence",
        "Evidence Missing",
        "Timeline",
        "Reflections",
        "Suggestions",
        "Settings",
        "DB Stats",
    ]
    nav = "".join(
        f'<div class="sidebar-item {"active" if item == active_scope else ""}">{escape(item)}</div>' for item in nav_items
    )
    cat_html = "".join(
        f'<div class="sidebar-item compact"><span>{escape(row["name"])}</span><b>{int(row["note_count"])}</b></div>'
        for row in categories[:12]
    )
    month_html = "".join(
        f'<div class="sidebar-item compact"><span>{escape(row["month"])}</span><b>{int(row["note_count"])}</b></div>'
        for row in months[:12]
    )
    status = "".join(
        f'<span class="status-badge">{escape(label)} {int(stats.get(table, 0))}</span>'
        for label, table in [
            ("notes", "notes"),
            ("summaries", "note_summaries"),
            ("events", "events"),
            ("thoughts", "thoughts"),
            ("embeddings", "chunk_embeddings"),
        ]
    )
    return f"""
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div class="sidebar-title">Notes LifeLog</div>
        <div class="sidebar-subtitle">Local memory workspace</div>
      </div>
      <div class="sidebar-section">{nav}</div>
      <div class="sidebar-section"><div class="sidebar-heading">Analysis Health</div>
        <div class="mini-health">
          <span>summaries {int(health.get("summaries") or 0)}/{int(health.get("notes") or 0)}</span>
          <span>events {int(health.get("events") or 0)}</span>
          <span>thoughts {int(health.get("thoughts") or 0)}</span>
          <span>{escape(str(health.get("health_status") or "Unknown"))}</span>
        </div>
      </div>
      <div class="sidebar-section"><div class="sidebar-heading">Status</div><div class="status-wrap">{status}</div></div>
      <div class="sidebar-section"><div class="sidebar-heading">Categories</div>{cat_html or render_empty_state("No categories yet")}</div>
      <div class="sidebar-section"><div class="sidebar-heading">Months</div>{month_html or render_empty_state("No dated notes yet")}</div>
    </aside>
    """


def render_note_cards(notes: list[dict[str, Any]], *, selected_note_id: str | None = None) -> str:
    if not notes:
        return render_empty_state("条件に合うメモがまだありません。検索条件を変えるか、Import / Analyze を実行してください。")
    cards = []
    for note in notes:
        note_id = str(note["id"])
        title = note.get("display_title") or note.get("title") or "Untitled"
        choice = f"{note_id[:12]} · {title}"
        selected = "selected" if selected_note_id and note_id.startswith(selected_note_id) else ""
        badges = "".join(f'<span class="note-badge">{escape(cat)}</span>' for cat in note.get("categories", [])[:4])
        if note.get("important"):
            badges += '<span class="note-badge important">Important</span>'
        if note.get("low_confidence"):
            badges += '<span class="note-badge review">Low confidence</span>'
        if note.get("evidence_missing"):
            badges += '<span class="note-badge review">Evidence check</span>'
        score = ""
        if note.get("search_score") is not None:
            score = f'<span class="score-pill">{float(note["search_score"]):.3f} · {escape(str(note.get("search_source") or "search"))}</span>'
        snippet = note.get("search_snippet") or note.get("one_line_summary") or "分析要約はまだありません。"
        confidence = note.get("confidence")
        importance = note.get("importance")
        cards.append(
            f"""
            <article class="note-card {selected}" role="button" tabindex="0"
              data-note-choice="{escape(choice, quote=True)}"
              aria-label="Select note {escape(title, quote=True)}">
              <div class="note-title">{escape(title)}</div>
              <div class="note-snippet">{escape(snippet)}</div>
              <div class="note-meta">
                <span>{escape(note.get("date_label") or "date unknown")}</span>
                <span>{escape(note.get("source_short") or "")}</span>
              </div>
              <div class="note-card-footer">
                <div class="badge-row">{badges}</div>
                <div class="metric-row">
                  {render_confidence_pill(confidence)}
                  {render_importance_pill(importance)}
                  <span class="score-pill">events {int(note.get("event_count") or 0)}</span>
                  <span class="score-pill">thoughts {int(note.get("thought_count") or 0)}</span>
                  {score}
                </div>
              </div>
            </article>
            """
        )
    return '<section class="note-list">' + "".join(cards) + "</section>"


def render_note_detail(detail: dict[str, Any]) -> str:
    if not detail.get("found"):
        return render_empty_state(detail.get("message", "メモを選択してください。"))
    note = detail["note"]
    summary = detail.get("summary")
    warnings = "".join(render_warning_banner(_warning_label(warning)) for warning in detail.get("warnings", []))
    category_badges = "".join(f'<span class="note-badge">{escape(item["name"])}</span>' for item in detail.get("categories", []))
    body = escape(note.get("body") or "")
    model_runs = _render_model_runs(detail.get("model_runs") or [])
    reflection = _render_reflection_card(detail.get("reflection"))
    return f"""
    <section class="detail-pane">
      {warnings}
      <article class="paper">
        <div class="paper-kicker">{escape(note.get("source_relative_path") or "")}</div>
        <h1 class="paper-title">{escape(note.get("title") or "Untitled")}</h1>
        <div class="note-meta detail-meta">
          <span>{escape(note.get("note_date") or note.get("modified_at") or note.get("imported_at") or "date unknown")}</span>
          <span>hash {escape(str(note.get("content_hash") or "")[:12])}</span>
        </div>
        {render_summary_card(summary)}
        <div class="badge-row detail-categories">{category_badges}</div>
        {render_event_cards(detail.get("events") or [])}
        {render_thought_cards(detail.get("thoughts") or [])}
        {reflection}
        {model_runs}
        <section class="paper-section">
          <h2>Original Note</h2>
          <pre class="paper-body">{body}</pre>
        </section>
      </article>
    </section>
    """


def render_summary_card(summary: dict[str, Any] | None) -> str:
    if not summary:
        return render_warning_banner("まだsummaryがありません。analyze-allを実行すると右ペインが豊かになります。")
    points = "".join(f"<li>{escape(str(point))}</li>" for point in (summary.get("important_points") or [])[:5])
    return f"""
    <section class="ai-summary-card">
      <div class="section-title-row">
        <h2>{escape(summary.get("generated_title") or "AI Summary")}</h2>
        <div>{render_confidence_pill(summary.get("confidence"))}{render_importance_pill(summary.get("importance"))}</div>
      </div>
      <p class="summary-line">{escape(summary.get("one_line_summary") or "")}</p>
      <p>{escape(summary.get("detailed_summary") or "")}</p>
      <ul class="important-points">{points}</ul>
      <p class="revisit">{escape(summary.get("revisit_reason") or "")}</p>
      {render_evidence_card(summary.get("evidence") or [])}
    </section>
    """


def render_event_cards(events: list[dict[str, Any]]) -> str:
    if not events:
        return render_warning_banner("eventsがまだ少ないため、Timelineはnote title fallbackに寄る可能性があります。")
    items = []
    for event in events:
        items.append(
            f"""
            <article class="event-card">
              <div class="section-title-row">
                <h3>{escape(event.get("title") or "Event")}</h3>
                <div>{render_confidence_pill(event.get("confidence"))}{render_importance_pill(event.get("importance"))}</div>
              </div>
              <p>{escape(event.get("summary") or "")}</p>
              <div class="note-meta"><span>{escape(event.get("event_type") or "event")}</span><span>{escape(event.get("date_label") or event.get("event_date") or "date unknown")}</span></div>
              {render_evidence_card(event.get("evidence") or [])}
            </article>
            """
        )
    return '<section class="paper-section"><h2>Timeline Cards</h2>' + "".join(items) + "</section>"


def render_thought_cards(thoughts: list[dict[str, Any]]) -> str:
    if not thoughts:
        return render_warning_banner("thoughtsがまだ少ないため、内省やリフレクションは薄く表示されます。")
    items = []
    for thought in thoughts:
        themes = "".join(f'<span class="note-badge">{escape(str(theme))}</span>' for theme in thought.get("themes", [])[:4])
        items.append(
            f"""
            <article class="thought-card">
              <div class="section-title-row">
                <h3>{escape(thought.get("title") or "Thought")}</h3>
                <div>{render_confidence_pill(thought.get("confidence"))}{render_importance_pill(thought.get("importance"))}</div>
              </div>
              <p>{escape(thought.get("summary") or "")}</p>
              <div class="badge-row">{themes}</div>
              <p class="revisit">{escape(thought.get("remember_reason") or "")}</p>
              {render_evidence_card(thought.get("evidence") or [])}
            </article>
            """
        )
    return '<section class="paper-section"><h2>Reflections In This Note</h2>' + "".join(items) + "</section>"


def render_evidence_card(evidence: list[dict[str, Any]] | dict[str, Any] | None) -> str:
    if evidence is None:
        return '<div class="evidence-card warning">Evidence is missing.</div>'
    values = evidence if isinstance(evidence, list) else [evidence]
    if not values:
        return '<div class="evidence-card warning">Evidence is missing.</div>'
    rows = []
    for item in values[:3]:
        note_id = escape(str(item.get("note_id") or "")[:12])
        quote = escape(str(item.get("quote") or "")[:180])
        rows.append(f'<blockquote><span>{note_id}</span>{quote or "quote missing"}</blockquote>')
    return '<div class="evidence-card"><div class="evidence-label">Evidence</div>' + "".join(rows) + "</div>"


def render_category_badges(categories: list[dict[str, Any]] | list[str]) -> str:
    badges = []
    for item in categories:
        name = item.get("name") if isinstance(item, dict) else str(item)
        badges.append(f'<span class="note-badge">{escape(str(name))}</span>')
    return '<div class="badge-row">' + "".join(badges) + "</div>"


def render_reflection(reflection: ReflectionReport | dict[str, Any] | None) -> str:
    if reflection is None:
        return render_empty_state("Reflectionはまだありません。月を選択して生成してください。")
    if isinstance(reflection, ReflectionReport):
        month = reflection.month
        messages = reflection.reminder_messages
        evidence = reflection.evidence
        confidence = reflection.confidence
        importance = reflection.importance
        warnings = reflection.quality_warnings
        coverage = reflection.coverage
    else:
        summary = reflection.get("summary") or {}
        month = reflection.get("month") or summary.get("month") or "unknown"
        messages = summary.get("reminder_messages") or []
        evidence = reflection.get("evidence") or summary.get("evidence") or []
        confidence = reflection.get("confidence") or summary.get("confidence")
        importance = reflection.get("importance") or summary.get("importance")
        warnings = summary.get("quality_warnings") or []
        coverage = summary.get("coverage") or {}
    message_html = "".join(f"<li>{escape(str(message))}</li>" for message in messages[:5]) or "<li>まだ十分な材料がありません。</li>"
    warning_html = "".join(render_warning_banner(str(item)) for item in warnings[:3])
    return f"""
    <section class="ai-summary-card">
      <div class="section-title-row">
        <h2>{escape(str(month))} Reflection</h2>
        <div>{render_confidence_pill(confidence)}{render_importance_pill(importance)}</div>
      </div>
      <div class="note-meta"><span>notes {int(coverage.get("notes") or 0)}</span><span>events {int(coverage.get("event_notes") or 0)}</span><span>thoughts {int(coverage.get("thought_notes") or 0)}</span></div>
      {warning_html}
      <ul class="important-points">{message_html}</ul>
      {render_evidence_card(evidence)}
    </section>
    """


def render_reflection_list(reflections: list[dict[str, Any]], *, selected_note_id: str | None = None) -> str:
    if not reflections:
        return render_empty_state("Monthly reflections are not generated yet.")
    cards = []
    for item in reflections[:24]:
        summary = item.get("summary") or {}
        messages = summary.get("reminder_messages") or []
        preview = messages[0] if messages else "Reflection summary is available."
        note_id = _first_evidence_note_id(item.get("evidence") or summary.get("evidence") or [])
        title = f"{item.get('month') or 'unknown'} Reflection"
        selected = "selected" if selected_note_id and note_id.startswith(selected_note_id) else ""
        click_attrs = _note_click_attrs(note_id, title)
        cards.append(
            f"""
            <article class="thought-card note-card {selected}"{click_attrs}>
              <div class="section-title-row">
                <h3>{escape(title)}</h3>
                <div>{render_confidence_pill(item.get("confidence"))}{render_importance_pill(item.get("importance"))}</div>
              </div>
              <p>{escape(str(preview))}</p>
              <div class="note-meta"><span>updated {escape(item.get("updated_at") or "")}</span><span>{escape(note_id[:12])}</span></div>
              {render_evidence_card(item.get("evidence") or [])}
            </article>
            """
        )
    return '<section class="timeline-cards">' + "".join(cards) + "</section>"


def render_suggestions(suggestions: list[dict[str, Any]], *, selected_note_id: str | None = None) -> str:
    if not suggestions:
        return render_empty_state("Suggestions are not generated yet. Run generate-suggestions.")
    cards = []
    for item in suggestions:
        note_id = str(item.get("note_id") or "")
        title = str(item.get("title") or "Suggestion")
        choice = f"{note_id[:12]} · {title}" if note_id else ""
        selected = "selected" if selected_note_id and note_id.startswith(selected_note_id) else ""
        click_attrs = (
            f' role="button" tabindex="0" data-note-choice="{escape(choice, quote=True)}"'
            f' aria-label="Select suggestion source {escape(title, quote=True)}"'
            if note_id
            else ""
        )
        cards.append(
            f"""
            <article class="note-card suggestion-card {selected}"{click_attrs}>
              <div class="section-title-row">
                <h3>{escape(title)}</h3>
                <div>{render_importance_pill(item.get("importance"))}{render_confidence_pill(item.get("confidence"))}</div>
              </div>
              <div class="badge-row"><span class="note-badge important">{escape(item.get("suggestion_type") or "suggestion")}</span><span class="note-badge">{escape(item.get("status") or "new")}</span></div>
              <p>{escape(item.get("message") or "")}</p>
              <div class="note-meta"><span>{escape(item.get("target_date") or "")}</span><span>{escape(item.get("source_relative_path") or item.get("note_title") or "")}</span><span>{escape(str(item.get("note_id") or "")[:12])}</span></div>
              {render_evidence_card(item.get("evidence") or [])}
            </article>
            """
        )
    return '<section class="timeline-cards suggestion-list">' + "".join(cards) + "</section>"


def render_quality_warnings(warnings: list[dict[str, Any]], *, selected_note_id: str | None = None) -> str:
    if not warnings:
        return render_empty_state("QA warnings are clear for the current filter.")
    rows = []
    for item in warnings:
        note_id = str(item.get("note_id") or "")
        title = str(item.get("title") or item.get("warning_type") or "warning")
        selected = "selected" if selected_note_id and note_id.startswith(selected_note_id) else ""
        click_attrs = _note_click_attrs(note_id, title)
        rows.append(
            f"""
            <article class="event-card note-card {selected}"{click_attrs}>
              <div class="section-title-row">
                <h3>{escape(item.get("warning_type") or "warning")} · {escape(title)}</h3>
                <div>{render_confidence_pill(item.get("confidence"))}{render_importance_pill(item.get("importance"))}</div>
              </div>
              <p>{escape(item.get("issue") or "")}</p>
              <div class="note-meta"><span>{escape(note_id[:12])}</span><span>{escape(item.get("source_path") or "")}</span></div>
              {render_evidence_card(item.get("evidence") or [])}
            </article>
            """
        )
    return '<section class="timeline-cards">' + "".join(rows) + "</section>"


def render_model_status_badges(statuses: list[dict[str, Any]]) -> str:
    badges = []
    for status in statuses[:8]:
        enabled = "ok" if status.get("enabled") else "warn"
        badges.append(
            f'<span class="status-badge {enabled}">{escape(status.get("purpose", ""))}: {escape(status.get("runtime_status", ""))} / {escape(status.get("cuda_status", ""))}</span>'
        )
    return '<div class="status-wrap">' + "".join(badges) + "</div>"


def render_model_status(statuses: list[dict[str, Any]]) -> str:
    rows = []
    for status in statuses:
        rows.append(
            f"<tr><td>{escape(status.get('purpose', ''))}</td><td>{escape(status.get('name', ''))}</td>"
            f"<td>{escape(status.get('runtime_status', ''))}</td><td>{escape(status.get('cuda_status', ''))}</td>"
            f"<td>{'yes' if status.get('enabled') else 'disabled'}</td><td>{escape(status.get('reason', ''))}</td></tr>"
        )
    return '<table class="model-run-table"><tbody>' + "".join(rows) + "</tbody></table>"


def render_cuda_status(cuda: dict[str, Any]) -> str:
    gpu = cuda.get("gpus", [{}])[0] if cuda.get("gpus") else {}
    return f"""
    <section class="analysis-health">
      <div class="section-title-row"><h2>CUDA</h2><span class="status-badge {'ok' if cuda.get('cuda_available') else 'warn'}">{'available' if cuda.get('cuda_available') else 'unavailable'}</span></div>
      <div class="note-meta">
        <span>torch {escape(str(cuda.get("torch_version") or "-"))}</span>
        <span>cuda {escape(str(cuda.get("torch_cuda_version") or "-"))}</span>
        <span>gpu {escape(str(gpu.get("name") or "-"))}</span>
        <span>{escape(str(cuda.get("likely_reason") or ""))}</span>
      </div>
    </section>
    """


def render_db_stats_badges(stats: dict[str, int]) -> str:
    keys = [
        ("Notes", "notes"),
        ("Summaries", "note_summaries"),
        ("Events", "events"),
        ("Thoughts", "thoughts"),
        ("Embeddings", "chunk_embeddings"),
        ("Failures", "model_runs"),
    ]
    return '<div class="status-wrap">' + "".join(
        f'<span class="status-badge">{escape(label)} {int(stats.get(key, 0))}</span>' for label, key in keys
    ) + "</div>"


def render_analysis_health(health: dict[str, Any]) -> str:
    notes = max(int(health.get("notes") or 0), 1)
    status = str(health.get("health_status") or "Unknown")
    sparse = status in {"Sparse", "Needs Review", "Running Job"}
    message = "; ".join(health.get("warnings") or []) or ("Analysis coverage looks usable." if not sparse else "Analysis needs review.")
    rows = [
        ("Summaries", health.get("summaries", 0), notes),
        ("Category notes", health.get("unique_category_notes", 0), notes),
        ("Event notes", health.get("unique_event_notes", 0), notes),
        ("Thought notes", health.get("unique_thought_notes", 0), notes),
        ("Embedding chunks", health.get("embedding_chunks", 0), max(int(health.get("chunks") or 0), 1)),
    ]
    meters = "".join(_meter(label, int(value), int(total)) for label, value, total in rows)
    return f"""
    <section class="analysis-health">
      <div class="section-title-row"><h2>Analysis Health</h2><span class="status-badge {'warn' if sparse else 'ok'}">{escape(status)}</span></div>
      <div class="health-grid">{meters}</div>
      <div class="note-meta">
        <span>suggestions {int(health.get("suggestions") or 0)}</span>
        <span>reflections {int(health.get("monthly_reflections") or 0)}</span>
        <span>model failures {int(health.get("model_runs_failures") or 0)}</span>
        <span>import errors {int(health.get("import_errors") or 0)}</span>
        <span>low confidence {int(health.get("low_confidence_items") or 0)}</span>
        <span>evidence warnings {int(health.get("evidence_warnings") or 0)}</span>
      </div>
      <p class="muted">{escape(message)}</p>
    </section>
    """


def render_empty_state(message: str) -> str:
    return f'<div class="empty-state">{escape(message)}</div>'


def render_warning_banner(message: str) -> str:
    return f'<div class="warning-banner">{escape(message)}</div>'


def render_confidence_pill(value: Any) -> str:
    if value is None:
        return '<span class="confidence-pill muted">conf -</span>'
    number = float(value)
    tone = "low" if number <= 0.45 else "ok"
    return f'<span class="confidence-pill {tone}">conf {number:.2f}</span>'


def render_importance_pill(value: Any) -> str:
    if value is None:
        return '<span class="importance-pill muted">imp -</span>'
    number = float(value)
    tone = "high" if number >= 0.75 else "ok"
    return f'<span class="importance-pill {tone}">imp {number:.2f}</span>'


def render_timeline_cards(items: list[TimelineItem], *, selected_note_id: str | None = None) -> str:
    if not items:
        return render_empty_state("この月のtimeline候補はまだありません。")
    cards = []
    for item in items:
        selected = "selected" if selected_note_id and item.note_id.startswith(selected_note_id) else ""
        title = item.source_title or item.title
        click_attrs = _note_click_attrs(item.note_id, title)
        cards.append(
            f"""
            <article class="event-card note-card {selected}"{click_attrs}>
              <div class="section-title-row">
                <h3>{escape(item.date_label)} · {escape(item.title)}</h3>
                <div>{render_confidence_pill(item.confidence)}{render_importance_pill(item.importance)}</div>
              </div>
              <p>{escape(item.summary)}</p>
              <div class="note-meta"><span>{escape(item.item_type)}</span><span>{escape(item.source_title)}</span><span>{escape(item.note_id[:12])}</span></div>
              {render_evidence_card(item.evidence)}
            </article>
            """
        )
    return '<section class="timeline-cards">' + "".join(cards) + "</section>"


def render_timeline_month_cards(
    snapshots: list[MonthTimelineSnapshot],
    *,
    selected_month: str | None = None,
) -> str:
    if not snapshots:
        return render_empty_state("Timeline month cards are not available yet. Run generate-timeline or refresh the filters.")
    cards = []
    for snapshot in snapshots:
        selected = "selected" if selected_month == snapshot.month else ""
        severity = timeline_qa_severity_for_snapshot(snapshot)
        severe_badges = "".join(
            f'<span class="note-badge review">{escape(str(warning))}</span>'
            for warning in severity["severe_warnings"][:2]
        )
        review_badges = "".join(
            f'<span class="note-badge review">{escape(str(warning))}</span>'
            for warning in severity["review_warnings"][:3]
        )
        info_badges = "".join(
            f'<span class="note-badge">{escape(str(warning))}</span>'
            for warning in severity["info_warnings"][:2]
        )
        themes = "".join(f'<span class="note-badge">{escape(theme)}</span>' for theme in snapshot.key_themes[:5])
        counts = snapshot.source_counts
        qa_score = float(snapshot.quality.get("quality_score") or 0.0)
        cards.append(
            f"""
            <article class="timeline-month-card note-card {selected}" role="button" tabindex="0"
              data-month-choice="{escape(snapshot.month, quote=True)}"
              aria-label="Open timeline month {escape(snapshot.month, quote=True)}">
              <div class="section-title-row">
                <h3>{escape(snapshot.month)} · {escape(snapshot.title)}</h3>
                <div>{render_confidence_pill(snapshot.confidence)}{render_importance_pill(snapshot.importance)}<span class="score-pill">QA {qa_score:.2f}</span></div>
              </div>
              <p class="note-snippet">{escape(snapshot.overview)}</p>
              <p class="note-snippet"><b>Thoughts</b>: {escape(snapshot.thought_summary)}</p>
              <p class="note-snippet"><b>Events</b>: {escape(snapshot.event_summary)}</p>
              <div class="badge-row">{themes}{severe_badges}{review_badges}{info_badges}</div>
              <div class="note-meta">
                <span>notes {int(counts.get("notes", 0))}</span>
                <span>events {int(counts.get("events", 0))}</span>
                <span>thoughts {int(counts.get("thoughts", 0))}</span>
                <span>items {len(snapshot.items)}</span>
              </div>
            </article>
            """
        )
    return '<section class="note-list timeline-month-list">' + "".join(cards) + "</section>"


def render_month_timeline_detail(
    snapshot: MonthTimelineSnapshot | None,
    *,
    grouped: bool = True,
    show_low_priority: bool = False,
    low_priority_limit: int = 3,
) -> str:
    if snapshot is None:
        return render_empty_state("月を選択してください。")
    severity = timeline_qa_severity_for_snapshot(snapshot)
    review_warnings = severity["review_warnings"]
    severe_warnings = severity["severe_warnings"]
    info_warnings = severity["info_warnings"]
    warnings = "".join(render_warning_banner(str(item)) for item in (severe_warnings + review_warnings)[:5])
    info_badges = "".join(f'<span class="note-badge">{escape(str(item))}</span>' for item in info_warnings[:4])
    themes = "".join(f'<span class="note-badge">{escape(theme)}</span>' for theme in snapshot.key_themes)
    categories = "".join(f'<span class="note-badge important">{escape(cat)}</span>' for cat in snapshot.dominant_categories)
    changes = "".join(f"<li>{escape(item)}</li>" for item in snapshot.important_changes) or "<li>まだ十分な材料がありません。</li>"
    rediscovery = "".join(f"<li>{escape(item)}</li>" for item in snapshot.rediscovery_points + snapshot.revisit_reasons) or "<li>まだ十分な材料がありません。</li>"
    groups = timeline_item_display_groups(snapshot.items, grouped=grouped)
    problem_items = timeline_qa_problem_items(snapshot)
    main_items = groups["main"]
    suggestion_items = groups["suggestions"]
    low_priority_items = groups["low_priority"]
    main_cards = "".join(_render_month_timeline_item(item, grouped=grouped) for item in main_items[:60])
    suggestion_cards = "".join(_render_month_timeline_item(item, grouped=grouped) for item in suggestion_items[:20])
    low_priority_visible = low_priority_items if show_low_priority else low_priority_items[: max(0, int(low_priority_limit))]
    low_priority_cards = "".join(_render_month_timeline_item(item, low_priority=True, grouped=grouped) for item in low_priority_visible)
    low_priority_summary = f"{len(low_priority_items)} low priority / needs review items"
    low_priority_reasons = _reason_badges(low_priority_items)
    low_priority_note = "" if show_low_priority else '<p class="muted">show low priority をオンにすると全件表示します。</p>'
    qa_problem_html = _render_timeline_qa_problem_items(problem_items)
    mode_badge = "Grouped view" if grouped else "Ungrouped raw items"
    return f"""
    <section class="detail-pane">
      <article class="paper">
        <div class="paper-kicker">Monthly Timeline Snapshot</div>
        <h1 class="paper-title">{escape(snapshot.month)} · {escape(snapshot.title)}</h1>
        <div class="metric-row">{render_confidence_pill(snapshot.confidence)}{render_importance_pill(snapshot.importance)}<span class="score-pill">quality {float(snapshot.quality.get("quality_score") or 0.0):.2f}</span><span class="score-pill">{escape(mode_badge)}</span>{info_badges}</div>
        {warnings}
        <section class="ai-summary-card">
          <h2>この月の概要</h2>
          <p>{escape(snapshot.overview)}</p>
        </section>
        <section class="paper-section">
          <h2>この月に考えていたこと</h2>
          <p>{escape(snapshot.thought_summary)}</p>
        </section>
        <section class="paper-section">
          <h2>この月に起きたこと</h2>
          <p>{escape(snapshot.event_summary)}</p>
        </section>
        <section class="paper-section">
          <h2>重要テーマ</h2>
          <div class="badge-row">{categories}{themes}</div>
        </section>
        <section class="paper-section">
          <h2>重要な変化</h2>
          <ul class="important-points">{changes}</ul>
        </section>
        <section class="paper-section">
          <h2>見返す価値</h2>
          <ul class="important-points">{rediscovery}</ul>
        </section>
        {render_evidence_card(snapshot.evidence)}
        <section class="paper-section">
          <h2>Main Timeline Items</h2>
          <p class="muted">thought / event / note summary を優先して構成した、この月の中心素材です。</p>
          {main_cards or render_empty_state("この月の中心itemsはまだありません。")}
        </section>
        <section class="paper-section">
          <h2>Timeline QA</h2>
          <p class="muted">月別QA score、warnings、problem itemsです。source noteをクリックすると下にdetail previewを表示します。</p>
          {qa_problem_html}
        </section>
        <section class="paper-section">
          <h2>Supporting Suggestions</h2>
          <p class="muted">suggestionsは補助素材として最大件数を絞り、overviewの中心には使いません。</p>
          {suggestion_cards or render_empty_state("この月に紐づくsupporting suggestionsはありません。")}
        </section>
        <details class="paper-section low-priority-details">
          <summary>{escape(low_priority_summary)}</summary>
          <p class="muted">歌詞、買い物、PDFノイズ、title-only evidenceなど、月の意味を歪めやすい候補です。</p>
          <div class="badge-row">{low_priority_reasons}</div>
          {low_priority_note}
          {low_priority_cards or render_empty_state("低優先レビュー候補はありません。")}
        </details>
      </article>
    </section>
    """


def _render_month_timeline_item(item, *, low_priority: bool = False, grouped: bool = True) -> str:
    badges = "".join(f'<span class="note-badge">{escape(value)}</span>' for value in (item.categories + item.themes)[:5])
    flags = visible_timeline_flags(item, low_priority=low_priority, grouped=grouped)
    flag_badges = "".join(f'<span class="note-badge review">{escape(value)}</span>' for value in flags[:8])
    extra_class = " low-priority" if low_priority else ""
    summary = _timeline_item_summary(item)
    sub_count = len(getattr(item, "sub_items", []) or [])
    sub_meta = f"<span>sub items {sub_count}</span>" if sub_count else ""
    click_attrs = _note_click_attrs(item.source_note_id, item.title)
    review_attrs = _timeline_review_attrs(item.id, item.source_note_id, item.month)
    return f"""
    <article class="note-card event-card month-item-card timeline-review-target{extra_class}"{click_attrs}{review_attrs}>
      <div class="section-title-row">
        <h3>{escape(item.date_label or "日付不明")} · {escape(item.title)}</h3>
        <div>{render_confidence_pill(item.confidence)}{render_importance_pill(item.importance)}</div>
      </div>
      <p>{escape(summary)}</p>
      <div class="badge-row"><span class="note-badge important">{escape(item.item_type)}</span>{badges}</div>
      <div class="badge-row">{flag_badges}</div>
      <div class="note-meta"><span>{escape(item.source_note_id[:12])}</span><span>{escape(item.source_table)}</span>{sub_meta}</div>
      {render_evidence_card(item.evidence)}
    </article>
    """


def _reason_badges(items) -> str:
    counts: dict[str, int] = {}
    for item in items:
        for flag in getattr(item, "quality_flags", []) or []:
            counts[flag] = counts.get(flag, 0) + 1
    values = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:6]
    return "".join(f'<span class="note-badge review">{escape(flag)} {count}</span>' for flag, count in values)


def _render_timeline_qa_problem_items(problem_items: dict[str, list[dict[str, Any]]]) -> str:
    if not problem_items:
        return render_empty_state("この月のTimeline QA problem itemsはありません。")
    sections = []
    for warning, items in problem_items.items():
        rows = []
        for item in items[:8]:
            title = str(item.get("title") or "Untitled")
            note_id = str(item.get("source_note_id") or "")
            item_id = str(item.get("item_id") or "")
            month = str(item.get("month") or "")
            flags = item.get("quality_flags") or []
            flag_badges = "".join(f'<span class="note-badge review">{escape(str(flag))}</span>' for flag in flags[:8])
            click_attrs = _note_click_attrs(note_id, title)
            review_attrs = _timeline_review_attrs(item_id, note_id, month)
            rows.append(
                f"""
                <article class="note-card event-card qa-problem-item timeline-review-target"{click_attrs}{review_attrs}>
                  <div class="section-title-row">
                    <h3>{escape(title)}</h3>
                    <span class="note-badge important">{escape(str(item.get("item_type") or ""))}</span>
                  </div>
                  <div class="badge-row">{flag_badges}</div>
                  <p class="muted">{escape(str(item.get("reason") or ""))}</p>
                  <div class="note-meta"><span>{escape(note_id[:12])}</span><span>{escape(item_id[:12])}</span><span>{escape(str(warning))}</span></div>
                </article>
                """
            )
        sections.append(
            f"""
            <details class="qa-warning-group" open>
              <summary>{escape(str(warning))} · {len(items)} items</summary>
              {''.join(rows)}
            </details>
            """
        )
    return '<section class="timeline-qa-problems">' + "".join(sections) + "</section>"


def _timeline_item_summary(item) -> str:
    summary = str(getattr(item, "summary", "") or "").strip()
    title = str(getattr(item, "title", "") or "").strip()
    if summary and not (summary.startswith("#") and len(summary) <= max(40, len(title) + 10)):
        return summary
    for row in getattr(item, "evidence", []) or []:
        quote = str(row.get("quote") or "").strip() if isinstance(row, dict) else ""
        if len(quote) >= 16 and quote != title:
            return quote
    return summary or title


def _note_click_attrs(note_id: str, title: str) -> str:
    if not note_id:
        return ""
    choice = f"{note_id[:12]} · {title or 'Note'}"
    return (
        f' role="button" tabindex="0" data-note-choice="{escape(choice, quote=True)}"'
        f' aria-label="Open source note {escape(title or "Note", quote=True)}"'
    )


def _timeline_review_attrs(item_id: str, source_note_id: str, month: str) -> str:
    if not item_id:
        return ""
    return (
        f' data-review-item-id="{escape(str(item_id), quote=True)}"'
        f' data-review-note-id="{escape(str(source_note_id or ""), quote=True)}"'
        f' data-review-month="{escape(str(month or ""), quote=True)}"'
    )


def _first_evidence_note_id(evidence: list[dict[str, Any]] | dict[str, Any] | None) -> str:
    if evidence is None:
        return ""
    values = evidence if isinstance(evidence, list) else [evidence]
    for item in values:
        note_id = str(item.get("note_id") or "") if isinstance(item, dict) else ""
        if note_id:
            return note_id
    return ""


def _render_reflection_card(reflection: dict[str, Any] | None) -> str:
    if not reflection:
        return ""
    return '<section class="paper-section"><h2>Related Monthly Reflection</h2>' + render_reflection(reflection) + "</section>"


def _render_model_runs(model_runs: list[dict[str, Any]]) -> str:
    if not model_runs:
        return render_warning_banner("current prompt/versionのmodel run情報はまだありません。")
    rows = []
    for row in model_runs[:8]:
        status = "success" if row.get("success") else "failed"
        rows.append(
            f"<tr><td>{escape(str(row.get('task_name') or ''))}</td><td>{escape(status)}</td>"
            f"<td>{escape(str(row.get('model_name') or ''))}</td><td>{escape(str(row.get('prompt_version') or ''))}</td>"
            f"<td>{escape(str(row.get('error_type') or ''))}</td></tr>"
        )
    return '<section class="paper-section"><h2>Model Runs</h2><table class="model-run-table"><tbody>' + "".join(rows) + "</tbody></table></section>"


def _meter(label: str, value: int, total: int) -> str:
    pct = 0 if total <= 0 else min(100, int(value / total * 100))
    return f"""
    <div class="health-meter">
      <div class="note-meta"><span>{escape(label)}</span><span>{value}/{total}</span></div>
      <div class="meter-track"><div class="meter-fill" style="width:{pct}%"></div></div>
    </div>
    """


def _warning_label(value: str) -> str:
    return {
        "summary_missing": "summaryが未生成です。",
        "events_sparse": "eventsが少ないため、timelineは弱くなる可能性があります。",
        "thoughts_sparse": "thoughtsが少ないため、reflectionは弱くなる可能性があります。",
        "low_confidence": "低confidenceのAI出力があります。原文確認を推奨します。",
        "evidence_missing": "evidenceが弱い、または不足しているAI出力があります。",
    }.get(value, value)
