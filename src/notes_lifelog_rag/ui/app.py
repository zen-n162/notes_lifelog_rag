from __future__ import annotations

from notes_lifelog_rag.config import raw_notes_path
from notes_lifelog_rag.ui import renderers, services
from notes_lifelog_rag.ui.styles import APP_CSS

try:
    from gradio.events import EventData
except ModuleNotFoundError:  # pragma: no cover - keeps import lightweight when UI extra is absent.
    class EventData:  # type: ignore[no-redef]
        pass


SORT_CHOICES = ["updated_desc", "created_desc", "importance_desc", "confidence_asc", "title"]
SCOPE_CHOICES = [
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
NOTE_CARD_CLICK_JS = """
element.addEventListener('click', (event) => {
  const monthCard = event.target.closest('.timeline-month-card[data-month-choice]');
  if (monthCard && element.contains(monthCard)) {
    event.preventDefault();
    trigger('click', {month_choice: monthCard.getAttribute('data-month-choice')});
    return;
  }
  const card = event.target.closest('.note-card[data-note-choice]');
  if (!card || !element.contains(card)) {
    return;
  }
  event.preventDefault();
  trigger('click', {note_choice: card.getAttribute('data-note-choice')});
});

element.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') {
    return;
  }
  const monthCard = event.target.closest('.timeline-month-card[data-month-choice]');
  if (monthCard && element.contains(monthCard)) {
    event.preventDefault();
    trigger('click', {month_choice: monthCard.getAttribute('data-month-choice')});
    return;
  }
  const card = event.target.closest('.note-card[data-note-choice]');
  if (!card || !element.contains(card)) {
    return;
  }
  event.preventDefault();
  trigger('click', {note_choice: card.getAttribute('data-note-choice')});
});
"""


def create_app():
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - covered by CLI behavior.
        raise RuntimeError("Gradio is not installed. Run: pip install -e '.[ui]'") from exc

    initial_state = services.get_sidebar_state()
    initial_notes = services.list_notes(limit=60)
    initial_note_choices = services.note_choices(initial_notes)
    initial_selected = initial_note_choices[0] if initial_note_choices else None
    initial_detail = (
        renderers.render_note_detail(services.get_note_detail(initial_selected))
        if initial_selected
        else renderers.render_empty_state("Import notes to start building your private memory workspace.")
    )
    category_choices = [""] + [row["name"] for row in initial_state["categories"]]
    month_choices = [""] + [row["month"] for row in initial_state["months"]]
    initial_reflection_month = month_choices[1] if len(month_choices) > 1 else ""
    initial_reflections = services.list_reflections()
    initial_qa_warnings = services.get_quality_warnings(limit=50)
    initial_qa_selected = _first_warning_choice(initial_qa_warnings)
    initial_timeline_items = services.get_timeline(limit=100)
    initial_timeline_selected = _first_timeline_choice(initial_timeline_items)
    initial_timeline_years = [""] + services.list_timeline_years()
    initial_timeline_year = initial_timeline_years[1] if len(initial_timeline_years) > 1 else ""
    initial_timeline_month_snapshots = services.get_timeline_month_snapshots(year=initial_timeline_year or None, limit=24)
    initial_timeline_month_selected = initial_timeline_month_snapshots[0].month if initial_timeline_month_snapshots else ""
    initial_reflection_selected = _first_reflection_choice(initial_reflections)
    initial_suggestions = services.list_suggestions(limit=50)
    initial_suggestion_selected = _first_suggestion_choice(initial_suggestions)

    with gr.Blocks(title="Notes LifeLog", elem_id="notes-root") as demo:
        with gr.Tabs(selected="Notes Workspace"):
            with gr.Tab("Notes Workspace", id="Notes Workspace"):
                with gr.Row(elem_classes=["top-toolbar"]):
                    with gr.Column(scale=2, min_width=230):
                        gr.HTML(
                            '<div><div class="toolbar-title">Notes LifeLog</div>'
                            '<div class="toolbar-subtitle">Notes-like local memory app. Private, evidence-first, 127.0.0.1 by default.</div></div>'
                        )
                    global_search = gr.Textbox(
                        label="Global search",
                        placeholder="研究、旅行、面接、当時考えていたこと...",
                        scale=4,
                    )
                    refresh_btn = gr.Button("Refresh", variant="primary", scale=1)
                    toolbar_import_btn = gr.Button("Import", scale=1)
                    toolbar_analysis_btn = gr.Button("Analyze Missing", scale=1)
                    toolbar_embedding_btn = gr.Button("Build Embeddings", scale=1)
                    toolbar_model_btn = gr.Button("Model Status", scale=1)
                toolbar_message = gr.HTML(renderers.render_db_stats_badges(initial_state["stats"]))
                health_html = gr.HTML(renderers.render_analysis_health(initial_state["health"]))

                with gr.Row(elem_classes=["workspace-grid"]):
                    with gr.Column(scale=1, min_width=245, elem_classes=["sidebar-shell"]):
                        scope = gr.Radio(
                            SCOPE_CHOICES,
                            value="All Notes",
                            label="Library",
                            elem_classes=["scope-selector"],
                        )
                        category = gr.Dropdown(category_choices, value="", label="Category")
                        month = gr.Dropdown(month_choices, value="", label="Month")
                        sidebar_html = gr.HTML(renderers.render_sidebar(initial_state))

                    with gr.Column(scale=2, min_width=360, elem_classes=["note-list-shell"]):
                        with gr.Column(elem_classes=["note-list-toolbar"]):
                            with gr.Row():
                                sort = gr.Dropdown(SORT_CHOICES, value="updated_desc", label="Sort", scale=2)
                                limit = gr.Slider(10, 300, value=60, step=10, label="Limit", scale=2)
                            with gr.Row():
                                has_summary = gr.Checkbox(
                                    label="has summary",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                                has_events = gr.Checkbox(
                                    label="has events",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                                has_thoughts = gr.Checkbox(
                                    label="has thoughts",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                            with gr.Row():
                                low_confidence = gr.Checkbox(
                                    label="low confidence",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                                evidence_missing = gr.Checkbox(
                                    label="evidence missing",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                                favorite = gr.Checkbox(
                                    label="important / revisit",
                                    value=False,
                                    elem_classes=["note-filter-toggle"],
                                )
                                search_btn = gr.Button("Search / Filter", variant="primary")
                        notes_state = gr.State(initial_notes)
                        note_cards = gr.HTML(
                            renderers.render_note_cards(
                                initial_notes,
                                selected_note_id=services.extract_note_id(initial_selected),
                            ),
                            js_on_load=NOTE_CARD_CLICK_JS,
                        )

                    with gr.Column(scale=3, min_width=520, elem_classes=["detail-shell"]):
                        detail_html = gr.HTML(initial_detail)
                        with gr.Accordion("Ask With Evidence", open=False):
                            ask_box = gr.Textbox(label="Question", placeholder="この時期、研究について何を考えていた？")
                            ask_btn = gr.Button("Ask", variant="primary")
                            ask_answer = gr.HTML(renderers.render_empty_state("Search evidence will appear here."))

                refresh_inputs = [
                    scope,
                    global_search,
                    category,
                    month,
                    sort,
                    limit,
                    has_summary,
                    has_events,
                    has_thoughts,
                    low_confidence,
                    evidence_missing,
                    favorite,
                ]
                refresh_outputs = [
                    sidebar_html,
                    toolbar_message,
                    health_html,
                    note_cards,
                    detail_html,
                    notes_state,
                ]
                for trigger in [
                    refresh_btn.click,
                    search_btn.click,
                    scope.change,
                    category.change,
                    month.change,
                    sort.change,
                    has_summary.change,
                    has_events.change,
                    has_thoughts.change,
                    low_confidence.change,
                    evidence_missing.change,
                    favorite.change,
                ]:
                    trigger(_refresh_workspace, inputs=refresh_inputs, outputs=refresh_outputs)
                note_cards.click(
                    _select_workspace_note_from_card,
                    inputs=[notes_state, scope],
                    outputs=[note_cards, detail_html],
                    show_progress="hidden",
                )
                ask_btn.click(_ask_workspace, inputs=[ask_box, global_search], outputs=[ask_answer])
                toolbar_import_btn.click(_quick_import, outputs=[toolbar_message, sidebar_html, health_html])
                toolbar_analysis_btn.click(_job_hint, outputs=[toolbar_message])
                toolbar_embedding_btn.click(_embedding_hint, outputs=[toolbar_message])
                toolbar_model_btn.click(_model_status_workspace, outputs=[detail_html])

            with gr.Tab("Import"):
                gr.Markdown("## Import\nExported note files stay local. Re-importing is idempotent.")
                with gr.Row():
                    input_path = gr.Textbox(label="Apple Notes export directory", value=str(raw_notes_path()), scale=4)
                    init_btn = gr.Button("Initialize DB", variant="primary", scale=1)
                    import_btn = gr.Button("Import Notes", variant="primary", scale=1)
                import_status = gr.Markdown()
                stats_table = gr.Dataframe(
                    headers=["Table", "Rows"],
                    datatype=["str", "number"],
                    value=services.db_stats_rows(),
                    interactive=False,
                )
                import_errors_table = gr.Dataframe(
                    headers=["Created", "Parser", "Source", "Reason"],
                    value=services.import_error_rows(),
                    interactive=False,
                    label="Import errors",
                )
                init_btn.click(services.initialize_database, outputs=[import_status, stats_table])
                import_btn.click(_import_notes_ui, inputs=[input_path], outputs=[import_status, stats_table, import_errors_table])

            with gr.Tab("Analysis Jobs"):
                gr.Markdown("## Analysis Jobs\nHeavy local jobs run only when you press a button.")
                running_jobs = gr.HTML(renderers.render_warning_banner(_running_jobs_text()))
                with gr.Row():
                    job_backend = gr.Dropdown(["mock", "auto", "local", "disabled"], value="mock", label="Backend")
                    job_device = gr.Dropdown(["auto", "cpu", "cuda", "cuda:0"], value="auto", label="Device")
                    job_dtype = gr.Dropdown(["auto", "float32", "float16", "bfloat16"], value="auto", label="Dtype")
                with gr.Row():
                    job_limit = gr.Textbox(label="Limit (blank = all eligible)", value="10")
                    job_month = gr.Dropdown(month_choices, value="", label="Month")
                    job_batch = gr.Slider(1, 32, value=1, step=1, label="Batch size")
                    job_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max new tokens")
                with gr.Row():
                    job_only_missing = gr.Checkbox(label="only missing", value=True)
                    job_force = gr.Checkbox(label="force", value=False)
                    job_dry_run = gr.Checkbox(label="dry run", value=True)
                with gr.Row():
                    emb_dry_btn = gr.Button("Build Embeddings Dry Run")
                    emb_run_btn = gr.Button("Build Embeddings Run", variant="primary")
                    analysis_dry_btn = gr.Button("Analyze All Dry Run")
                    analysis_run_btn = gr.Button("Analyze All Run", variant="primary")
                    sug_btn = gr.Button("Generate Suggestions")
                    refl_btn = gr.Button("Generate Reflections")
                job_output = gr.Textbox(label="Job output", lines=12, interactive=False)
                emb_dry_btn.click(
                    _run_embeddings_job,
                    inputs=[job_limit, job_backend, job_only_missing, job_force, job_device, job_dtype, job_batch],
                    outputs=[job_output],
                )
                emb_run_btn.click(
                    _run_embeddings_job_real,
                    inputs=[job_limit, job_backend, job_only_missing, job_force, job_device, job_dtype, job_batch],
                    outputs=[job_output],
                )
                analysis_dry_btn.click(
                    _run_analysis_job,
                    inputs=[job_limit, job_backend, job_only_missing, job_force, job_device, job_dtype, job_batch, job_tokens],
                    outputs=[job_output],
                )
                analysis_run_btn.click(
                    _run_analysis_job_real,
                    inputs=[job_limit, job_backend, job_only_missing, job_force, job_device, job_dtype, job_batch, job_tokens],
                    outputs=[job_output],
                )
                sug_btn.click(
                    _run_suggestions_job,
                    inputs=[job_limit, job_month, job_force],
                    outputs=[job_output],
                )
                refl_btn.click(
                    _run_reflections_job,
                    inputs=[job_month, job_force],
                    outputs=[job_output],
                )

            with gr.Tab("QA Review"):
                gr.Markdown("## QA Review\nReview low-confidence outputs, evidence gaps, model failures, unknown dates, and parser errors.")
                with gr.Row():
                    qa_month = gr.Dropdown(month_choices, value="", label="Month")
                    qa_limit = gr.Slider(10, 200, value=50, step=10, label="Limit")
                    qa_btn = gr.Button("Generate QA Report", variant="primary")
                qa_state = gr.State(initial_qa_warnings)
                with gr.Row():
                    with gr.Column(scale=2, min_width=420, elem_classes=["note-list-shell"]):
                        qa_output = gr.HTML(
                            renderers.render_quality_warnings(
                                initial_qa_warnings,
                                selected_note_id=services.extract_note_id(initial_qa_selected),
                            ),
                            js_on_load=NOTE_CARD_CLICK_JS,
                        )
                    with gr.Column(scale=3, min_width=520, elem_classes=["detail-shell"]):
                        qa_detail = gr.HTML(renderers.render_note_detail(services.get_note_detail(initial_qa_selected)))
                qa_btn.click(_refresh_qa_tab, inputs=[qa_month, qa_limit], outputs=[qa_output, qa_detail, qa_state])
                qa_output.click(_select_qa_card, inputs=[qa_state], outputs=[qa_output, qa_detail], show_progress="hidden")

            with gr.Tab("Timeline"):
                gr.Markdown("## Timeline\nMonthly memory cards that combine thoughts, events, summaries, categories, and evidence.")
                with gr.Row():
                    timeline_year = gr.Dropdown(initial_timeline_years, value=initial_timeline_year, label="Year")
                    timeline_month = gr.Dropdown(month_choices, value=initial_timeline_month_selected, label="Month")
                    timeline_category = gr.Dropdown(category_choices, value="", label="Category")
                    timeline_theme = gr.Textbox(label="Theme", placeholder="研究、就活、内省...")
                with gr.Row():
                    timeline_item_type = gr.Dropdown(["all", "thoughts", "events", "note_summaries", "suggestions"], value="all", label="Item type")
                    timeline_sort = gr.Dropdown(["chronological_asc", "chronological_desc", "importance_desc"], value="chronological_desc", label="Sort")
                    timeline_limit = gr.Slider(5, 60, value=24, step=1, label="Months")
                    timeline_force = gr.Checkbox(label="force regenerate", value=False)
                    timeline_dry_run = gr.Checkbox(label="dry run", value=True)
                    timeline_btn = gr.Button("Refresh Timeline", variant="primary")
                    timeline_generate_btn = gr.Button("Generate / Refresh Month")
                timeline_state = gr.State(initial_timeline_month_snapshots)
                timeline_status = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=2, min_width=420, elem_classes=["note-list-shell"]):
                        timeline_output = gr.HTML(
                            renderers.render_timeline_month_cards(
                                initial_timeline_month_snapshots,
                                selected_month=initial_timeline_month_selected,
                            ),
                            js_on_load=NOTE_CARD_CLICK_JS,
                        )
                    with gr.Column(scale=3, min_width=520, elem_classes=["detail-shell"]):
                        timeline_detail = gr.HTML(
                            renderers.render_month_timeline_detail(
                                initial_timeline_month_snapshots[0] if initial_timeline_month_snapshots else None
                            )
                        )
                timeline_btn.click(
                    _refresh_timeline_tab,
                    inputs=[timeline_year, timeline_month, timeline_category, timeline_theme, timeline_item_type, timeline_sort, timeline_limit],
                    outputs=[timeline_output, timeline_detail, timeline_state, timeline_status],
                )
                timeline_generate_btn.click(
                    _generate_timeline_month_tab,
                    inputs=[timeline_month, timeline_force, timeline_dry_run],
                    outputs=[timeline_output, timeline_detail, timeline_state, timeline_status],
                )
                timeline_output.click(_select_timeline_month_card, inputs=[timeline_state], outputs=[timeline_output, timeline_detail], show_progress="hidden")

            with gr.Tab("Reflections"):
                gr.Markdown("## Reflections\nMonthly reflections built from thoughts first, events second, summaries third.")
                with gr.Row():
                    reflection_month = gr.Dropdown(month_choices, value=initial_reflection_month, label="Month")
                    reflection_force = gr.Checkbox(label="force regenerate", value=False)
                    reflection_btn = gr.Button("Generate / Refresh Reflection", variant="primary")
                reflection_state = gr.State(initial_reflections)
                with gr.Row():
                    with gr.Column(scale=2, min_width=420, elem_classes=["note-list-shell"]):
                        reflection_list = gr.HTML(
                            renderers.render_reflection_list(
                                initial_reflections,
                                selected_note_id=services.extract_note_id(initial_reflection_selected),
                            ),
                            js_on_load=NOTE_CARD_CLICK_JS,
                        )
                        reflection_output = gr.HTML(_stored_reflection_detail(initial_reflection_month, initial_reflections))
                    with gr.Column(scale=3, min_width=520, elem_classes=["detail-shell"]):
                        reflection_detail = gr.HTML(renderers.render_note_detail(services.get_note_detail(initial_reflection_selected)))
                reflection_btn.click(
                    _refresh_reflection_tab,
                    inputs=[reflection_month, reflection_force],
                    outputs=[reflection_list, reflection_output, reflection_detail, reflection_state],
                )
                reflection_list.click(
                    _select_reflection_card,
                    inputs=[reflection_state],
                    outputs=[reflection_list, reflection_detail],
                    show_progress="hidden",
                )

            with gr.Tab("Suggestions"):
                gr.Markdown("## Suggestions\nRule-based rediscovery suggestions generated from important thoughts, events, summaries, and QA warnings.")
                with gr.Row():
                    suggestion_month = gr.Dropdown(month_choices, value="", label="Month")
                    suggestion_limit = gr.Slider(10, 200, value=50, step=10, label="Limit")
                    suggestion_today = gr.Checkbox(label="today rediscovery", value=False)
                    suggestion_force = gr.Checkbox(label="force", value=False)
                    suggestion_btn = gr.Button("Generate Suggestions", variant="primary")
                    suggestion_refresh = gr.Button("Refresh")
                suggestion_status = gr.Markdown()
                suggestion_state = gr.State(initial_suggestions)
                with gr.Row():
                    with gr.Column(scale=2, min_width=420, elem_classes=["note-list-shell"]):
                        suggestion_output = gr.HTML(
                            renderers.render_suggestions(
                                initial_suggestions,
                                selected_note_id=services.extract_note_id(initial_suggestion_selected),
                            ),
                            js_on_load=NOTE_CARD_CLICK_JS,
                        )
                    with gr.Column(scale=3, min_width=520, elem_classes=["detail-shell"]):
                        suggestion_detail = gr.HTML(renderers.render_note_detail(services.get_note_detail(initial_suggestion_selected)))
                suggestion_btn.click(
                    _generate_suggestions_tab,
                    inputs=[suggestion_limit, suggestion_month, suggestion_today, suggestion_force],
                    outputs=[suggestion_status, suggestion_output, suggestion_detail, suggestion_state],
                )
                suggestion_refresh.click(
                    _refresh_suggestions_tab,
                    inputs=[suggestion_limit],
                    outputs=[suggestion_output, suggestion_detail, suggestion_state],
                )
                suggestion_output.click(
                    _select_suggestion_card,
                    inputs=[suggestion_state],
                    outputs=[suggestion_output, suggestion_detail],
                    show_progress="hidden",
                )

            with gr.Tab("Models / Settings"):
                settings_md = gr.Markdown(services.settings_markdown())
                model_badges = gr.HTML(renderers.render_model_status_badges(services.get_model_status()))
                model_table = gr.Dataframe(
                    headers=["Purpose", "Name", "Path", "Runtime", "CUDA", "Enabled", "Reason"],
                    value=services.model_rows(),
                    interactive=False,
                )
                settings_stats = gr.HTML(renderers.render_db_stats_badges(services.get_db_stats()))
                settings_health = gr.HTML(renderers.render_analysis_health(services.get_analysis_health()))
                settings_cuda = gr.HTML(renderers.render_cuda_status(services.get_cuda_status_summary()))
                settings_jobs = gr.HTML(renderers.render_warning_banner(_running_jobs_text()))
                refresh_models = gr.Button("Refresh Models / Settings", variant="primary")
                refresh_models.click(
                    _settings_refresh,
                    outputs=[settings_md, model_badges, model_table, settings_stats, settings_health, settings_cuda, settings_jobs],
                )
    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860, *, share: bool = False, prevent_thread_lock: bool = False):
    app = create_app()
    return app.launch(
        server_name=host,
        server_port=port,
        share=share,
        prevent_thread_lock=prevent_thread_lock,
        css=APP_CSS,
    )


def _refresh_workspace(
    scope: str,
    query: str,
    category: str,
    month: str,
    sort: str,
    limit: int,
    has_summary: bool,
    has_events: bool,
    has_thoughts: bool,
    low_confidence: bool,
    evidence_missing: bool,
    favorite: bool,
):
    try:
        state = services.get_sidebar_state()
        sidebar = renderers.render_sidebar(state, active_scope=scope)
        badges = renderers.render_db_stats_badges(state["stats"])
        health = renderers.render_analysis_health(state["health"])
        if scope == "Suggestions":
            suggestions = services.list_suggestions(limit=int(limit))
            selected = _first_suggestion_choice(suggestions)
            selected_id = services.extract_note_id(selected)
            rendered_cards = renderers.render_suggestions(suggestions, selected_note_id=selected_id)
            detail = renderers.render_note_detail(services.get_note_detail(selected))
            return sidebar, badges, health, rendered_cards, detail, suggestions
        notes = _workspace_notes(
            scope,
            query=query,
            category=category or None,
            month=month or None,
            sort=sort,
            limit=int(limit),
            has_summary=has_summary,
            has_events=has_events,
            has_thoughts=has_thoughts,
            low_confidence=low_confidence,
            evidence_missing=evidence_missing,
            favorite=favorite,
        )
        choices = services.note_choices(notes)
        selected = choices[0] if choices else None
        detail = _workspace_detail(scope, selected, month or None)
        selected_id = services.extract_note_id(selected)
        rendered_cards = renderers.render_note_cards(notes, selected_note_id=selected_id)
        return sidebar, badges, health, rendered_cards, detail, notes
    except Exception as exc:
        message = renderers.render_warning_banner(f"Workspace refresh failed: {exc}")
        return message, message, message, renderers.render_empty_state("refresh failed"), message, []


def _workspace_notes(
    scope: str,
    *,
    query: str,
    category: str | None,
    month: str | None,
    sort: str,
    limit: int,
    has_summary: bool,
    has_events: bool,
    has_thoughts: bool,
    low_confidence: bool,
    evidence_missing: bool,
    favorite: bool,
) -> list[dict]:
    if scope == "Today Rediscovery":
        return services.today_rediscovery(limit=limit)
    if scope == "Important":
        favorite = True
    if scope == "Low Confidence":
        low_confidence = True
    if scope == "Evidence Missing":
        evidence_missing = True
    if query.strip():
        return services.search_notes(query, backend="auto", device="auto", limit=limit)
    return services.list_notes(
        category=category,
        month=month,
        query=query,
        sort=sort,
        limit=limit,
        has_summary=has_summary,
        has_events=has_events,
        has_thoughts=has_thoughts,
        low_confidence=low_confidence,
        evidence_missing=evidence_missing,
        favorite=favorite,
    )


def _workspace_detail(scope: str, selected: str | None, month: str | None) -> str:
    if scope == "Timeline":
        items = services.get_timeline(month, limit=80)
        return renderers.render_timeline_cards(items)
    if scope == "Reflections":
        return _stored_reflection_detail(month, services.list_reflections())
    if scope == "Settings":
        return _model_status_html()
    if scope == "DB Stats":
        return renderers.render_analysis_health(services.get_analysis_health()) + renderers.render_db_stats_badges(services.get_db_stats())
    return renderers.render_note_detail(services.get_note_detail(selected))


def _select_workspace_note(choice: str | None, notes: list[dict] | None, scope: str | None = None):
    try:
        selected_id = services.extract_note_id(choice)
        if scope == "Suggestions":
            return (
                renderers.render_suggestions(notes or [], selected_note_id=selected_id),
                renderers.render_note_detail(services.get_note_detail(choice)),
            )
        return (
            renderers.render_note_cards(notes or [], selected_note_id=selected_id),
            renderers.render_note_detail(services.get_note_detail(choice)),
        )
    except Exception as exc:
        return renderers.render_empty_state("selection failed"), renderers.render_warning_banner(str(exc))


def _select_workspace_note_from_card(notes: list[dict] | None, scope: str | None = None, evt: EventData = None):
    choice = None
    if evt is not None:
        choice = getattr(evt, "note_choice", None)
        if choice is None:
            data = getattr(evt, "_data", {}) or {}
            choice = data.get("note_choice")
    return _select_workspace_note(choice, notes, scope)


def _refresh_qa_tab(month: str | None, limit: int):
    warnings = services.get_quality_warnings(month=month or None, limit=int(limit))
    selected = _first_warning_choice(warnings)
    selected_id = services.extract_note_id(selected)
    return (
        renderers.render_quality_warnings(warnings, selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(selected)),
        warnings,
    )


def _select_qa_card(warnings: list[dict] | None, evt: EventData = None):
    choice = _event_note_choice(evt)
    selected_id = services.extract_note_id(choice)
    return (
        renderers.render_quality_warnings(warnings or [], selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(choice)),
    )


def _refresh_timeline_tab(year, month, category, theme, item_type, sort, limit):
    snapshots = services.get_timeline_month_snapshots(
        year=year or None,
        category=category or None,
        theme=theme or None,
        item_type=item_type or "all",
        sort=sort or "chronological_desc",
        limit=int(limit),
    )
    selected_month = month if month and any(snapshot.month == month for snapshot in snapshots) else (snapshots[0].month if snapshots else None)
    selected = _snapshot_by_month(snapshots, selected_month)
    status = f"{len(snapshots)} monthly timeline cards"
    return (
        renderers.render_timeline_month_cards(snapshots, selected_month=selected_month),
        renderers.render_month_timeline_detail(selected),
        snapshots,
        status,
    )


def _generate_timeline_month_tab(month: str | None, force: bool, dry_run: bool):
    status, snapshot = services.generate_timeline_snapshot_ui(month=month or None, force=bool(force), dry_run=bool(dry_run))
    snapshots = [snapshot] if snapshot else []
    return (
        renderers.render_timeline_month_cards(snapshots, selected_month=snapshot.month if snapshot else None),
        renderers.render_month_timeline_detail(snapshot),
        snapshots,
        status,
    )


def _select_timeline_month_card(snapshots: list | None, evt: EventData = None):
    month = _event_month_choice(evt)
    selected = _snapshot_by_month(snapshots or [], month)
    return (
        renderers.render_timeline_month_cards(snapshots or [], selected_month=month),
        renderers.render_month_timeline_detail(selected),
    )


def _refresh_reflection_tab(month: str | None, force: bool):
    selected_month = month or None
    if selected_month:
        reflection_html = renderers.render_reflection(
            services.generate_reflections(month=selected_month, force=bool(force))[0]
        )
    else:
        reflection_html = renderers.render_reflection(services.get_reflection(None))
    reflections = services.list_reflections()
    selected = _first_reflection_choice(reflections)
    selected_id = services.extract_note_id(selected)
    return (
        renderers.render_reflection_list(reflections, selected_note_id=selected_id),
        reflection_html,
        renderers.render_note_detail(services.get_note_detail(selected)),
        reflections,
    )


def _select_reflection_card(reflections: list[dict] | None, evt: EventData = None):
    choice = _event_note_choice(evt)
    selected_id = services.extract_note_id(choice)
    return (
        renderers.render_reflection_list(reflections or [], selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(choice)),
    )


def _refresh_suggestions_tab(limit: int):
    suggestions = services.list_suggestions(limit=int(limit))
    selected = _first_suggestion_choice(suggestions)
    selected_id = services.extract_note_id(selected)
    return (
        renderers.render_suggestions(suggestions, selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(selected)),
        suggestions,
    )


def _select_suggestion_card(suggestions: list[dict] | None, evt: EventData = None):
    choice = _event_note_choice(evt)
    selected_id = services.extract_note_id(choice)
    return (
        renderers.render_suggestions(suggestions or [], selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(choice)),
    )


def _ask_workspace(question: str, fallback_query: str) -> str:
    query = question.strip() or fallback_query.strip()
    result = services.ask_with_evidence(query, backend="auto", device="auto", limit=8)
    notes_html = renderers.render_note_cards(result["evidence_notes"]) if result["evidence_notes"] else ""
    return (
        '<section class="ai-summary-card">'
        '<div class="section-title-row"><h2>Evidence-grounded Answer</h2>'
        f'{renderers.render_confidence_pill(result.get("confidence"))}</div>'
        f'<p>{result["answer"]}</p>{notes_html}</section>'
    )


def _quick_import():
    message, _rows = services.import_notes()
    state = services.get_sidebar_state()
    return renderers.render_warning_banner(message), renderers.render_sidebar(state), renderers.render_analysis_health(state["health"])


def _import_notes_ui(input_path):
    message, rows = services.import_notes(input_path)
    return message, rows, services.import_error_rows()


def _job_hint() -> str:
    return renderers.render_warning_banner("Analysis Jobs tabでbackend/device/dry-runを確認してから実行してください。")


def _embedding_hint() -> str:
    return renderers.render_warning_banner("Analysis Jobs tabでdry-runしてからembedding buildを実行してください。")


def _model_status_workspace() -> str:
    return _model_status_html()


def _model_status_html() -> str:
    return (
        '<section class="detail-pane"><article class="paper">'
        '<h1 class="paper-title">Models / Runtime</h1>'
        f'{renderers.render_model_status_badges(services.get_model_status())}'
        f'{renderers.render_analysis_health(services.get_analysis_health())}'
        f'{renderers.render_cuda_status(services.get_cuda_status_summary())}'
        f'{renderers.render_db_stats_badges(services.get_db_stats())}'
        '</article></section>'
    )


def _run_embeddings_job(limit, backend, only_missing, force, device, dtype, batch_size):
    return services.run_build_embeddings(
        limit,
        backend,
        dry_run=True,
        only_missing=bool(only_missing),
        force=bool(force),
        device=device,
        dtype=dtype,
        batch_size=int(batch_size),
    )


def _run_embeddings_job_real(limit, backend, only_missing, force, device, dtype, batch_size):
    return services.run_build_embeddings(
        limit,
        backend,
        dry_run=False,
        only_missing=bool(only_missing),
        force=bool(force),
        device=device,
        dtype=dtype,
        batch_size=int(batch_size),
    )


def _run_analysis_job(limit, backend, only_missing, force, device, dtype, batch_size, max_new_tokens):
    return services.run_analyze(
        limit,
        backend,
        dry_run=True,
        only_missing=bool(only_missing),
        force=bool(force),
        device=device,
        dtype=dtype,
        batch_size=int(batch_size),
        max_new_tokens=int(max_new_tokens),
    )


def _run_analysis_job_real(limit, backend, only_missing, force, device, dtype, batch_size, max_new_tokens):
    return services.run_analyze(
        limit,
        backend,
        dry_run=False,
        only_missing=bool(only_missing),
        force=bool(force),
        device=device,
        dtype=dtype,
        batch_size=int(batch_size),
        max_new_tokens=int(max_new_tokens),
    )


def _run_suggestions_job(limit, month, force):
    return services.run_generate_suggestions(limit=limit, month=month or None, today=False, force=bool(force))


def _run_reflections_job(month, force):
    return services.run_generate_reflections(month=month or None, all_months=not bool(month), force=bool(force))


def _generate_suggestions_tab(limit, month, today, force):
    result = services.generate_suggestions(limit=int(limit), month=month or None, today=bool(today), force=bool(force))
    status = f"created={result['created']}, skipped={result['skipped']}, candidates={result['candidates']}"
    suggestions = services.list_suggestions(limit=int(limit))
    selected = _first_suggestion_choice(suggestions)
    selected_id = services.extract_note_id(selected)
    return (
        status,
        renderers.render_suggestions(suggestions, selected_note_id=selected_id),
        renderers.render_note_detail(services.get_note_detail(selected)),
        suggestions,
    )


def _event_note_choice(evt: EventData = None) -> str | None:
    if evt is None:
        return None
    choice = getattr(evt, "note_choice", None)
    if choice is None:
        data = getattr(evt, "_data", {}) or {}
        choice = data.get("note_choice")
    return choice


def _event_month_choice(evt: EventData = None) -> str | None:
    if evt is None:
        return None
    choice = getattr(evt, "month_choice", None)
    if choice is None:
        data = getattr(evt, "_data", {}) or {}
        choice = data.get("month_choice")
    return choice


def _snapshot_by_month(snapshots: list | None, month: str | None):
    if not snapshots:
        return None
    for snapshot in snapshots:
        if month and getattr(snapshot, "month", None) == month:
            return snapshot
    return snapshots[0]


def _first_warning_choice(warnings: list[dict]) -> str | None:
    for item in warnings:
        note_id = str(item.get("note_id") or "")
        if note_id:
            title = item.get("title") or item.get("warning_type") or "QA warning"
            return f"{note_id[:12]} · {title}"
    return None


def _first_timeline_choice(items: list) -> str | None:
    for item in items:
        note_id = str(getattr(item, "note_id", "") or "")
        if note_id:
            title = getattr(item, "source_title", None) or getattr(item, "title", None) or "Timeline note"
            return f"{note_id[:12]} · {title}"
    return None


def _first_reflection_choice(reflections: list[dict]) -> str | None:
    for item in reflections:
        summary = item.get("summary") or {}
        evidence = item.get("evidence") or summary.get("evidence") or []
        for evidence_item in evidence:
            note_id = str(evidence_item.get("note_id") or "")
            if note_id:
                return f"{note_id[:12]} · {item.get('month') or 'Reflection'}"
    return None


def _first_suggestion_choice(suggestions: list[dict]) -> str | None:
    for item in suggestions:
        note_id = str(item.get("note_id") or "")
        if note_id:
            return f"{note_id[:12]} · {item.get('title') or 'Suggestion'}"
    return None


def _running_jobs_text() -> str:
    jobs = services.get_running_jobs()
    if jobs.get("analyze_all"):
        return "analyze-all is running. UI heavy jobs are disabled until it finishes."
    return "No analyze-all job detected."


def _stored_reflection_detail(month: str | None, reflections: list[dict] | None = None) -> str:
    values = reflections if reflections is not None else services.list_reflections()
    if not values:
        return renderers.render_empty_state("Monthly reflections are not generated yet. Use Generate / Refresh Reflection.")
    if month:
        for item in values:
            if item.get("month") == month:
                return renderers.render_reflection(item)
    return renderers.render_reflection(values[0])


def _settings_refresh():
    return (
        services.settings_markdown(),
        renderers.render_model_status_badges(services.get_model_status()),
        services.model_rows(),
        renderers.render_db_stats_badges(services.get_db_stats()),
        renderers.render_analysis_health(services.get_analysis_health()),
        renderers.render_cuda_status(services.get_cuda_status_summary()),
        renderers.render_warning_banner(_running_jobs_text()),
    )
