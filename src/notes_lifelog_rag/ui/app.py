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
                qa_output = gr.HTML(renderers.render_quality_warnings(services.get_quality_warnings(limit=50)))
                qa_btn.click(lambda m, l: renderers.render_quality_warnings(services.get_quality_warnings(month=m or None, limit=int(l))), inputs=[qa_month, qa_limit], outputs=[qa_output])

            with gr.Tab("Timeline"):
                gr.Markdown("## Timeline\nEvents and thoughts, ordered for monthly rediscovery.")
                with gr.Row():
                    timeline_month = gr.Dropdown(month_choices, value="", label="Month")
                    timeline_sort = gr.Dropdown(["date", "importance"], value="date", label="Sort")
                    timeline_limit = gr.Slider(20, 200, value=100, step=20, label="Limit")
                    timeline_btn = gr.Button("Refresh Timeline", variant="primary")
                timeline_output = gr.HTML(renderers.render_timeline_cards(services.get_timeline(limit=100)))
                timeline_btn.click(
                    lambda m, s, l: renderers.render_timeline_cards(services.get_timeline(m or None, sort=s, limit=int(l))),
                    inputs=[timeline_month, timeline_sort, timeline_limit],
                    outputs=[timeline_output],
                )

            with gr.Tab("Reflections"):
                gr.Markdown("## Reflections\nMonthly reflections built from thoughts first, events second, summaries third.")
                with gr.Row():
                    reflection_month = gr.Dropdown(month_choices, value=initial_reflection_month, label="Month")
                    reflection_force = gr.Checkbox(label="force regenerate", value=False)
                    reflection_btn = gr.Button("Generate / Refresh Reflection", variant="primary")
                reflection_list = gr.HTML(renderers.render_reflection_list(initial_reflections))
                reflection_output = gr.HTML(_stored_reflection_detail(initial_reflection_month, initial_reflections))
                reflection_btn.click(
                    lambda m, f: renderers.render_reflection(services.generate_reflections(month=m or None, force=bool(f))[0] if m else services.get_reflection(None)),
                    inputs=[reflection_month, reflection_force],
                    outputs=[reflection_output],
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
                suggestion_output = gr.HTML(renderers.render_suggestions(services.list_suggestions(limit=50)))
                suggestion_btn.click(
                    _generate_suggestions_tab,
                    inputs=[suggestion_limit, suggestion_month, suggestion_today, suggestion_force],
                    outputs=[suggestion_status, suggestion_output],
                )
                suggestion_refresh.click(
                    lambda l: renderers.render_suggestions(services.list_suggestions(limit=int(l))),
                    inputs=[suggestion_limit],
                    outputs=[suggestion_output],
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
    return status, renderers.render_suggestions(services.list_suggestions(limit=int(limit)))


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
