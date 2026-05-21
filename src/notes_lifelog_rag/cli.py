from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from notes_lifelog_rag import __version__
from notes_lifelog_rag.analysis.service import (
    AnalysisSummary,
    analyze_sample,
    analyze_all,
    categorize_notes,
    extract_events,
    extract_thoughts,
    summarize_notes,
)
from notes_lifelog_rag.config import database_path, raw_notes_path
from notes_lifelog_rag.db.schema import connect, init_db, table_count
from notes_lifelog_rag.embeddings.engines import get_embedding_backend
from notes_lifelog_rag.embeddings.repository import build_chunk_embeddings
from notes_lifelog_rag.ingest.importer import ingest_directory
from notes_lifelog_rag.models.status import model_statuses
from notes_lifelog_rag.runtime.cuda import CudaStatus, collect_cuda_status
from notes_lifelog_rag.runtime.device import DeviceInfo, DeviceResolutionError, effective_dtype, resolve_device
from notes_lifelog_rag.search.hybrid import hybrid_search_notes
from notes_lifelog_rag.timeline.service import (
    build_monthly_reflection,
    build_timeline,
    format_month_timeline_markdown,
    format_reflection_markdown,
    format_timeline_report,
    format_timeline_markdown,
    generate_month_timeline_snapshot,
    generate_timeline_snapshots,
    get_month_sources,
    get_month_timeline_snapshot,
    list_month_timeline_snapshots,
    list_timeline_months,
    timeline_qa,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local-first Apple Notes lifelog RAG CLI.",
)
console = Console()


def main() -> None:
    app()


@app.callback()
def callback(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(f"notes_lifelog_rag {__version__}")
        raise typer.Exit()


@app.command("init-db")
def init_db_command(
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    db_path = init_db(db)
    console.print(f"[green]Initialized database:[/green] {db_path}")


@app.command("ingest-notes")
def ingest_notes_command(
    input_path: Annotated[Path | None, typer.Option("--input", "-i", help="Exported notes directory.")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    target = raw_notes_path(input_path)
    summary = ingest_directory(target, db)
    table = Table(title="Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Scanned files", str(summary.scanned_files))
    table.add_row("Imported notes", str(summary.imported_notes))
    table.add_row("Skipped duplicates", str(summary.skipped_duplicates))
    table.add_row("Skipped unsupported", str(summary.skipped_unsupported))
    table.add_row("Parser errors", str(summary.parser_errors))
    console.print(table)


@app.command("stats")
def stats_command(
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    init_db(db)
    with connect(db) as conn:
        table = Table(title=f"Database Stats: {database_path(db)}")
        table.add_column("Table")
        table.add_column("Rows", justify="right")
        for name in (
            "notes",
            "note_chunks",
            "categories",
            "note_categories",
            "note_summaries",
            "events",
            "thoughts",
            "suggestions",
            "monthly_reflections",
            "monthly_timeline_snapshots",
            "monthly_timeline_items",
            "model_runs",
            "chunk_embeddings",
            "import_errors",
        ):
            table.add_row(name, str(table_count(conn, name)))
        console.print(table)


@app.command("db-schema")
def db_schema_command(
    table: Annotated[str, typer.Option("--table", help="Table name to inspect.")] = "model_runs",
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    init_db(db)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        console.print("[red]Invalid table name.[/red]")
        raise typer.Exit(code=1)
    with connect(db) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
            (table,),
        ).fetchone()
        if not exists:
            console.print(f"[red]Table not found:[/red] {table}")
            raise typer.Exit(code=1)
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    output = Table(title=f"DB Schema: {table}")
    output.add_column("cid", justify="right")
    output.add_column("name")
    output.add_column("type")
    output.add_column("notnull")
    output.add_column("default")
    output.add_column("pk")
    for row in rows:
        output.add_row(
            str(row["cid"]),
            str(row["name"]),
            str(row["type"]),
            str(row["notnull"]),
            str(row["dflt_value"] or ""),
            str(row["pk"]),
        )
    console.print(output)


@app.command("analysis-failures")
def analysis_failures_command(
    task: Annotated[str | None, typer.Option("--task", help="Filter by task name.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum failed rows to show.")] = 50,
    group_by_error: Annotated[bool, typer.Option("--group-by-error", help="Group failures by task and error type.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    init_db(db)
    with connect(db) as conn:
        if group_by_error:
            params: tuple[object, ...] = (task,) if task else ()
            where = "WHERE success = 0"
            if task:
                where += " AND task_name = ?"
            rows = conn.execute(
                f"""
                SELECT task_name, COALESCE(NULLIF(error_type, ''), 'legacy_unknown_failure') AS error_type,
                       COUNT(*) AS count
                FROM model_runs
                {where}
                GROUP BY task_name, COALESCE(NULLIF(error_type, ''), 'legacy_unknown_failure')
                ORDER BY count DESC, task_name ASC
                """,
                params,
            ).fetchall()
            table = Table(title="Analysis Failures by Error")
            table.add_column("Task")
            table.add_column("Error type")
            table.add_column("Count", justify="right")
            for row in rows:
                table.add_row(row["task_name"], row["error_type"], str(row["count"]))
            console.print(table)
            return

        params = [task] if task else []
        where = "WHERE model_runs.success = 0"
        if task:
            where += " AND model_runs.task_name = ?"
        rows = conn.execute(
            f"""
            SELECT
                model_runs.task_name,
                model_runs.model_name,
                model_runs.note_id,
                notes.title,
                notes.source_relative_path AS source_path,
                model_runs.success,
                COALESCE(NULLIF(model_runs.error_type, ''), 'legacy_unknown_failure') AS error_type,
                COALESCE(
                    NULLIF(model_runs.error_message, ''),
                    'Failure was recorded before error diagnostics were added.'
                ) AS error_message,
                model_runs.created_at,
                model_runs.input_hash,
                model_runs.body_hash,
                model_runs.raw_output,
                model_runs.output_json
            FROM model_runs
            LEFT JOIN notes ON notes.id = model_runs.note_id
            {where}
            ORDER BY model_runs.created_at DESC
            LIMIT ?
            """,
            (*params, int(limit)),
        ).fetchall()
    table = Table(title="Analysis Failures")
    for column in [
        "Task",
        "Model",
        "Note ID",
        "Title",
        "Source",
        "Error type",
        "Error message",
        "Created",
        "Input hash",
        "Body hash",
        "Raw preview",
        "Output preview",
    ]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["task_name"]),
            str(row["model_name"]),
            _short_cell(row["note_id"], 12),
            _short_cell(row["title"], 28),
            _short_cell(row["source_path"], 32),
            str(row["error_type"]),
            _short_cell(row["error_message"], 80),
            str(row["created_at"] or ""),
            _short_cell(row["input_hash"], 12),
            _short_cell(row["body_hash"], 12),
            _short_cell(row["raw_output"], 500),
            _short_cell(row["output_json"], 500),
        )
    console.print(table)
    if rows:
        console.print("Failure previews:")
        for row in rows:
            console.print(
                "- "
                f"{row['task_name']} "
                f"{row['error_type']} "
                f"note={_short_cell(row['note_id'], 12) or '-'} "
                f"input={_short_cell(row['input_hash'], 12)} "
                f"message={_short_cell(row['error_message'], 120)}"
            )


@app.command("model-status")
def model_status_command(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Batch size hint for model status.")] = 16,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Generation token hint.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
) -> None:
    _ = (batch_size, max_new_tokens)
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    statuses = [
        status.to_dict()
        for status in model_statuses(
            requested_device=device,
            require_cuda=require_cuda or no_cpu_fallback,
            allow_cpu_fallback=not no_cpu_fallback,
            device_info=device_info,
        )
    ]
    if json_output:
        print(json.dumps(statuses, ensure_ascii=False, indent=2))
        return
    if show_device:
        _print_device_info(device_info, dtype=dtype)

    table = Table(title="Local Model Status")
    table.add_column("Purpose")
    table.add_column("Name")
    table.add_column("Path")
    table.add_column("Runtime")
    table.add_column("CUDA")
    table.add_column("Enabled")
    table.add_column("Reason")
    for status in statuses:
        path_label = "[green]exists[/green]" if status["path_exists"] else "[yellow]missing[/yellow]"
        runtime_label = (
            "[green]ready[/green]" if status["runtime_available"] else f"[yellow]{status['runtime_status']}[/yellow]"
        )
        cuda_label = _cuda_label(str(status["cuda_status"]))
        enabled_label = "[green]yes[/green]" if status["enabled"] else "[yellow]disabled[/yellow]"
        table.add_row(
            str(status["purpose"]),
            str(status["name"]),
            path_label,
            runtime_label,
            cuda_label,
            enabled_label,
            str(status["reason"]),
        )
    console.print(table)


@app.command("cuda-status")
def cuda_status_command(json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False) -> None:
    status = collect_cuda_status()
    if json_output:
        print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
        return
    _print_cuda_status(status)


@app.command("build-embeddings")
def build_embeddings_command(
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Maximum chunks to embed.")] = None,
    only_missing: Annotated[
        bool,
        typer.Option("--only-missing/--no-only-missing", help="Embed only chunks missing a successful embedding."),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Rebuild existing embeddings and ignore skip/cache checks.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be embedded without storing embeddings."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    model_name: Annotated[str | None, typer.Option("--model", help="Embedding model name from configs/models.yaml.")] = None,
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Embedding batch size.")] = 16,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Ignored for embeddings.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    allow_mock_fallback: Annotated[
        bool,
        typer.Option("--allow-mock-fallback", help="Use deterministic mock embeddings if local runtime is disabled."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    _ = max_new_tokens
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, model=model_name, batch_size=batch_size)
    selected_backend = get_embedding_backend(
        backend,
        model_name=model_name,
        allow_mock_fallback=allow_mock_fallback,
        device_info=device_info,
        dtype=dtype,
        batch_size=batch_size,
    )
    with _progress() as progress:
        task_id = progress.add_task("embedding chunks", total=1)
        summary = build_chunk_embeddings(
            selected_backend,
            db_path=db,
            limit=limit,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            batch_size=batch_size,
            progress_callback=lambda done, total, label: _update_progress(progress, task_id, done, total, label),
        )
    table = Table(title="Embedding Build Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Backend", summary.backend)
    table.add_row("Model", summary.model_name)
    table.add_row("Device", device_info.resolved_device)
    table.add_row("Dtype", effective_dtype(dtype, device_info))
    table.add_row("Batch size", str(batch_size))
    table.add_row("Scanned chunks", str(summary.scanned_chunks))
    table.add_row("Selected chunks", str(summary.selected_chunks))
    table.add_row("Skipped existing", str(summary.skipped_existing))
    table.add_row("Dry run", "yes" if summary.dry_run else "no")
    table.add_row("Would embed chunks", str(summary.would_embed_chunks))
    table.add_row("Embedded chunks", str(summary.embedded_chunks))
    table.add_row("Failed chunks", str(summary.failed_chunks))
    if summary.disabled_reason:
        table.add_row("Disabled reason", summary.disabled_reason)
    console.print(table)


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Keyword query.")],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=50, help="Maximum results.")] = 10,
    show_snippets: Annotated[
        bool, typer.Option("--show-snippets", help="Show short body snippets in terminal output.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON including snippets.")] = False,
    backend: Annotated[str | None, typer.Option("--backend", help="Set both embedding and reranker backend.")] = None,
    embedding_backend: Annotated[str, typer.Option("--embedding-backend", help="auto, local, mock, none.")] = "auto",
    reranker_backend: Annotated[str, typer.Option("--reranker-backend", help="auto, local, mock, none.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Embedding/reranker batch size.")] = 16,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Ignored for search.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    _ = max_new_tokens
    if backend:
        embedding_backend = backend
        reranker_backend = backend
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(
            device_info,
            dtype=dtype,
            backend=f"embedding={embedding_backend}, reranker={reranker_backend}",
            batch_size=batch_size,
        )
    results = hybrid_search_notes(
        query,
        limit=limit,
        db_path=db,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
        device_info=device_info,
        dtype=dtype,
        batch_size=batch_size,
    )
    if json_output:
        print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))
        return
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    table = Table(title=f"Search Results: {query}")
    table.add_column("Note ID")
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Score", justify="right")
    if show_snippets:
        table.add_column("Snippet")
    for result in results:
        row = [
            result.note_id[:12],
            result.title,
            result.source_relative_path,
            f"{result.score:.3f}",
        ]
        if show_snippets:
            row.append(result.snippet)
        table.add_row(*row)
    console.print(table)


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question to answer with retrieved evidence.")],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=20, help="Maximum evidence results.")] = 5,
    backend: Annotated[str | None, typer.Option("--backend", help="Set both embedding and reranker backend.")] = None,
    embedding_backend: Annotated[str, typer.Option("--embedding-backend", help="auto, local, mock, none.")] = "auto",
    reranker_backend: Annotated[str, typer.Option("--reranker-backend", help="auto, local, mock, none.")] = "none",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Embedding/reranker batch size.")] = 16,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Reserved for future generated answers.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    _ = max_new_tokens
    if backend:
        embedding_backend = backend
        reranker_backend = backend
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(
            device_info,
            dtype=dtype,
            backend=f"embedding={embedding_backend}, reranker={reranker_backend}",
            batch_size=batch_size,
        )
    results = hybrid_search_notes(
        question,
        limit=limit,
        db_path=db,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
        device_info=device_info,
        dtype=dtype,
        batch_size=batch_size,
    )
    if not results:
        console.print("[yellow]No evidence found.[/yellow]")
        return
    console.print("## Evidence-grounded Answer")
    console.print("検索された元メモ断片に基づく暫定回答です。根拠が弱い場合は断定していません。")
    console.print(f"\n{results[0].title} などのメモに、質問に関連する記録が残っている可能性があります。\n")
    console.print("### Evidence")
    for result in results:
        console.print(f"- `{result.note_id[:12]}` {result.title}: {result.snippet}")


@app.command("summarize-notes")
def summarize_notes_command(
    limit: Annotated[int | None, typer.Option("--limit", help="Limit notes to summarize.")] = None,
    all_notes: Annotated[
        bool, typer.Option("--all", help="Compatibility flag; omitted --limit already scans all eligible notes.")
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(
            "--only-missing/--no-only-missing",
            help="Process only notes without current task/model/content-hash/prompt-version output.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Regenerate outputs and ignore model_runs cache.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be processed without storing analysis outputs."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Analysis batch size hint.")] = 1,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, batch_size=batch_size, max_new_tokens=max_new_tokens)
    _print_analysis_summary(
        _run_analysis_with_progress(
            summarize_notes,
            db_path=db,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    )


@app.command("categorize-notes")
def categorize_notes_command(
    limit: Annotated[int | None, typer.Option("--limit", help="Limit notes to categorize.")] = None,
    all_notes: Annotated[
        bool, typer.Option("--all", help="Compatibility flag; omitted --limit already scans all eligible notes.")
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(
            "--only-missing/--no-only-missing",
            help="Process only notes without current task/model/content-hash/prompt-version output.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Regenerate outputs and ignore model_runs cache.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be processed without storing analysis outputs."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Analysis batch size hint.")] = 1,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, batch_size=batch_size, max_new_tokens=max_new_tokens)
    _print_analysis_summary(
        _run_analysis_with_progress(
            categorize_notes,
            db_path=db,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    )


@app.command("extract-events")
def extract_events_command(
    limit: Annotated[int | None, typer.Option("--limit", help="Limit notes to process.")] = None,
    all_notes: Annotated[
        bool, typer.Option("--all", help="Compatibility flag; omitted --limit already scans all eligible notes.")
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(
            "--only-missing/--no-only-missing",
            help="Process only notes without current task/model/content-hash/prompt-version output.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Regenerate outputs and ignore model_runs cache.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be processed without storing analysis outputs."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Analysis batch size hint.")] = 1,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, batch_size=batch_size, max_new_tokens=max_new_tokens)
    _print_analysis_summary(
        _run_analysis_with_progress(
            extract_events,
            db_path=db,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    )


@app.command("extract-thoughts")
def extract_thoughts_command(
    limit: Annotated[int | None, typer.Option("--limit", help="Limit notes to process.")] = None,
    all_notes: Annotated[
        bool, typer.Option("--all", help="Compatibility flag; omitted --limit already scans all eligible notes.")
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(
            "--only-missing/--no-only-missing",
            help="Process only notes without current task/model/content-hash/prompt-version output.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Regenerate outputs and ignore model_runs cache.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be processed without storing analysis outputs."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Analysis batch size hint.")] = 1,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, batch_size=batch_size, max_new_tokens=max_new_tokens)
    _print_analysis_summary(
        _run_analysis_with_progress(
            extract_thoughts,
            db_path=db,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    )


@app.command("analyze-all")
def analyze_all_command(
    limit: Annotated[int | None, typer.Option("--limit", help="Limit notes per task.")] = None,
    all_notes: Annotated[
        bool, typer.Option("--all", help="Compatibility flag; omitted --limit already scans all eligible notes.")
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(
            "--only-missing/--no-only-missing",
            help="Process only notes without current task/model/content-hash/prompt-version outputs.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Regenerate outputs and ignore model_runs cache.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be processed without storing analysis outputs."),
    ] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Analysis batch size hint.")] = 1,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 512,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, batch_size=batch_size, max_new_tokens=max_new_tokens)
    with _progress() as progress:
        task_id = progress.add_task("analysis", total=1)
        summaries = analyze_all(
            db_path=db,
            limit=limit,
            all_notes=all_notes,
            force=force,
            only_missing=only_missing,
            dry_run=dry_run,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_callback=lambda done, total, label: _update_progress(progress, task_id, done, total, label),
        )
    for summary in summaries:
        _print_analysis_summary(summary)


@app.command("analyze-sample")
def analyze_sample_command(
    task: Annotated[str, typer.Option("--task", help="summary, categories, events, or thoughts.")] = "summary",
    note_id: Annotated[str | None, typer.Option("--note-id", help="Analyze a specific note id.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum sample notes.")] = 1,
    show_prompt: Annotated[bool, typer.Option("--show-prompt", help="Show the prompt sent to the local backend.")] = False,
    show_raw_output: Annotated[bool, typer.Option("--show-raw-output", help="Show raw model output preview.")] = False,
    save: Annotated[bool, typer.Option("--save", help="Store the parsed output and model_run diagnostics.")] = False,
    backend: Annotated[str, typer.Option("--backend", help="auto, local, mock, or disabled.")] = "auto",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    dtype: Annotated[str, typer.Option("--dtype", help="auto, float32, float16, or bfloat16.")] = "auto",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens.")] = 1024,
    show_device: Annotated[bool, typer.Option("--show-device", help="Show resolved device details.")] = False,
    no_cpu_fallback: Annotated[
        bool,
        typer.Option("--no-cpu-fallback", help="Fail instead of falling back to CPU when CUDA is requested."),
    ] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    device_info = _resolve_cli_device(device, require_cuda, no_cpu_fallback, dtype)
    if show_device:
        _print_device_info(device_info, dtype=dtype, backend=backend, max_new_tokens=max_new_tokens)
    try:
        results = analyze_sample(
            task_name=task,
            db_path=db,
            note_id=note_id,
            limit=limit,
            backend_name=backend,
            device_info=device_info,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            save=save,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not results:
        console.print("[yellow]No sample notes found.[/yellow]")
        return
    for result in results:
        table = Table(title=f"Analyze Sample: {result.task_name} / {result.note_id[:12]}")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Title", result.title)
        table.add_row("Model", result.model_name)
        table.add_row("Success", "yes" if result.success else "no")
        table.add_row("Saved", "yes" if result.saved else "no")
        if result.error_type:
            table.add_row("Error type", result.error_type)
        if result.error_message:
            table.add_row("Error message", result.error_message)
        if result.parsed_json is not None:
            table.add_row("Parsed JSON", json.dumps(result.parsed_json, ensure_ascii=False, indent=2)[:2000])
        if show_prompt:
            table.add_row("Prompt", _short_cell(result.prompt, 2000))
        if show_raw_output:
            table.add_row("Raw output", _short_cell(result.raw_output, 2000))
        console.print(table)


@app.command("timeline")
def timeline_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    year: Annotated[str | None, typer.Option("--year", help="YYYY year for monthly cards.")] = None,
    monthly: Annotated[bool, typer.Option("--monthly", help="Show monthly memory cards.")] = False,
    all_months: Annotated[bool, typer.Option("--all-months", help="Show every month as memory cards.")] = False,
    rich: Annotated[bool, typer.Option("--rich", help="Show the month snapshot instead of raw items.")] = False,
    order: Annotated[str, typer.Option("--order", help="asc or desc.")] = "desc",
    limit: Annotated[int, typer.Option("--limit", help="Maximum timeline items.")] = 100,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    if rich or monthly or all_months or year:
        if month and not monthly and not all_months:
            snapshot = get_month_timeline_snapshot(month, db_path=db, generate_if_missing=True)
            console.print(format_month_timeline_markdown(snapshot) if snapshot else "Timeline snapshot not found.")
            return
        snapshots = list_month_timeline_snapshots(year=year, db_path=db, order=order, limit=limit if limit else None)
        console.print(format_timeline_report(snapshots, title=f"Timeline {year or 'All Months'}", order=order))
        return
    console.print(format_timeline_markdown(build_timeline(month, db_path=db, limit=limit), month=month))


@app.command("timeline-months")
def timeline_months_command(
    order: Annotated[str, typer.Option("--order", help="asc or desc.")] = "desc",
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    rows = list_timeline_months(db_path=db, order=order)
    table = Table(title="Timeline Months")
    table.add_column("Month")
    table.add_column("Notes", justify="right")
    table.add_column("Summaries", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Thoughts", justify="right")
    table.add_column("Suggestions", justify="right")
    table.add_column("Snapshot")
    table.add_column("Quality")
    for row in rows:
        table.add_row(
            row.month,
            str(row.notes_count),
            str(row.summaries_count),
            str(row.events_count),
            str(row.thoughts_count),
            str(row.suggestions_count),
            "yes" if row.has_snapshot else "no",
            row.quality,
        )
    console.print(table)


@app.command("generate-timeline")
def generate_timeline_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    all_months: Annotated[bool, typer.Option("--all-months", help="Generate every month.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace target month snapshots/items.")] = False,
    backend: Annotated[str, typer.Option("--backend", help="rule, local, or mock.")] = "rule",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, or cuda:N.")] = "auto",
    require_cuda: Annotated[bool, typer.Option("--require-cuda", help="Fail if CUDA cannot be used.")] = False,
    limit_months: Annotated[int | None, typer.Option("--limit-months", min=1, help="Limit month count.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Do not write timeline tables.")] = False,
    show_sources: Annotated[bool, typer.Option("--show-sources", help="Show source counts.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    if backend == "local":
        _resolve_cli_device(device, require_cuda, False, "auto")
    if not month and not all_months:
        console.print("[red]Specify --month or --all-months.[/red]")
        raise typer.Exit(code=1)
    months = [month] if month else None
    rows = months or [row.month for row in list_timeline_months(db_path=db, order="desc")]
    if limit_months:
        rows = rows[:limit_months]
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("generate timeline", total=len(rows))
        snapshots = generate_timeline_snapshots(
            months=months,
            all_months=all_months,
            limit_months=limit_months,
            db_path=db,
            backend=backend,
            force=force,
            dry_run=dry_run,
            progress_callback=lambda done, total, label: _update_progress(progress, task_id, done, total, label),
        )
    table = Table(title="Generated Timeline Snapshots" + (" (dry-run)" if dry_run else ""))
    table.add_column("Month")
    table.add_column("Title")
    table.add_column("Items", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Importance", justify="right")
    table.add_column("Warnings")
    for snapshot in snapshots:
        warnings = ", ".join(snapshot.quality.get("warnings") or [])
        table.add_row(
            snapshot.month,
            snapshot.title,
            str(len(snapshot.items)),
            f"{snapshot.confidence:.2f}",
            f"{snapshot.importance:.2f}",
            warnings or "none",
        )
        if show_sources:
            console.print(f"{snapshot.month} sources: {json.dumps(snapshot.source_counts, ensure_ascii=False)}")
    console.print(table)


@app.command("timeline-report")
def timeline_report_command(
    year: Annotated[str | None, typer.Option("--year", help="YYYY year.")] = None,
    all_years: Annotated[bool, typer.Option("--all-years", help="Include all years.")] = False,
    output: Annotated[Path, typer.Option("--output", help="Markdown output path.")] = Path("data/exports/timelines/timeline.md"),
    order: Annotated[str, typer.Option("--order", help="asc or desc.")] = "asc",
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    if not year and not all_years:
        console.print("[red]Specify --year or --all-years.[/red]")
        raise typer.Exit(code=1)
    snapshots = list_month_timeline_snapshots(year=None if all_years else year, db_path=db, order=order)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        format_timeline_report(snapshots, title=f"Timeline {year or 'All Years'}", order=order),
        encoding="utf-8",
    )
    console.print(f"[green]Wrote timeline report:[/green] {output}")


@app.command("timeline-qa")
def timeline_qa_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    all_months: Annotated[bool, typer.Option("--all-months", help="Check every month.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    if not month and not all_months:
        console.print("[red]Specify --month or --all-months.[/red]")
        raise typer.Exit(code=1)
    rows = timeline_qa(month=month, all_months=all_months, db_path=db)
    table = Table(title="Timeline QA")
    table.add_column("Month")
    table.add_column("Score", justify="right")
    table.add_column("Warnings")
    table.add_column("Source counts")
    table.add_column("Recommended action")
    for row in rows:
        table.add_row(
            row["month"],
            f"{float(row['quality_score']):.2f}",
            ", ".join(row["warnings"]) or "none",
            json.dumps(row["source_counts"], ensure_ascii=False),
            row["recommended_action"],
        )
    console.print(table)


@app.command("qa-report")
def qa_report_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum QA rows.")] = 50,
    output: Annotated[Path | None, typer.Option("--output", help="Write markdown report to path.")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    from notes_lifelog_rag.ui.services import qa_report

    report = qa_report(month=month, limit=limit, db_path=db)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        console.print(f"[green]Wrote QA report:[/green] {output}")
    else:
        console.print(report)


@app.command("generate-suggestions")
def generate_suggestions_command(
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum suggestions to generate.")] = 100,
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    today: Annotated[bool, typer.Option("--today", help="Prefer today rediscovery suggestions.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Regenerate matching suggestions.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    from notes_lifelog_rag.ui.services import generate_suggestions

    result = generate_suggestions(limit=limit, month=month, today=today, force=force, db_path=db)
    table = Table(title="Suggestions Generation")
    table.add_column("Metric")
    table.add_column("Value")
    for key in ["created", "skipped", "candidates", "limit"]:
        table.add_row(key, str(result.get(key, 0)))
    console.print(table)


@app.command("reflections")
def reflections_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Regenerate and upsert the monthly reflection.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    console.print(format_reflection_markdown(build_monthly_reflection(month, db_path=db, force=force)))


@app.command("generate-reflections")
def generate_reflections_command(
    month: Annotated[str | None, typer.Option("--month", help="YYYY-MM month.")] = None,
    all_months: Annotated[bool, typer.Option("--all-months", help="Generate reflections for every month in DB.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Regenerate existing reflection rows.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    from notes_lifelog_rag.ui.services import generate_reflections

    reports = generate_reflections(month=month, all_months=all_months, force=force, db_path=db)
    table = Table(title="Generated Reflections")
    table.add_column("Month")
    table.add_column("Confidence")
    table.add_column("Importance")
    table.add_column("Warnings")
    for report in reports:
        table.add_row(
            report.month,
            f"{report.confidence:.2f}",
            f"{report.importance:.2f}",
            str(len(report.quality_warnings)),
        )
    console.print(table)


@app.command("ui")
def ui_command(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 7860,
) -> None:
    if host != "127.0.0.1":
        console.print("[yellow]Warning:[/yellow] default privacy-safe host is 127.0.0.1.")
    from notes_lifelog_rag.ui.app import launch_ui

    launch_ui(host=host, port=port)


def _short_cell(value: object, limit: int = 80) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _resolve_cli_device(device: str, require_cuda: bool, no_cpu_fallback: bool, dtype: str) -> DeviceInfo:
    try:
        return resolve_device(
            device,
            require_cuda=require_cuda or no_cpu_fallback,
            allow_cpu_fallback=not no_cpu_fallback,
            dtype=dtype,
        )
    except DeviceResolutionError as exc:
        console.print(f"[red]Device error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_device_info(
    device_info: DeviceInfo,
    *,
    dtype: str,
    backend: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    max_new_tokens: int | None = None,
) -> None:
    table = Table(title="Resolved Runtime Device")
    table.add_column("Metric")
    table.add_column("Value")
    if backend:
        table.add_row("Backend", backend)
    if model:
        table.add_row("Model", model)
    table.add_row("Requested device", device_info.requested_device)
    table.add_row("Resolved device", device_info.resolved_device)
    table.add_row("CUDA available", "yes" if device_info.cuda_available else "no")
    table.add_row("Device count", str(device_info.device_count))
    table.add_row("Selected GPU", device_info.selected_device_name or "-")
    table.add_row("Selected GPU memory", device_info.selected_device_memory or "-")
    table.add_row("Dtype", effective_dtype(dtype, device_info))
    if batch_size is not None:
        table.add_row("Batch size", str(batch_size))
    if max_new_tokens is not None:
        table.add_row("Max new tokens", str(max_new_tokens))
    table.add_row("Reason", device_info.reason)
    if device_info.warning:
        table.add_row("Warning", device_info.warning)
    console.print(table)


def _print_cuda_status(status: CudaStatus) -> None:
    table = Table(title="CUDA Status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Python version", status.python_version)
    table.add_row("torch installed", "yes" if status.torch_installed else "no")
    table.add_row("torch version", status.torch_version or "-")
    table.add_row("torch.version.cuda", status.torch_cuda_version or "-")
    table.add_row("torch CUDA build", "yes" if status.torch_cuda_build else "no")
    table.add_row("torch.cuda.is_available()", "yes" if status.cuda_available else "no")
    table.add_row("torch.cuda.device_count()", str(status.device_count))
    table.add_row("CUDA_VISIBLE_DEVICES", status.cuda_visible_devices if status.cuda_visible_devices is not None else "-")
    table.add_row("nvidia-smi", status.nvidia_smi_path or "missing")
    table.add_row("nvidia-smi driver", status.nvidia_smi_driver_version or "-")
    table.add_row("nvidia-smi CUDA", status.nvidia_smi_cuda_version or "-")
    table.add_row("Likely reason", status.likely_reason)
    if status.init_error:
        table.add_row("Init error", status.init_error)
    if status.warnings:
        table.add_row("Warnings", "\n".join(status.warnings[:3]))
    console.print(table)

    gpu_table = Table(title="CUDA / nvidia-smi GPUs")
    gpu_table.add_column("Source")
    gpu_table.add_column("Index")
    gpu_table.add_column("Name")
    gpu_table.add_column("Capability")
    gpu_table.add_column("Memory")
    for device in status.devices:
        gpu_table.add_row(
            "torch",
            str(device.index),
            device.name,
            device.capability,
            _memory_label(device.memory_free_mb, device.memory_total_mb, free_first=True),
        )
    for gpu in status.nvidia_smi_gpus:
        gpu_table.add_row(
            "nvidia-smi",
            str(gpu.index),
            gpu.name,
            "-",
            _memory_label(gpu.memory_used_mb, gpu.memory_total_mb, free_first=False),
        )
    if status.devices or status.nvidia_smi_gpus:
        console.print(gpu_table)

    rec_table = Table(title="Recommended Actions")
    rec_table.add_column("Action")
    for recommendation in status.recommendations:
        rec_table.add_row(recommendation)
    rec_table.add_row("CPU fallback remains available unless --require-cuda or --no-cpu-fallback is used.")
    console.print(rec_table)


def _memory_label(first: int | None, total: int | None, *, free_first: bool) -> str:
    if total is None:
        return "-"
    label = "free" if free_first else "used"
    return f"{first if first is not None else 0}/{total} MB {label}"


def _cuda_label(value: str) -> str:
    if value == "available":
        return "[green]available[/green]"
    if value == "fallback_cpu":
        return "[yellow]fallback_cpu[/yellow]"
    if value == "unavailable":
        return "[yellow]unavailable[/yellow]"
    return value


def _print_analysis_summary(summary: AnalysisSummary) -> None:
    table = Table(title=f"Analysis Summary: {summary.task_name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Backend", summary.backend_name)
    table.add_row("Model", summary.model_name)
    table.add_row("Device", summary.device or "-")
    table.add_row("Total notes in DB", str(summary.total_notes))
    table.add_row("Scanned notes", str(summary.scanned_notes))
    table.add_row("Eligible notes", str(summary.eligible_notes))
    table.add_row(_already_label(summary.task_name), str(summary.skipped_existing))
    table.add_row("Selected notes", str(summary.selected_notes))
    table.add_row("Limit", str(summary.limit) if summary.limit is not None else "all")
    table.add_row("Force", "yes" if summary.force else "no")
    table.add_row("Cache hits", str(summary.cache_hits))
    table.add_row("Cached outputs used", str(summary.cached_notes))
    table.add_row("Cached empty results", str(summary.skipped_cached_empty))
    table.add_row("Dry run", "yes" if summary.dry_run else "no")
    table.add_row("Would process notes", str(summary.would_process_notes))
    table.add_row("Processed notes", str(summary.processed_notes))
    table.add_row("Failed notes", str(summary.failed_notes))
    table.add_row("Created items", str(summary.created_items))
    if summary.dry_run and summary.would_process_note_ids:
        preview = ", ".join(summary.would_process_note_ids[:20])
        suffix = "" if len(summary.would_process_note_ids) <= 20 else f" ... (+{len(summary.would_process_note_ids) - 20})"
        table.add_row(_would_ids_label(summary.task_name), f"{preview}{suffix}")
    if summary.disabled_reason:
        table.add_row("Disabled reason", summary.disabled_reason)
    console.print(table)


def _already_label(task_name: str) -> str:
    return {
        "summary": "Already summarized",
        "categories": "Already categorized",
        "events": "Already event-extracted",
        "thoughts": "Already thought-extracted",
    }.get(task_name, "Already processed")


def _would_ids_label(task_name: str) -> str:
    return {
        "summary": "would_process_summary_note_ids",
        "categories": "would_process_category_note_ids",
        "events": "would_process_event_note_ids",
        "thoughts": "would_process_thought_note_ids",
    }.get(task_name, "would_process_note_ids")


def _run_analysis_with_progress(command, **kwargs) -> AnalysisSummary:
    with _progress() as progress:
        task_id = progress.add_task("analysis", total=1)
        return command(
            **kwargs,
            progress_callback=lambda done, total, label: _update_progress(progress, task_id, done, total, label),
        )


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _update_progress(progress: Progress, task_id: int, done: int, total: int, label: str) -> None:
    display_total = max(total, 1)
    display_done = display_total if total == 0 else min(done, display_total)
    progress.update(task_id, description=label, total=display_total, completed=display_done)


if __name__ == "__main__":
    main()
