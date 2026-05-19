# notes_lifelog_rag

Local-first Apple Notes / iPhone Notes lifelog RAG application.

This MVP ingests exported note files, stores them in a local SQLite database,
builds optional local embeddings, performs hybrid search, and generates
evidence-aware analysis outputs. It does not call external APIs, download
models, upload data, or require API keys.

## Privacy Policy

- Note contents, summaries, embeddings, logs, and metadata stay local.
- The app does not use hosted LLMs, cloud embeddings, telemetry, or remote logs.
- The default UI host is `127.0.0.1`.
- CLI list views avoid printing full note bodies. Full text inspection should be
  an explicit user action in later UI work.

## Setup

```bash
cd ~/MyApplication/notes_lifelog_rag
conda create -n notes_lifelog_rag python=3.11 pip -y
conda activate notes_lifelog_rag
pip install -e ".[dev]"
```

An equivalent environment definition is provided in `environment.yml`.

## Local Model Paths

Model paths are configured in `configs/models.yaml` and checked by:

```bash
python -m notes_lifelog_rag.cli model-status
```

`model-status` reports both path availability and runtime availability.
If a model path or runtime dependency is missing, that model is disabled and
unrelated commands keep working.

The local model wrappers use `local_files_only=True` and set offline
Transformers environment flags before loading. They never download model files.
Heavy packages such as `torch`, `transformers`, or `sentence-transformers` are
optional runtime dependencies. Tests use mock backends and do not load models.
When you want to enable real local model inference in this conda environment,
install the optional runtime extra:

```bash
pip install -e ".[models]"
```

The Gradio UI is in the `ui` optional extra:

```bash
pip install -e ".[ui]"
```

## CUDA / GPU Execution

Check the local CUDA/PyTorch/driver state first:

```bash
python -m notes_lifelog_rag.cli cuda-status
```

`cuda-status` reports Python, PyTorch, `torch.version.cuda`,
`torch.cuda.is_available()`, visible CUDA devices, `nvidia-smi` driver/CUDA
summary, likely failure reasons, and recommended actions. If PyTorch prints a
warning such as `The NVIDIA driver on your system is too old`, the app can
diagnose it but cannot fix the driver mismatch. PyTorch CUDA builds and the
installed NVIDIA driver must be compatible. CPU fallback remains available.

Device options used by model commands:

- `--device auto`: use `cuda:0` when CUDA works, otherwise CPU fallback.
- `--device cpu`: force CPU.
- `--device cuda` / `--device cuda:0`: request CUDA.
- `--require-cuda`: stop with a clear error if CUDA cannot be used.
- `--no-cpu-fallback`: stop instead of falling back to CPU.
- `--dtype auto`: CUDA prefers bfloat16 when supported, otherwise float16; CPU uses float32.
- `--dtype float32|float16|bfloat16`: explicitly request a dtype.

Recommended 877-note differential update flow:

```bash
python -m notes_lifelog_rag.cli cuda-status

python -m notes_lifelog_rag.cli analyze-all \
  --backend local \
  --only-missing \
  --device auto \
  --dtype auto \
  --batch-size 2 \
  --max-new-tokens 512 \
  --show-device \
  --dry-run

python -m notes_lifelog_rag.cli build-embeddings \
  --backend local \
  --only-missing \
  --device auto \
  --batch-size 16 \
  --show-device

python -m notes_lifelog_rag.cli analyze-all \
  --backend local \
  --only-missing \
  --device auto \
  --dtype auto \
  --batch-size 2 \
  --max-new-tokens 512 \
  --show-device
```

To require CUDA and fail rather than falling back:

```bash
python -m notes_lifelog_rag.cli build-embeddings \
  --backend local \
  --only-missing \
  --device cuda \
  --require-cuda \
  --show-device
```

If GPU memory is tight, lower `--batch-size`, lower `--max-new-tokens`, build
embeddings first, and split `analyze-all` with `--limit`. All analysis commands
still use `--only-missing`, `model_runs` cache, and per-note commits so they can
resume safely after interruption.

## Apple Notes Export Workflow

Place exported notes under:

```text
data/raw/apple_notes_export/
```

Supported Phase 2 input formats:

- `.md`
- `.txt`
- `.html`
- `.htm`
- `.json`
- `.pdf`

Unsupported attachments are skipped.

## Database Initialization

```bash
python -m notes_lifelog_rag.cli init-db
```

This creates `data/processed/notes.db` with the core tables, including SQLite
FTS5 keyword search.

## Ingestion

```bash
python -m notes_lifelog_rag.cli ingest-notes --input data/raw/apple_notes_export
```

Re-ingestion is idempotent. Duplicate note content is skipped using a stable
content hash. Parser errors are recorded and do not stop the import.

## Stats

```bash
python -m notes_lifelog_rag.cli stats
```

## Embeddings

Build chunk embeddings with a local configured embedding model:

```bash
python -m notes_lifelog_rag.cli build-embeddings
```

If the local ML runtime is not installed, the command reports a disabled reason
without crashing. For development and tests only, deterministic local mock
embeddings can be built explicitly:

```bash
python -m notes_lifelog_rag.cli build-embeddings --backend mock
```

`build-embeddings` is resume-safe for larger note sets. By default it uses
`--only-missing`, skips chunks that already have a successful row in
`chunk_embeddings` for the same model and text hash, shows a progress bar, and
commits after each batch. Useful modes:

```bash
python -m notes_lifelog_rag.cli build-embeddings --only-missing --limit 100
python -m notes_lifelog_rag.cli build-embeddings --dry-run --limit 100
python -m notes_lifelog_rag.cli build-embeddings --force --limit 100
```

Use `--dry-run` before processing many notes to confirm how many chunks would be
embedded without storing embeddings. Use `--force` only when you intentionally
want to rebuild existing embeddings.

## Search

```bash
python -m notes_lifelog_rag.cli search "研究"
python -m notes_lifelog_rag.cli search "研究" --show-snippets
```

Search now uses hybrid retrieval: SQLite FTS/LIKE keyword search plus stored
embedding vectors when available. Reranking is optional and local-only. The
default output avoids body snippets. Use `--show-snippets` when you explicitly
want short evidence snippets in the terminal.

## Analysis

The Phase 4 commands store structured JSON-derived outputs with short evidence
quotes and model-run cache records:

```bash
python -m notes_lifelog_rag.cli summarize-notes
python -m notes_lifelog_rag.cli categorize-notes
python -m notes_lifelog_rag.cli extract-events
python -m notes_lifelog_rag.cli extract-thoughts
python -m notes_lifelog_rag.cli analyze-all
```

These commands are designed for safe differential updates. `--limit` defaults to
`None`, so omitting it scans all notes and selects every missing or stale note.
Use `--limit 10` only for a small sample run. `--only-missing` is enabled by
default and skips only outputs whose current `model_runs` cache matches the
task, model, note id, note content hash, and prompt version. If a note body
changes, the content hash changes and that note becomes eligible for analysis
again.

`--dry-run` is read-only: it does not insert or update `model_runs`,
`note_summaries`, `note_categories`, `events`, `thoughts`, reflections,
suggestions, or import errors. It reports the total scanned notes, eligible
notes, skip counts, cache-hit estimates, selected count, and up to 20
`would_process_*_note_ids` per task.

The `model_runs` cache avoids re-running the local LLM when a successful cached
JSON payload for the current hash already exists. If an output table row is
missing, cached payloads can be restored without generating again. Successful
cached empty outputs such as `{"events": []}` or `{"thoughts": []}` are treated
as completed work, which prevents notes with no extracted events or thoughts
from blocking later `--only-missing --limit` batches. Each note is committed
independently so the batch can be resumed after interruption.

```bash
python -m notes_lifelog_rag.cli analyze-all --only-missing --dry-run
python -m notes_lifelog_rag.cli analyze-all --only-missing
python -m notes_lifelog_rag.cli analyze-all --only-missing --limit 10
python -m notes_lifelog_rag.cli summarize-notes --only-missing --limit 100
python -m notes_lifelog_rag.cli extract-events --force
```

By default, analysis uses a local model when the runtime is enabled and falls
back to the deterministic mock backend otherwise. To require the real local LLM
path/runtime, pass:

```bash
python -m notes_lifelog_rag.cli summarize-notes --backend local
```

## Timeline And Reflections

```bash
python -m notes_lifelog_rag.cli timeline-months
python -m notes_lifelog_rag.cli generate-timeline --month 2026-05 --backend rule --dry-run
python -m notes_lifelog_rag.cli generate-timeline --all-months --backend rule
python -m notes_lifelog_rag.cli timeline --month 2026-05
python -m notes_lifelog_rag.cli timeline --month 2026-05 --rich
python -m notes_lifelog_rag.cli timeline --year 2026 --monthly --order asc
python -m notes_lifelog_rag.cli timeline-report --year 2026 --output data/exports/timelines/timeline_2026.md
python -m notes_lifelog_rag.cli timeline-qa --month 2026-05
python -m notes_lifelog_rag.cli reflections --month 2026-05 --force
python -m notes_lifelog_rag.cli generate-reflections --all-months
```

Timeline is now a core monthly memory feature, not just a flat event list.
`generate-timeline` builds month cards from `thoughts`, `events`,
`note_summaries`, categories, suggestions, and existing reflections. Each
monthly card answers:

- what this month seemed to be about;
- what you may have been thinking about;
- what happened or progressed;
- which themes and categories were dominant;
- what changed or may be worth revisiting;
- which evidence note quotes support the summary.

The generated data is stored in `monthly_timeline_snapshots` and
`monthly_timeline_items`. `--dry-run` previews the month card without writing DB
rows. `--force` replaces only the target month's timeline snapshot/items; it
does not truncate the whole database.

Use `timeline-months` to see month coverage and whether a saved snapshot exists.
Use `timeline --month 2026-05 --rich` to read a rich month detail. Use
`timeline --year 2026 --monthly --order asc` to browse a year in chronological
order. Use `timeline-report` to export a Markdown year/all-year timeline.

`timeline-qa` checks whether month cards have evidence, thought/event summaries,
reasonable confidence, and whether the card is too fallback-heavy. A
`fallback_heavy` warning means the month card is leaning on note summaries or
titles because structured thoughts/events are sparse; run `analyze-all`, then
regenerate the timeline with `generate-timeline --month YYYY-MM --force`.

Timeline and reflection outputs include `evidence`, `confidence`, and
`importance`, and keep source note IDs visible. Reflections prefer `thoughts`,
then `events`, then `note_summaries`; title fallback is used only when
structured analysis is sparse. Reflection output also includes coverage and
quality warnings.

## Suggestions And QA Review

Phase 7 adds a rule-based suggestion engine. It does not call external APIs or
download models. Suggestions are generated from high-importance thoughts/events,
summary `revisit_reason`, same-month/day rediscovery, low-confidence outputs,
and weak evidence warnings.

```bash
python -m notes_lifelog_rag.cli generate-suggestions --limit 100
python -m notes_lifelog_rag.cli generate-suggestions --month 2026-05
python -m notes_lifelog_rag.cli generate-suggestions --today
python -m notes_lifelog_rag.cli qa-report --limit 50
python -m notes_lifelog_rag.cli qa-report --month 2026-05 --output data/exports/qa_reports/latest.md
```

QA Review highlights low confidence, missing evidence, title-only evidence,
unknown event dates, model-run failures, import errors, empty summaries, and
very short summaries.

## UI

Launch the local Notes-like Gradio UI:

```bash
python -m notes_lifelog_rag.cli ui --host 127.0.0.1 --port 7860
```

The UI is a dark-mode Notes-like workspace inspired by the calm structure of
native notes apps and Apple HIG principles, but it is not Apple Notes and does
not use Apple logos, Apple Notes branding, or Apple-owned iconography. It binds
to `127.0.0.1` by default so the private notes workspace stays local.

The first screen is **Notes Workspace**, a three-pane memory workspace:

- **Sidebar**: app status, note counts, categories, months, Today Rediscovery,
  Timeline, Reflection, Models / Settings, and DB stats. The Library,
  Category, and Month controls sit at the top of the left pane for quick
  navigation.
- **Note List**: searchable note cards with generated title, one-line summary,
  category badges, source path, confidence, importance, and review badges.
- **Detail Pane**: a dark paper-like note preview with AI summary, important
  points, categories, events, thoughts, evidence quotes, model-run info, body
  hash, source path, and original note body.

After Search / Filter, click any note card in the center pane to open it. The
selected card is highlighted and the right detail pane refreshes immediately;
keyboard focus plus Enter/Space works as well. Selected Library and filter
controls use filled yellow indicators and a subtle highlighted row so active
scope/filter state is easy to see in dark mode.

Phase 7 tabs:

- Notes Workspace
- Import
- Analysis Jobs
- QA Review
- Timeline
- Reflections
- Suggestions
- Models / Settings

Use **Import** to initialize the DB and ingest exported notes. Use the global
search in Notes Workspace for hybrid search; the Ask box gives a cautious answer
from retrieved evidence rather than inventing unsupported history. Use the
Timeline tab to browse year/month memory cards: each month shows overview,
thought summary, event summary, key themes, quality warnings, evidence, and
related notes. Use **Suggestions** to generate and review “today rediscovery”,
monthly reflection, important thought/event, revisit note, low-confidence
review, and evidence-review candidates. In the workspace,
choosing **Suggestions** shows suggestion cards in the center pane; selecting one
opens the source note and original body in the right detail pane. The
Suggestions card stack has its own scroll area, so the right-side original note
preview stays in place while you browse many rediscovery candidates.
The dedicated **QA Review**, **Timeline**, **Reflections**, and **Suggestions**
tabs also use selectable cards. Choosing a card opens the supporting source note
in the right detail pane with its AI summary, events, thoughts, evidence, model
run metadata, and original note body.

**Analysis Health** appears in the workspace and settings. It shows summaries,
category coverage, event coverage, thought coverage, embedding coverage,
suggestions, monthly reflections, model-run failures, import errors,
low-confidence items, and evidence warnings. If summaries/events/thoughts are
sparse, run:

```bash
python -m notes_lifelog_rag.cli analyze-all --backend local --only-missing --device auto --dtype auto --max-new-tokens 512 --show-device --dry-run
python -m notes_lifelog_rag.cli analyze-all --backend local --only-missing --device auto --dtype auto --max-new-tokens 512 --show-device
```

If a separate `analyze-all` is already running, the UI displays a warning and
does not start heavy analysis, embedding, suggestion, or reflection jobs from
the Analysis Jobs tab. For GPU analysis, the recommended full differential run
is:

```bash
python -m notes_lifelog_rag.cli analyze-all \
  --backend local \
  --only-missing \
  --device auto \
  --dtype auto \
  --batch-size 1 \
  --max-new-tokens 512 \
  --show-device
```

AI-derived outputs in the UI always keep evidence, confidence, and importance
visible. Note bodies are not shown in bulk list views; the original body appears
only in the selected note detail.

## Tests

```bash
pytest
bash scripts/smoke_test.sh
```

## Troubleshooting

- If `model-status` reports a missing path, the feature should remain disabled.
  Do not download models automatically.
- If PDF parsing fails, confirm that `pypdf` is installed in the active conda
  environment.
- If no search results appear for Japanese text, ingestion should still have
  populated notes; the search command also uses a SQLite `LIKE` fallback.
