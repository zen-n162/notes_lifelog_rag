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

`model_runs` also records failure diagnostics. New rows include `note_id`,
`body_hash`, `prompt_version`, `raw_output`, `error_type`, `error_message`,
`empty_result`, `retry_count`, and `fallback_used`. Legacy failed rows are
backfilled with `legacy_unknown_failure` or a best-effort classification such as
`json_parse_error`, `truncated_output`, `cuda_oom`, or `model_load_error`.
Use these commands when local LLM output fails to parse:

```bash
python -m notes_lifelog_rag.cli analyze-all --only-missing --dry-run
python -m notes_lifelog_rag.cli analyze-all --only-missing
python -m notes_lifelog_rag.cli analyze-all --only-missing --limit 10
python -m notes_lifelog_rag.cli summarize-notes --only-missing --limit 100
python -m notes_lifelog_rag.cli extract-events --force
python -m notes_lifelog_rag.cli analysis-failures --group-by-error
python -m notes_lifelog_rag.cli analysis-failures --task events --limit 20
python -m notes_lifelog_rag.cli db-schema --table model_runs
python -m notes_lifelog_rag.cli analyze-sample --task summary --backend local --device auto --max-new-tokens 1024 --show-raw-output
```

`analyze-sample` analyzes one or a few notes for debugging and does not save to
the DB unless `--save` is passed. It can show the prompt, raw model output,
parsed JSON, and classified error type without changing analysis tables.
JSON parsing is intentionally lenient: fenced JSON, text before/after JSON,
trailing commas, common full-width quotes, empty output, and truncated output
are handled or classified so the original raw output is not lost.

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
python -m notes_lifelog_rag.cli generate-timeline --all-months --backend rule --force
python -m notes_lifelog_rag.cli generate-timeline --month 2026-05 --backend rule --force --show-sources
python -m notes_lifelog_rag.cli timeline --month 2026-05
python -m notes_lifelog_rag.cli timeline --month 2026-05 --rich
python -m notes_lifelog_rag.cli timeline --month 2026-05 --rich --low-priority-limit 3
python -m notes_lifelog_rag.cli timeline --month 2026-05 --rich --ungrouped --show-low-priority
python -m notes_lifelog_rag.cli timeline --year 2026 --monthly --order asc
python -m notes_lifelog_rag.cli timeline-report --year 2026 --grouped --low-priority-limit 3 --output data/exports/timelines/timeline_2026.md
python -m notes_lifelog_rag.cli timeline-qa --all-months
python -m notes_lifelog_rag.cli timeline-qa --month 2026-05 --show-items
python -m notes_lifelog_rag.cli timeline-qa --all-months --only-problems
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
When running `--all-months`, generation continues month-by-month even if one
month has malformed source data, and item IDs include the month so source items
that appear in more than one month do not collide.

Timeline month attribution deliberately avoids using a suggestion's
`created_at`, because that is the date the recommendation was generated, not
the date of the memory. The month decision now follows this order:

- `thought`: `thoughts.date_start` / `date_label`, then evidence/source note
  dates.
- `event`: `events.date_start` / `date_label`, then evidence/source note dates.
- `note_summary`: source note `created_at`, then `modified_at`.
- `suggestion`: `target_date`, then evidence/source note dates. If `target_date`
  looks like the generation day and no evidence/source date can be found, the
  suggestion is excluded from Timeline.
- `monthly_reflection`: the stored reflection month.

Each timeline item stores a `date_source` and `date_quality`. Explicit event or
thought dates are high confidence; source note `created_at` is medium
confidence and is not treated as suspicious by itself; `modified_at` is a
lighter signal; unknown or invalid dates trigger `date_attribution_uncertain`.

Suggestions are supporting material, not the main source for a month. By
default, month generation keeps at most 10 thoughts, 10 events, 10 summaries, 5
suggestions, and 3 fallback items. You can tune this with
`--max-thoughts-per-month`, `--max-events-per-month`,
`--max-summaries-per-month`, `--max-suggestions-per-month`, and
`--max-fallback-per-month`.

Low priority items are kept visible for review but are not used to drive the
month overview, thought summary, event summary, or title. Lyrics/music notes,
shopping/menu notes, link-only notes, noisy or scanned PDF text, title-only
evidence, very short summaries, weak confidence, and weak importance are marked
as `low_priority`. Research PDFs are not low priority just because they are PDF
files; they are demoted only when the extracted text looks noisy, scanned, or
low-value. In the GUI Timeline detail, these appear under **Low Priority /
Needs Review** rather than **Main Timeline Items**.

**Main Timeline Items** are the high-signal thoughts, events, summaries, and
reflection material used to explain the month. Direct thoughts/events are always
considered for the month text when they exist, even if the final wording adds a
caution for low confidence. **Low Priority / Needs Review** remains available
for audit, but `timeline --month YYYY-MM --rich` shows only the first three by
default. Use `--low-priority-limit N` to change that count,
`--show-low-priority` to print all low-priority items, or
`--hide-low-priority` to suppress them entirely. Low-priority rows include
reason flags such as `noisy_pdf`, `scanned_document`, `mojibake`,
`shopping_list`, `lyric_or_song`, `low_confidence`, `low_importance`,
`weak_evidence`, `section_fragment`, `duplicate_same_note`, and
`date_uncertain`.

Timeline rich output and timeline reports use **grouped view** by default.
Items from the same `source_note_id` are merged into one representative card so
long notes with many sections do not overwhelm the month. For example, a QST ES
note can group `QST ES`, `QST ES の考え`, `自己PR`, `志望動機`, `趣味`, and
`特技` into a single **QST ES・自己PR・志望動機の整理** card with `sub_items`.
Use `--ungrouped` when you need to audit every raw extracted item. Use
`--grouped` explicitly when you want to make the default visible in scripts.

Timeline evidence is also enriched from the original note body. If an AI output
only provides a title-like quote such as `# QST ES`, Timeline tries to extract a
100-180 character source quote from the note body and marks the item as
`evidence_enriched`. A remaining `title_only_evidence` warning means the source
body could not provide a stronger quote and the original note should be checked.

Month overview, thought summary, event summary, important changes, and revisit
reasons are generated by a local rule-based narrative summarizer. It uses
grouped main items and themes to write short Japanese sentences instead of
concatenating raw quotes, and keeps evidence quotes in the evidence section.

For months like `2026-05`, read the overview as a cautious synthesis of direct
thoughts/events plus summaries: QST ES, research planning, machine learning,
moon exploration, and public-interest research support may appear together, but
low-confidence or noisy PDF-derived items should be reviewed in the low-priority
section rather than treated as the month theme.

Use `timeline-months` to see month coverage and whether a saved snapshot exists.
Use `timeline --month 2026-05 --rich` to read a rich month detail. Use
`timeline --year 2026 --monthly --order asc` to browse a year in chronological
order. Use `timeline-report` to export a Markdown year/all-year timeline.

`timeline-qa` checks whether month cards have evidence, thought/event summaries,
reasonable confidence, and whether the card is too fallback-heavy. A
`fallback_heavy` warning means the month card is leaning on note summaries or
titles because structured thoughts/events are sparse; run `analyze-all`, then
regenerate the timeline with `generate-timeline --month YYYY-MM --force`.
Use `timeline-qa --show-items` to inspect which items are main versus
low-priority, and `--only-problems` to focus on months that still need review.
Other useful warnings:

- `suggestions_dominated`: suggestions still outnumber direct thought/event/
  summary material.
- `low_direct_thoughts` / `low_direct_events`: the month needs more structured
  analysis before "what I thought" or "what happened" can be trusted.
- `noisy_items_present` / `low_value_items_present`: low priority notes are
  present and should be reviewed separately.
- `date_attribution_uncertain`: some items have weak or missing dates.
- `generated_date_used_for_suggestion`: a suggestion looked tied to its
  generated day and was treated cautiously.
- `invalid_month_1900`: the month is an unknown-date fallback.

`1900-01` is treated as an unknown-date bucket. It is hidden from
`timeline-months`, `timeline --all-months`, reports, QA, and GUI month lists by
default. Use `--include-unknown` when you intentionally want to inspect unknown
dates.

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
