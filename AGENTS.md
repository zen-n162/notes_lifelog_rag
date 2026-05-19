# AGENTS.md

This file defines persistent instructions for Codex and other coding agents working on this repository.

Repository root:

```text
~/MyApplication/notes_lifelog_rag
```

Primary conda environment:

```text
notes_lifelog_rag
```

## 1. Project purpose

This project is a local-first Apple Notes / iPhone Notes lifelog RAG application.

The application ingests user-exported notes from Apple Notes and helps the user:

- organize notes by category;
- generate titles, one-line summaries, detailed summaries, and important points;
- extract events from notes and build timelines;
- extract thoughts, values, worries, insights, decisions, and turning points;
- rediscover past ideas and memories by month, theme, or question;
- search notes using keyword search, embedding search, and reranking;
- always show evidence from the original note when producing AI summaries or suggestions.

The core value is not simple summarization. The application should help the user remember:

- what happened;
- when it happened;
- what they were thinking at the time;
- why that note may matter now;
- which original note text supports the suggestion.

## 2. Non-negotiable privacy and safety rules

This repository handles private personal notes.

Always follow these rules:

- Do not send note contents, embeddings, summaries, logs, or metadata to external APIs.
- Do not use OpenAI API, Gemini API, Claude API, cloud OCR, cloud embeddings, cloud vector databases, or any hosted LLM service.
- Do not add telemetry, analytics, remote logging, or background upload behavior.
- Do not require API keys.
- Do not scrape iCloud.com.
- Do not ask for or store Apple ID credentials.
- Do not attempt to directly read Apple Notes internal databases unless the user explicitly asks for a separate, clearly reviewed experimental feature.
- Default UI host must be `127.0.0.1`.
- Do not print large amounts of personal note text to the terminal.
- Do not include personal note contents in public examples, test snapshots, or committed fixtures.
- AI outputs must include evidence references and must separate fact from inference.
- Never present an LLM inference as a confirmed fact unless the source note explicitly supports it.

If a requested change conflicts with these rules, explain the conflict and implement a safer local-only alternative.

## 3. Relationship to other projects

There is an existing project:

```text
~/MyApplication/personal_lifelog_rag
```

Do not modify it unless the user explicitly asks.

This repository is a separate project:

```text
~/MyApplication/notes_lifelog_rag
```

You may inspect patterns from the existing project if helpful, but do not copy private data, do not alter its files, and do not make this repository depend on it at runtime.

## 4. Environment rules

Use the conda environment:

```bash
conda activate notes_lifelog_rag
```

Expected setup commands:

```bash
cd ~/MyApplication/notes_lifelog_rag
pip install -e ".[dev]"
```

Prefer Python 3.11.

Do not install packages globally.

Do not run model download commands automatically.

Forbidden unless explicitly requested by the user:

```bash
hf download ...
huggingface-cli download ...
wget ...models...
curl ...models...
```

The models are assumed to already exist under:

```text
/home/zennakamura/MyApplication/models
```

If a model path is missing:

- mark the corresponding feature as disabled;
- show a clear warning in `model-status`;
- do not crash unrelated commands;
- do not attempt to download the model.

## 5. Local model inventory

Use local models from:

```text
/home/zennakamura/MyApplication/models
```

Primary text models:

```text
/home/zennakamura/MyApplication/models/qwen/Qwen3-Swallow-8B-RL-v0.2
/home/zennakamura/MyApplication/models/qwen/Qwen3-4B-Instruct-2507
/home/zennakamura/MyApplication/models/qwen/Qwen3-Embedding-0.6B
/home/zennakamura/MyApplication/models/qwen/Qwen3-Reranker-0.6B
/home/zennakamura/MyApplication/models/embedding/ruri-v3-310m
/home/zennakamura/MyApplication/models/reranker/ruri-v3-reranker-310m
/home/zennakamura/MyApplication/models/whisper/faster-whisper-large-v3
```

Existing vision / OCR / face / VL models:

```text
/home/zennakamura/MyApplication/models/face/opencv/yunet/face_detection_yunet_2023mar.onnx
/home/zennakamura/MyApplication/models/face/opencv/sface/face_recognition_sface_2021dec.onnx
/home/zennakamura/MyApplication/models/insightface/models/buffalo_l
/home/zennakamura/MyApplication/models/florence/Florence-2-large
/home/zennakamura/MyApplication/models/paddleocr/PaddleOCR-VL
/home/zennakamura/MyApplication/models/qwen/Qwen3-VL-8B-Thinking
/home/zennakamura/MyApplication/models/qwen/Qwen3-VL-Embedding-8B
/home/zennakamura/MyApplication/models/qwen/Qwen3-VL-Reranker-2B
/home/zennakamura/MyApplication/models/qwen/Qwen3-VL-Reranker-8B
```

Model usage policy:

- Use text-specific models for ordinary Apple Notes analysis.
- Use VL/OCR models only for image/PDF/OCR-related features.
- Heavy GPU usage is acceptable.
- Tests must not require loading large models.
- Provide a mock backend for LLM, embedding, and reranker tests.

## 6. Expected repository structure

Maintain this structure unless there is a strong reason to change it:

```text
notes_lifelog_rag/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── environment.yml
├── configs/
│   ├── app.yaml
│   ├── categories.yaml
│   ├── models.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/
│   │   └── apple_notes_export/
│   │   |   ├── Notes
│   │   |   │   ├── attachments
│   │   |   │   ├── images
│   │   |   ├── Tanpopo
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── いおり
│   │   |   │   ├── attachments
│   │   |   │   ├── images
│   │   |   │   ├── ディズニー旅行 2025-03-04-06
│   │   |   │   │   ├── attachments
│   │   |   │   │   └── images
│   │   |   │   └── 大阪-京都-金沢-新潟旅行 2025-09-17-23
│   │   |   │       ├── attachments
│   │   |   │       └── images
│   │   |   ├── やることリスト
│   │   |   ├── エピソード
│   │   |   ├── ダンス
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── 映画
│   │   |   ├── 歌詞
│   │   |   │   ├── attachments
│   │   |   │   ├── images
│   │   |   │   └── 歌詞 - 森七菜
│   │   |   │       ├── attachments
│   │   |   │       └── images
│   │   |   ├── 研究
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── 就活
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── 新規スマートフォルダ
│   │   |   ├── 森七菜
│   │   |   ├── 大学
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── 大分旅行 しおり
│   │   |   │   ├── attachments
│   │   |   │   └── images
│   │   |   ├── 日記
│   │   |   ├── 買い物
│   │   |   └── 病院
│   ├── interim/
│   │   ├── parsed_notes/
│   │   ├── chunks/
│   │   └── llm_outputs/
│   ├── processed/
│   │   ├── notes.db
│   │   └── vector_index/
│   └── exports/
├── scripts/
│   ├── run_ui.sh
│   ├── run_analyze_all.sh
│   └── smoke_test.sh
├── src/
│   └── notes_lifelog_rag/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── ingest/
│       ├── db/
│       ├── search/
│       ├── llm/
│       ├── analysis/
│       ├── ui/
│       └── utils/
└── tests/
```

## 7. Supported input data

The application should ingest user-exported note files, not Apple Notes directly.

Initial supported formats:

- `.md`
- `.txt`
- `.pdf`
- `.html`
- `.htm`
- `.json`

Data input directory:

```text
data/raw/apple_notes_export/
```

Ingestion rules:

- Preserve source path.
- Preserve title if available.
- Preserve body.
- Extract metadata where possible.
- Use a stable content hash to avoid duplicate imports.
- Re-ingesting the same file should be idempotent.
- If parser errors occur, record them clearly and continue with other files.

## 8. Database expectations

Use SQLite.

Use FTS5 for keyword search.

Expected core tables:

- `notes`
- `notes_fts`
- `note_chunks`
- `categories`
- `note_categories`
- `note_summaries`
- `events`
- `thoughts`
- `suggestions`
- `model_runs`

Schema changes should be implemented as migrations or idempotent initialization logic.

Do not drop existing user data without explicit user approval.

## 9. CLI contract

Use Typer for the CLI.

The following commands should exist or be preserved once implemented:

```bash
python -m notes_lifelog_rag.cli init-db
python -m notes_lifelog_rag.cli ingest-notes --input data/raw/apple_notes_export
python -m notes_lifelog_rag.cli stats
python -m notes_lifelog_rag.cli model-status
python -m notes_lifelog_rag.cli search "研究"
python -m notes_lifelog_rag.cli ask "2026年4月にはどんな研究の考えをしていた？"
python -m notes_lifelog_rag.cli summarize-notes --limit 10
python -m notes_lifelog_rag.cli summarize-notes --all
python -m notes_lifelog_rag.cli categorize-notes --all
python -m notes_lifelog_rag.cli extract-events --all
python -m notes_lifelog_rag.cli extract-thoughts --all
python -m notes_lifelog_rag.cli analyze-all
python -m notes_lifelog_rag.cli timeline --month 2026-05
python -m notes_lifelog_rag.cli reflections --month 2026-05
python -m notes_lifelog_rag.cli ui --host 127.0.0.1 --port 7860
```

All CLI commands should:

- fail gracefully with readable messages;
- avoid dumping private note bodies unless explicitly requested;
- support deterministic and testable output where practical;
- keep heavy model loading optional until needed.

## 10. Web UI contract

Use Gradio for the initial MVP unless the user asks for FastAPI + React.

Default command:

```bash
python -m notes_lifelog_rag.cli ui --host 127.0.0.1 --port 7860
```

Initial UI tabs:

1. Import
2. Notes
3. Search / Ask
4. Timeline
5. Reflections
6. Models / Settings

UI requirements:

- Always allow the user to inspect the original note behind an AI output.
- Show `evidence`, `confidence`, and `importance` for summaries, thoughts, events, and suggestions.
- Make fact vs inference visually clear.
- Do not expose the UI on `0.0.0.0` by default.
- Do not display excessive raw private text in list views.
- Use short snippets in tables and full text only in detail views.

## 11. LLM output requirements

All LLM analysis outputs must be structured JSON.

For every LLM-generated item, store:

- model name;
- task name;
- input hash;
- output JSON;
- success/failure;
- error message if any;
- creation time.

Cache LLM runs by:

```text
task_name + model_name + input_hash
```

If the same task is rerun on the same note with the same model, reuse cached output unless `--force` is provided.

All LLM outputs that interpret notes must include evidence.

Minimum evidence object:

```json
{
  "note_id": "...",
  "quote": "short supporting quote"
}
```

Do not allow long quotes in logs or public test fixtures.

## 12. Prompt behavior rules

Prompts must instruct the model to:

- write Japanese output by default;
- return JSON only for extraction tasks;
- avoid unsupported claims;
- lower confidence when evidence is ambiguous;
- distinguish actual events from plans;
- distinguish thoughts from facts;
- avoid over-psychologizing the user;
- use phrases like `可能性があります` when inference is involved;
- include `importance` and `confidence` separately;
- include evidence for every extracted event, thought, or suggestion.

## 13. Analysis feature requirements

### Summaries

For each note, generate:

- generated title;
- one-line summary;
- detailed summary;
- important points;
- revisit reason;
- confidence;
- evidence.

### Categories

Initial categories should be defined in `configs/categories.yaml`.

Include at least:

- 研究
- 修論
- ハイパースペクトル
- 月面探査
- 機械学習
- アプリ開発
- AIエージェント
- 就職活動
- 企業研究
- ES・面接
- QST
- Sony
- Canon
- Nikon
- アイデア
- 学習
- 日記・出来事
- 感情・内省
- 人間関係
- 生活
- 予定・タスク
- 写真
- ブレイキン
- その他

Multiple categories per note are allowed.

### Events

Events should capture:

- actual events;
- plans;
- decisions;
- progress;
- emotional turning points;
- ideas;
- learning events;
- other notable happenings.

Each event must include:

- title;
- summary;
- event type;
- date or date label;
- date confidence;
- importance;
- confidence;
- evidence.

### Thoughts

Thoughts should capture:

- insights;
- values;
- worries;
- decisions;
- reflections;
- goals;
- ideas;
- regret;
- growth;
- other meaningful internal states.

Each thought must include:

- title;
- summary;
- thought type;
- themes;
- emotion label/intensity when supported;
- date label;
- importance;
- confidence;
- why this may be worth remembering;
- evidence.

### Reflections

Monthly reflection should summarize:

- main events;
- main thoughts;
- important changes;
- rediscovery points;
- suggested reminder messages;
- supporting evidence.

The reflection should help the user remember what they were doing and thinking in that month.

## 14. Date handling

Implement date parsing in `utils/dates.py`.

Support at least:

- `YYYY-MM-DD`
- `YYYY/MM/DD`
- `YYYY年M月D日`
- `M月D日`
- `今日`
- `昨日`
- `明日`
- `先週`
- `先月`
- `春頃`
- `夏頃`
- `秋頃`
- `冬頃`
- `何年何月ごろ`

Use the note creation/modification/export date as context for relative dates.

Date confidence policy:

- explicit full date: high confidence;
- relative date with reliable note date: medium confidence;
- vague month/season expression: low confidence;
- unknown date: keep `date_label` and low confidence rather than inventing a date.

## 15. Search and RAG requirements

Implement hybrid search:

1. SQLite FTS5 keyword search.
2. Embedding semantic search.
3. Merge and deduplicate results.
4. Rerank top candidates if a reranker is available.
5. Search across notes, summaries, events, and thoughts where appropriate.
6. Return evidence note IDs and snippets.

For `ask`, use retrieved evidence to answer.

`ask` must:

- answer in Japanese by default;
- cite source note IDs or titles internally in the output;
- say when there is insufficient evidence;
- not invent unsupported history;
- display relevant snippets or evidence.

## 16. Testing rules

Tests must pass without loading large local models.

Use mock backends for:

- LLM generation;
- embeddings;
- reranking;
- ASR.

Expected test command:

```bash
pytest
```

Also maintain:

```bash
scripts/smoke_test.sh
```

Smoke test should verify at least:

```bash
python -m notes_lifelog_rag.cli init-db
python -m notes_lifelog_rag.cli ingest-notes --input data/raw/apple_notes_export
python -m notes_lifelog_rag.cli stats
python -m notes_lifelog_rag.cli search "研究"
python -m notes_lifelog_rag.cli model-status
```

Before reporting completion, run:

```bash
pytest
bash scripts/smoke_test.sh
```

If a command fails, report the failure honestly with the error and what remains to fix.

## 17. Code quality rules

Use clear, maintainable Python.

Prefer:

- small modules;
- typed functions where practical;
- Pydantic schemas for structured LLM outputs;
- rich console output for CLI readability;
- explicit errors;
- idempotent DB operations;
- deterministic tests.

Avoid:

- hidden network calls;
- broad `except Exception` without logging;
- mixing UI code with DB logic;
- hardcoding private note content;
- destructive migrations;
- huge monolithic files;
- changing public CLI behavior without updating README and tests.

## 18. Implementation priority

When starting from scratch, implement in this order.

### Phase 1: Project foundation

- `pyproject.toml`
- `environment.yml`
- `README.md`
- `AGENTS.md`
- `configs/`
- package layout
- DB connection/schema
- `init-db`
- `stats`
- sample note
- base tests

### Phase 2: Ingestion and keyword search

- Markdown parser
- text parser
- HTML parser
- PDF parser
- JSON parser
- importer
- duplicate detection
- FTS5 table
- `search`

### Phase 3: Model registry and retrieval

- `model-status`
- model path validation
- embedding wrapper
- chunking
- vector storage
- reranker wrapper
- hybrid search

### Phase 4: LLM analysis

- local LLM wrapper
- mock LLM backend
- JSON parsing and repair
- summaries
- categories
- events
- thoughts
- `analyze-all`
- model run cache

### Phase 5: Timeline and reflections

- timeline builder
- monthly reflections
- suggestions
- exportable reports

### Phase 6: Gradio UI

- Import tab
- Notes tab
- Search / Ask tab
- Timeline tab
- Reflections tab
- Models / Settings tab

Keep the repository runnable after every phase.

## 19. File and data handling

Do not delete user data.

Do not run destructive commands such as:

```bash
rm -rf data/
rm -rf ~/MyApplication/models
rm -rf ~/MyApplication/personal_lifelog_rag
```

unless the user explicitly requests it and the command is narrowly scoped.

Prefer creating backups before schema migrations.

Generated local data may live under:

```text
data/interim/
data/processed/
data/exports/
```

Do not commit large databases, model files, embeddings, or private note exports.

Ensure `.gitignore` excludes at least:

```text
data/raw/
data/interim/
data/processed/*.db
data/processed/vector_index/
data/exports/
*.sqlite
*.db
__pycache__/
.pytest_cache/
```

## 20. README update rule

Whenever CLI commands, setup steps, model paths, or UI behavior change, update `README.md`.

Also update the application explanation page:

```text
docs/application_overview.html
```

Update this HTML file whenever the application behavior, UI tabs/layout, CLI commands, setup flow, DB tables, model configuration, privacy policy, or evidence/confidence/importance display rules change. The HTML page must stay public-safe: do not include private note contents, raw personal examples, large excerpts, embeddings, or generated outputs derived from the user's real notes. Use generic examples only.

README should include:

- project overview;
- privacy policy;
- setup;
- conda environment;
- model paths;
- Apple Notes export workflow;
- DB initialization;
- ingestion;
- analysis;
- search;
- UI launch;
- tests;
- troubleshooting.

## 21. GitHub and Git update rule

This repository's GitHub remote is:

```text
https://github.com/zen-n162/notes_lifelog_rag.git
```

Use this as the `origin` remote unless the user explicitly asks for a different remote.

After making any repository change, update Git before reporting completion:

1. Run `git status --short --branch`.
2. Review the changed files and stage only the intended source, config, test, script, README, AGENTS, or public-safe docs changes.
3. Do not stage or commit private note exports, local databases, embeddings, generated vector indexes, model files, cache directories, logs with private note text, or large generated artifacts.
4. Commit the change with a concise message that describes the user-visible outcome.
5. Push the commit to `origin` when network access and credentials are available. If push is not possible, report the local commit hash and that it remains unpushed.

If a task only inspects the repository and makes no file changes, do not create an empty commit.

## 22. Completion report format

When finishing a task, report:

- files created or changed;
- CLI commands added or changed;
- DB tables added or changed;
- model behavior changed;
- tests run;
- smoke test result;
- known limitations;
- recommended next task.

Be honest about incomplete work.

## 23. Design principle

The application should act like a private memory assistant, not a judgment system.

Good output:

```text
2026年4月ごろ、あなたは月面資源探査において、スペクトル情報だけでなく空間情報も重要ではないかと考えていた可能性があります。これは、HSIを単なる物質識別だけでなく、採掘地最適化や地形的アクセス評価へ広げる重要な気づきだった可能性があります。
```

Bad output:

```text
あなたは2026年4月に人生の方向性を完全に変えました。
```

The first output is evidence-aware and cautious. The second overstates unsupported inference.

Always prefer cautious, evidence-grounded, useful reflections.
