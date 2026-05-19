#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -m notes_lifelog_rag.cli init-db
python -m notes_lifelog_rag.cli ingest-notes --input data/raw/apple_notes_export
python -m notes_lifelog_rag.cli stats
python -m notes_lifelog_rag.cli search "研究" --limit 3 --embedding-backend none --reranker-backend none
python -m notes_lifelog_rag.cli model-status
python -m notes_lifelog_rag.cli build-embeddings --backend mock --limit 2 --dry-run
python -m notes_lifelog_rag.cli build-embeddings --backend mock --limit 2
python -m notes_lifelog_rag.cli analyze-all --backend mock --limit 1 --dry-run
python -m notes_lifelog_rag.cli analyze-all --backend mock --limit 1
python -m notes_lifelog_rag.cli timeline --month 1900-01 --limit 3
python -m notes_lifelog_rag.cli reflections --month 1900-01
