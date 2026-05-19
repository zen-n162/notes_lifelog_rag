#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/MyApplication/notes_lifelog_rag"

conda run --no-capture-output -n notes_lifelog_rag \
  python -m notes_lifelog_rag.cli build-embeddings \
    --backend local \
    --only-missing \
    --limit 10 \
    --dry-run \
    --device auto \
    --batch-size 8 \
    --show-device
