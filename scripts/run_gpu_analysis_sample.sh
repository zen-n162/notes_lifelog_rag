#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/MyApplication/notes_lifelog_rag"

conda run --no-capture-output -n notes_lifelog_rag \
  python -m notes_lifelog_rag.cli analyze-all \
    --backend local \
    --only-missing \
    --limit 3 \
    --device auto \
    --dtype auto \
    --batch-size 2 \
    --max-new-tokens 512 \
    --show-device
