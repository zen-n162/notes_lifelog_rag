#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/MyApplication/notes_lifelog_rag"

conda run --no-capture-output -n notes_lifelog_rag \
  python -m notes_lifelog_rag.cli cuda-status

conda run --no-capture-output -n notes_lifelog_rag \
  python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
PY
