#!/usr/bin/env bash
# One-command setup for a fresh Colab runtime:
#   !bash /content/drive/MyDrive/civicdesk/scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Installing pinned packages..."
pip install -q -r requirements.txt

echo "Checking GPU..."
python - <<'PY'
import torch
assert torch.cuda.is_available(), "No GPU. In Colab: Runtime > Change runtime type > T4 GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} | {p.total_memory / 1024**3:.2f} GiB")
PY
echo "Setup complete."
