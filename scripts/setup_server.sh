#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ] || [ ! -f .venv/bin/activate ]; then
  python3 -m venv .venv || true
  if [ ! -x .venv/bin/python ] || [ ! -f .venv/bin/activate ]; then
    echo "python3-venv is unavailable; using user-local virtualenv fallback."
    python3 -m pip install --user virtualenv
    python3 -m virtualenv --clear .venv
  fi
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/detect_gpu.py

if [ ! -f "weights/fashn-vton-1.5/model.safetensors" ]; then
  mkdir -p weights/fashn-vton-1.5
  python scripts/download_fashn_weights.py --weights-dir weights/fashn-vton-1.5
fi

echo "Setup complete. Copy .env.example to .env and set TELEGRAM_BOT_TOKEN manually."
