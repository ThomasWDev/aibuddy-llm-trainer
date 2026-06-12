#!/usr/bin/env bash
# Laptop quickstart: train a tiny (~13M param) language model on your own
# machine — no GPU required — then sample from it.
#
#   1. Downloads the public-domain validation slice of the Pile (~450 MB).
#   2. Tokenizes a small sample of it into train/dev HDF5 files.
#   3. Pretrains the tiny laptop config for 300 steps on CPU (a few minutes).
#   4. Prints a text continuation from the freshly trained checkpoint.
#
# Usage (from the repo root, after `uv venv --python 3.11` +
# `uv pip install -r requirements-laptop.txt`):
#
#   bash scripts/laptop_quickstart.sh
#
# Every step is idempotent: re-running skips work that is already done.
set -euo pipefail

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -f "data/val/val.jsonl.zst" ]; then
  "$PY" scripts/data_download.py --train_max 0
fi

if [ ! -f "data/train/pile_train.h5" ]; then
  "$PY" scripts/data_preprocess.py \
    --train_dir data/val --val_dir data/val \
    --out_train_file data/train/pile_train.h5 \
    --out_val_file data/val/pile_dev.h5 \
    --max_data 2000
fi

PYTHONPATH=. "$PY" scripts/pretrain_base.py --config configs/laptop/pretrain.json

PYTHONPATH=. "$PY" scripts/chat.py \
  --ckpt checkpoints/laptop-tiny.pt --raw --device cpu \
  --max_new_tokens 80 --prompt "Once upon a time"
