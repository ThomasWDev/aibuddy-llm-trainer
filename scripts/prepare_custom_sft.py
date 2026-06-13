"""
Pack a CUSTOM instruction dataset (JSONL) into SFT train/dev HDF5 files.

Bridges "my own data" into the same packed format ``scripts/train_sft.py``
consumes, using the exact same chat template + packing code as the public-data
pipeline (``prepare_sft_data.py``) — so a model fine-tuned on your data behaves
identically at inference time.

Input format — one JSON object per line:

    {"prompt": "Write a meta description for ...", "response": "..."}

This is deliberately the same shape AI assistants tend to produce when asked
for "JSONL with prompt/response keys", so generated datasets drop straight in.

Example (laptop):
    PYTHONPATH=. python scripts/prepare_custom_sft.py \
        --jsonl data/my_dataset.jsonl --context_length 256 \
        --out_train data/sft_custom.h5 --out_dev data/sft_custom_dev.h5
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from src.post_training.chat_template import encode_chat
from src.post_training.sft import pack_examples


def jsonl_to_examples(path: str, context_length: int) -> tuple[list, int]:
    """Tokenize prompt/response rows through the chat template.

    Rows that are blank, malformed JSON, missing either field, or longer than
    ``context_length`` once tokenized are SKIPPED (never truncated — a response
    cut mid-sentence teaches the model to stop mid-sentence).

    Returns:
        (examples, skipped) where examples are ``(ids, loss_mask)`` tuples.
    """
    examples: list[tuple[list[int], list[int]]] = []
    skipped = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompt = str(obj.get("prompt", "") or "").strip()
            response = str(obj.get("response", "") or "").strip()
            if not prompt or not response:
                skipped += 1
                continue
            ids, mask = encode_chat(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            )
            if len(ids) > context_length:
                skipped += 1
                continue
            examples.append((ids, mask))
    return examples, skipped


def write_packed(examples: list, context_length: int, out_path: str) -> int:
    tokens, masks = pack_examples(examples, context_length)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("loss_mask", data=masks)
    print(f"  wrote {tokens.shape[0]} packed rows x {context_length} -> {out_path}")
    return tokens.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", action="append", required=True,
                   help="path to a prompt/response JSONL file (repeatable)")
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--out_train", default="data/sft_custom.h5")
    p.add_argument("--out_dev", default="data/sft_custom_dev.h5")
    p.add_argument("--dev_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    examples: list = []
    total_skipped = 0
    for path in args.jsonl:
        ex, skipped = jsonl_to_examples(path, args.context_length)
        print(f"{path}: {len(ex)} usable examples ({skipped} skipped)")
        examples.extend(ex)
        total_skipped += skipped

    if not examples:
        raise SystemExit("No usable examples — check the JSONL has prompt/response keys.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * args.dev_frac))
    dev, train = examples[:n_dev], examples[n_dev:]

    write_packed(train, args.context_length, args.out_train)
    write_packed(dev, args.context_length, args.out_dev)
    print(f"Done. {len(train)} train / {len(dev)} dev examples ({total_skipped} skipped).")


if __name__ == "__main__":
    main()
