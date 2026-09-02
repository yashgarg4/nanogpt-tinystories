"""
data.py — download TinyStories, tokenize it, and write train.bin / val.bin.

THE BIG PICTURE (what this file produces and why)
-------------------------------------------------
A language model trains on a giant 1-D stream of token ids. So our job here is:

    raw stories (text)  ->  token ids (ints)  ->  one flat uint16 array on disk

We concatenate every story's tokens end-to-end into a single long sequence,
putting an <|endoftext|> (EOT, id 50256) token between stories so the model can
learn where a story starts and ends. Two files come out:

    data/train.bin   (the big one — used to learn)
    data/val.bin     (held-out — used only to measure generalization)

WHY a raw .bin of uint16 instead of, say, a CSV or keeping the HF dataset?
  - Training reads RANDOM fixed-length windows out of this stream millions of
    times. A flat array of fixed-width integers makes that a trivial slice.
  - uint16 = 2 bytes and covers 0..65535. GPT-2's largest id is 50256, so every
    token fits. Using uint16 instead of int64 shrinks the file 4x.
  - We memory-map the file (np.memmap): the OS pages in only the bytes we touch,
    so we can train on a multi-GB corpus without loading it all into RAM.

This file is CPU/IO work only — no PyTorch, no GPU needed for Phase 1.

Usage:
    python -m src.data                      # full dataset (download + tokenize)
    python -m src.data --max-train-docs 20000 --max-val-docs 2000   # quick subset
    python -m src.data --verify-only        # just decode a chunk from train.bin
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from .config import TrainConfig
from .tokenizer import get_tokenizer

# TinyStories on the HuggingFace Hub. Simple text dataset of short, synthetic
# children's stories using a small vocabulary — perfect for training a tiny LM.
HF_DATASET = "roneneldan/TinyStories"

# uint16 can store 0..65535. We assert every token id fits below this so the
# .bin file is unambiguous. (GPT-2 max id is 50256 — comfortably under.)
UINT16_MAX = np.iinfo(np.uint16).max  # 65535


def _process(example: dict) -> dict:
    """Tokenize ONE story and append the EOT separator.

    This is a top-level function (not a lambda / nested def) on purpose: when we
    parallelize with datasets.map(num_proc>1), each worker process must be able
    to import and pickle this function by name. Keeping it module-level makes
    that work on Windows (spawn) as well as Linux (fork).

    Returns the token ids and their count; `len` lets us size the output array
    exactly before writing, without holding every story's ids in RAM at once.
    """
    tok = get_tokenizer()  # cheap; re-created once per worker process
    ids = tok.encode_with_eot(example["text"])
    return {"ids": ids, "len": len(ids)}


def prepare(
    num_proc: int,
    max_train_docs: int | None,
    max_val_docs: int | None,
) -> None:
    """Download TinyStories, tokenize both splits, and write the .bin files."""
    # Import here (not at top) so that `--verify-only` doesn't require `datasets`
    # to be importable, and so the heavy import only happens when we actually
    # prepare data.
    from datasets import load_dataset

    cfg = TrainConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    tok = get_tokenizer()
    # Sanity: our config's vocab_size must match the tokenizer, or the model's
    # output layer won't line up with the ids we produce here.
    print(f"Tokenizer: GPT-2 BPE | vocab_size={tok.vocab_size} | eot={tok.eot_token}")

    print(f"Loading '{HF_DATASET}' from HuggingFace (first run downloads ~2GB)...")
    # TinyStories ships with 'train' and 'validation' splits.
    dataset = load_dataset(HF_DATASET)

    # Optionally shrink the splits for a fast end-to-end pipeline test.
    # .select(range(N)) keeps the first N stories; enough to prove correctness.
    if max_train_docs is not None:
        n = min(max_train_docs, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    if max_val_docs is not None:
        n = min(max_val_docs, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(n))

    print(
        f"Stories to tokenize: train={len(dataset['train']):,} "
        f"val={len(dataset['validation']):,}"
    )

    # Tokenize every story. datasets caches the result on disk, so re-running is
    # instant. remove_columns drops the original 'text' to save space.
    print(f"Tokenizing with num_proc={num_proc} (this is the slow part)...")
    tokenized = dataset.map(
        _process,
        remove_columns=["text"],
        desc="tokenizing",
        num_proc=num_proc,
    )

    # HuggingFace names the held-out split 'validation'; we write it as val.bin.
    split_to_filename = {"train": cfg.train_bin, "validation": cfg.val_bin}

    for split, dset in tokenized.items():
        filename = split_to_filename[split]

        # Total number of tokens in this split = sum of every story's length.
        # dtype=uint64 so the sum can't overflow for large corpora.
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))

        # Pre-allocate the exact-size file on disk and memory-map it. mode="w+"
        # creates/overwrites. We then fill it in chunks, never holding the whole
        # token stream in RAM.
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        # Write in shards to keep memory bounded: concatenate one shard's ids,
        # copy them into the right slice of the memmap, repeat.
        total_shards = min(1024, len(dset)) or 1
        write_idx = 0
        for shard_idx in tqdm(range(total_shards), desc=f"writing {filename.name}"):
            shard = dset.shard(
                num_shards=total_shards, index=shard_idx, contiguous=True
            ).with_format("numpy")
            # Each element of shard["ids"] is a per-story array; glue them.
            shard_ids = np.concatenate(shard["ids"])

            # Guard: every id must fit in uint16, or the file would be corrupt.
            if shard_ids.size and shard_ids.max() > UINT16_MAX:
                raise ValueError(
                    f"Token id {shard_ids.max()} exceeds uint16 max {UINT16_MAX}."
                )

            arr[write_idx : write_idx + len(shard_ids)] = shard_ids
            write_idx += len(shard_ids)

        arr.flush()  # ensure everything is written to disk
        size_mb = filename.stat().st_size / 1e6
        print(
            f"  wrote {filename}  |  {arr_len:,} tokens  |  {size_mb:,.1f} MB"
        )

    print("\nData preparation complete.")


def verify(num_tokens: int = 256, seed: int = 1337) -> None:
    """Load a RANDOM chunk from train.bin, decode it, and print it.

    This is the Phase 1 acceptance test: if the decoded text reads like a
    coherent TinyStories snippet, then encode -> write -> read -> decode all
    round-tripped correctly and the .bin layout is right.
    """
    cfg = TrainConfig()
    tok = get_tokenizer()

    if not cfg.train_bin.exists():
        print(
            f"ERROR: {cfg.train_bin} not found. Run `python -m src.data` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read-only memmap: we don't load the whole file, just page in what we slice.
    train = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
    val = (
        np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")
        if cfg.val_bin.exists()
        else None
    )

    print("=" * 70)
    print("TOKEN COUNTS")
    print(f"  train.bin : {len(train):,} tokens")
    if val is not None:
        print(f"  val.bin   : {len(val):,} tokens")
    print("=" * 70)

    # Pick a random start position and decode the next `num_tokens` tokens.
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(train) - num_tokens)))
    chunk = train[start : start + num_tokens].tolist()

    print(f"Decoded chunk from train.bin at offset {start:,} ({num_tokens} tokens):")
    print("-" * 70)
    # The chunk may start/end mid-story and mid-word — that's expected, since it's
    # a random window into one long concatenated stream. It should still read as
    # recognizable story text. We show <|endoftext|> where story boundaries fall.
    text = tok.decode(chunk)
    print(text.replace("<|endoftext|>", "\n[END OF STORY]\n"))
    print("-" * 70)


def main() -> None:
    # Windows consoles default to a legacy code page (e.g. cp1252) that crashes
    # when asked to print characters outside it (smart quotes, em-dashes, and the
    # like appear in real story text). Force UTF-8 output. No-op if unsupported.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Prepare TinyStories -> .bin files")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel tokenizer workers (default: half your CPU cores)",
    )
    parser.add_argument(
        "--max-train-docs",
        type=int,
        default=None,
        help="limit training stories (for a quick pipeline test); default: all",
    )
    parser.add_argument(
        "--max-val-docs",
        type=int,
        default=None,
        help="limit validation stories; default: all",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip preparation; just decode a random chunk from train.bin",
    )
    args = parser.parse_args()

    if not args.verify_only:
        prepare(
            num_proc=args.num_proc,
            max_train_docs=args.max_train_docs,
            max_val_docs=args.max_val_docs,
        )

    verify()


if __name__ == "__main__":
    main()
