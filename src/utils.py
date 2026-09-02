"""
utils.py — small helpers shared across the project.

Phase 2 adds:
  - enable_utf8_stdout(): make prints safe on Windows consoles.
  - count_parameters():   report the model's parameter count + a per-component
                          breakdown, correctly handling weight-tied tensors.

(get_batch / estimate_loss / the LR schedule arrive in Phase 3, when we train.)
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext

import numpy as np
import torch

from .config import GPTConfig, TrainConfig


def enable_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page (e.g. cp1252) and raise
    UnicodeEncodeError when asked to print characters outside it (check marks,
    em-dashes, smart quotes in generated text). Reconfiguring to UTF-8 avoids
    that. No-op where already UTF-8 or unsupported.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            # line_buffering=True flushes on every newline, so progress is visible
            # live even when output is redirected to a file (e.g. background runs).
            stream.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _human(n: int) -> str:
    """Format a big integer as e.g. '30.0M' or '19.3M' for readability."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def count_parameters(model, verbose: bool = True) -> int:
    """Count trainable parameters and (optionally) print a component breakdown.

    Weight tying subtlety: our output layer `lm_head` SHARES its weight tensor
    with the token embedding `wte` (see model.py). A shared tensor must be
    counted ONCE, not twice. We therefore dedupe by the tensor's identity
    (id(p)); PyTorch's `named_parameters()` also dedupes by default, but we guard
    explicitly so the breakdown is unambiguous.

    Returns the total number of unique parameters.
    """
    seen: set[int] = set()
    groups: dict[str, int] = {}
    total = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:  # a tied/shared tensor we've already counted
            continue
        seen.add(id(p))
        # Group by the top-level attribute name: 'wte', 'wpe', 'blocks', 'ln_f'.
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
        total += p.numel()

    if verbose:
        # Friendly labels for each top-level component.
        labels = {
            "wte": "token embedding  (wte)",
            "wpe": "position embedding (wpe)",
            "blocks": "transformer blocks",
            "ln_f": "final LayerNorm   (ln_f)",
            "lm_head": "output head (lm_head)",
        }
        print("=" * 60)
        print("PARAMETER BREAKDOWN")
        for top, count in groups.items():
            print(f"  {labels.get(top, top):28s}: {count:>12,}  ({_human(count)})")
        # "Non-embedding" params = everything except the two lookup tables. This
        # is the number people usually quote as the model's "real" size, since
        # the token embedding is dominated by the large vocabulary, not depth.
        non_embed = total - groups.get("wte", 0) - groups.get("wpe", 0)
        print("-" * 60)
        print(f"  {'TOTAL (unique)':28s}: {total:>12,}  ({_human(total)})")
        print(f"  {'non-embedding':28s}: {non_embed:>12,}  ({_human(non_embed)})")
        print("  note: lm_head is weight-tied to wte, so it is counted once.")
        print("=" * 60)

    return total


# ===========================================================================
# Training helpers (Phase 3)
# ===========================================================================


def get_batch(
    split: str,
    gcfg: GPTConfig,
    tcfg: TrainConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one random mini-batch of (inputs x, targets y) from a .bin file.

    The .bin is one long 1-D stream of token ids. To make a training example we
    pick a random start position i and take a window of `block_size` tokens as
    the input x; the target y is the SAME window shifted right by one, because
    the model's job at each position t is to predict the token at t+1:

        stream:   ... a  b  c  d  e  f ...
        x (in):       a  b  c  d          (positions 0..T-1)
        y (target):   b  c  d  e          (each is the next token of x)

    So a single forward pass yields a prediction (and a loss) at EVERY position
    at once — very efficient. We draw `batch_size` such windows per batch.

    We re-open the memmap every call on purpose: keeping one long-lived memmap
    around while sampling can leak memory over a long run (a known NumPy/memmap
    quirk). Re-opening is cheap because memmap doesn't read the file into RAM.
    """
    path = tcfg.train_bin if split == "train" else tcfg.val_bin
    data = np.memmap(path, dtype=np.uint16, mode="r")

    # Random start indices, leaving room for a full block_size (+1 for the shift).
    ix = torch.randint(len(data) - gcfg.block_size, (tcfg.batch_size,))

    # Token ids are stored as uint16 but embeddings / cross-entropy need int64.
    x = torch.stack(
        [torch.from_numpy(data[i : i + gcfg.block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + gcfg.block_size].astype(np.int64))
            for i in ix
        ]
    )

    if device.startswith("cuda"):
        # pin_memory + non_blocking overlaps the host->GPU copy with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    gcfg: GPTConfig,
    tcfg: TrainConfig,
    device: str,
    eval_iters: int,
    ctx=None,
) -> dict[str, float]:
    """Estimate mean train and val loss over several random batches.

    WHY average many batches? A single batch's loss is noisy (it's just a random
    window of text). Averaging `eval_iters` batches gives a stable read on how the
    model is really doing. WHY no_grad + eval mode? We're only measuring, not
    learning: no_grad skips building the backprop graph (faster, less memory), and
    eval() disables dropout so the measurement is deterministic given the data.
    """
    if ctx is None:
        ctx = nullcontext()
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, gcfg, tcfg, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # restore training mode (re-enable dropout etc.)
    return out


def get_lr(it: int, tcfg: TrainConfig) -> float:
    """Learning-rate schedule: linear warmup, then cosine decay to min_lr.

    WHY a schedule at all? A fixed LR is a compromise: too high early on and the
    randomly-initialized model diverges; too low late and it crawls. So we shape
    the LR over time:

      1) WARMUP (first `warmup_iters` steps): ramp LR linearly 0 -> peak. Early
         gradients are large and noisy; easing in avoids blowing up the weights.
      2) COSINE DECAY (until `max_iters`): smoothly lower LR from peak to `min_lr`
         following half a cosine wave. Big steps early to make fast progress,
         then ever-smaller steps to settle into a good minimum without bouncing.
      3) After `max_iters`: hold at `min_lr`.
    """
    # Degenerate config guard (e.g. warmup >= total): just use peak LR.
    if tcfg.max_iters <= tcfg.warmup_iters:
        return tcfg.learning_rate

    # 1) linear warmup
    if it < tcfg.warmup_iters:
        return tcfg.learning_rate * (it + 1) / tcfg.warmup_iters
    # 3) past the decay horizon -> floor
    if it > tcfg.max_iters:
        return tcfg.min_lr
    # 2) cosine decay from peak -> min_lr
    decay_ratio = (it - tcfg.warmup_iters) / (tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)
