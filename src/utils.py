"""
utils.py — small helpers shared across the project.

Phase 2 adds:
  - enable_utf8_stdout(): make prints safe on Windows consoles.
  - count_parameters():   report the model's parameter count + a per-component
                          breakdown, correctly handling weight-tied tensors.

(get_batch / estimate_loss / the LR schedule arrive in Phase 3, when we train.)
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page (e.g. cp1252) and raise
    UnicodeEncodeError when asked to print characters outside it (check marks,
    em-dashes, smart quotes in generated text). Reconfiguring to UTF-8 avoids
    that. No-op where already UTF-8 or unsupported.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
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
