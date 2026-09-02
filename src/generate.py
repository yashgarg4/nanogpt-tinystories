"""
generate.py — write stories with a trained model (autoregressive sampling).

Given a checkpoint, we encode a prompt into token ids, let the model extend it one
token at a time (see GPT.generate), then decode the ids back to text. The two dials
you'll play with:

  --temperature : creativity. <1 = safer/repetitive, 1 = the model's own
                  distribution, >1 = wilder. (Very low ≈ greedy/argmax.)
  --top-k       : only sample from the k most likely tokens (filters out the
                  nonsense tail). Smaller = more focused.

Usage:
  python -m src.generate --prompt "Once upon a time"
  python -m src.generate --prompt "The dragon" --temperature 0.7 --top-k 200 --num-samples 3
"""

from __future__ import annotations

import argparse

import torch

from .config import GPTConfig, TrainConfig
from .model import GPT
from .tokenizer import get_tokenizer
from .utils import enable_utf8_stdout

# Marker we print where the model emitted an end-of-text token (a story boundary).
EOT_TEXT = "<|endoftext|>"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path, device: str):
    """Rebuild the exact model from a checkpoint and load its trained weights."""
    # weights_only=False: this is OUR OWN checkpoint (it also holds optimizer state
    # and Python metadata), so we trust it. Never load untrusted checkpoints this
    # way. map_location moves tensors to the right device on load.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gcfg = GPTConfig(**ckpt["model_config"])  # same architecture we trained
    model = GPT(gcfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, gcfg, ckpt


def main() -> None:
    enable_utf8_stdout()
    tcfg = TrainConfig()

    p = argparse.ArgumentParser(description="Generate stories from a checkpoint")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200, help="0 disables top-k")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--ckpt", type=str, default=str(tcfg.best_ckpt))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument(
        "--full",
        action="store_true",
        help="print the full generation (don't stop at the first end-of-story)",
    )
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ckpt_path = tcfg.checkpoint_dir / "best.pt" if args.ckpt is None else args.ckpt
    from pathlib import Path

    if not Path(ckpt_path).exists():
        raise SystemExit(
            f"Checkpoint {ckpt_path} not found — train first (python -m src.train)."
        )

    model, gcfg, ckpt = load_model(ckpt_path, device)
    tok = get_tokenizer()
    top_k = None if args.top_k in (0, None) else args.top_k

    print(
        f"loaded {ckpt_path} | trained to iter {ckpt.get('iter_num', '?')} | "
        f"best val loss {ckpt.get('best_val_loss', float('nan')):.4f} | device={device}"
    )
    print(
        f"prompt={args.prompt!r} | temperature={args.temperature} | top_k={top_k} | "
        f"max_new_tokens={args.max_new_tokens}\n"
    )

    # Encode the prompt. If empty, seed with the EOT token so the model starts a
    # fresh document from scratch.
    start_ids = tok.encode(args.prompt) if args.prompt else [tok.eot_token]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # (1, T)

    for s in range(args.num_samples):
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
        text = tok.decode(y[0].tolist())
        # By default show just the first story: cut at the first EOT that appears
        # AFTER the prompt (the model signalling "the end").
        if not args.full and EOT_TEXT in text[len(args.prompt):]:
            cut = text.index(EOT_TEXT, len(args.prompt))
            text = text[:cut]
        print(f"----- sample {s + 1}/{args.num_samples} -----")
        print(text.strip())
        print()


if __name__ == "__main__":
    main()
