"""
train.py — the training loop that teaches the GPT to predict the next token.

THE LEARNING LOOP IN PLAIN WORDS (one iteration):
  1. get_batch: grab a random chunk of the training stream (inputs x, targets y).
  2. forward:   the model predicts next-token logits for every position; compare
                to y with cross-entropy -> a single number `loss` = "how wrong".
  3. backward:  loss.backward() computes, for every weight, the gradient = "which
                way, and how much, to nudge this weight to reduce the loss".
  4. clip:      scale gradients down if their combined size is too big (stops a
                rare huge step from blowing up training).
  5. step:      the AdamW optimizer nudges every weight a little (size set by the
                current learning rate from our schedule).
  6. zero_grad: clear the gradients so they don't accumulate into the next step.
  Repeat thousands of times; the loss falls from ~10.8 (random) toward ~1.7.

Every `eval_interval` steps we measure train+val loss on held-out batches and
save a checkpoint whenever the validation loss hits a new best.

Usage:
  python -m src.train                              # full run (config defaults)
  python -m src.train --max-iters 300 --eval-interval 50 --batch-size 16  # smoke
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from .config import GPTConfig, TrainConfig
from .model import GPT
from .utils import count_parameters, enable_utf8_stdout, estimate_loss, get_batch, get_lr


def pick_device(requested: str) -> str:
    """Choose the compute device. 'auto' -> cuda if available, else cpu."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> str:
    """Pick the training precision.

    - CPU: float32 (no autocast; half precision on CPU is slow/unsupported).
    - CUDA: prefer bfloat16 when the GPU supports it (Ampere+, e.g. A100). bf16
      needs no loss-scaling and is very stable. On older cards (e.g. the Colab
      T4, which lacks native bf16) we fall back to float16, which is fast there
      but needs a GradScaler (handled below).
    """
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


def save_checkpoint(path, model, optimizer, gcfg, it, best_val_loss, history):
    """Persist everything needed to resume training or to generate later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # Store the architecture so generate.py can rebuild an identical model.
            "model_config": asdict(gcfg),
            "iter_num": it,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        path,
    )


def main() -> None:
    enable_utf8_stdout()

    tcfg = TrainConfig()
    gcfg = GPTConfig()

    # --- CLI overrides (handy for a quick smoke run vs the full run) ---
    p = argparse.ArgumentParser(description="Train the GPT on TinyStories")
    p.add_argument("--max-iters", type=int, default=tcfg.max_iters)
    p.add_argument("--eval-interval", type=int, default=tcfg.eval_interval)
    p.add_argument("--eval-iters", type=int, default=tcfg.eval_iters)
    p.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    p.add_argument("--learning-rate", type=float, default=tcfg.learning_rate)
    p.add_argument("--warmup-iters", type=int, default=tcfg.warmup_iters)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=tcfg.seed)
    args = p.parse_args()

    # Apply overrides back onto the config objects so every helper sees them.
    tcfg.max_iters = args.max_iters
    tcfg.eval_interval = args.eval_interval
    tcfg.eval_iters = args.eval_iters
    tcfg.batch_size = args.batch_size
    tcfg.learning_rate = args.learning_rate
    tcfg.warmup_iters = args.warmup_iters
    tcfg.seed = args.seed

    # --- Reproducibility ---
    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)
    # Allow TF32 matmuls on Ampere+ for speed (harmless elsewhere).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Device & precision ---
    device = pick_device(args.device)
    dtype = pick_dtype(device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    # autocast runs eligible ops in low precision (faster, less memory) while
    # keeping a float32 master copy of weights. nullcontext = "do nothing" on CPU.
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=pt_dtype)
    )
    # GradScaler is only needed for float16 (it prevents tiny grads underflowing
    # to zero). It's a no-op passthrough when disabled (bf16 / cpu).
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    if not tcfg.train_bin.exists():
        raise SystemExit(
            f"{tcfg.train_bin} not found — run `python -m src.data` first (Phase 1)."
        )

    print(f"device={device} | dtype={dtype} | batch_size={tcfg.batch_size} | "
          f"block_size={gcfg.block_size} | max_iters={tcfg.max_iters}")

    # --- Model & optimizer ---
    model = GPT(gcfg).to(device)
    count_parameters(model)
    optimizer = model.configure_optimizers(
        weight_decay=tcfg.weight_decay,
        learning_rate=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
        device_type=device_type,
    )

    best_val_loss = float("inf")
    history: list[dict] = []  # records (iter, train_loss, val_loss, lr) at evals
    t0 = time.time()

    # --- The loop. We include iter==max_iters so we evaluate the final model. ---
    for it in range(tcfg.max_iters + 1):
        # Set this step's learning rate from the warmup+cosine schedule.
        lr = get_lr(it, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic evaluation + checkpointing (also at step 0 and the last step).
        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            losses = estimate_loss(model, gcfg, tcfg, device, tcfg.eval_iters, ctx)
            dt = time.time() - t0
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {lr:.2e} | {dt:6.1f}s"
            )
            history.append(
                {"iter": it, "train": losses["train"], "val": losses["val"], "lr": lr}
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    tcfg.best_ckpt, model, optimizer, gcfg, it, best_val_loss, history
                )
                print(f"          -> new best val loss {best_val_loss:.4f}; saved {tcfg.best_ckpt.name}")

        if it == tcfg.max_iters:
            break

        # --- one training step ---
        x, y = get_batch("train", gcfg, tcfg, device)
        with ctx:
            _, loss = model(x, y)          # forward: predictions + loss
        scaler.scale(loss).backward()      # backward: gradients for every weight
        if tcfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)     # undo loss-scaling before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)             # AdamW update
        scaler.update()                    # adjust the fp16 loss scale
        optimizer.zero_grad(set_to_none=True)  # clear grads for next step

    # --- Summary: print the loss curve values (the Phase 3 completion signal) ---
    print("\n" + "=" * 60)
    print("LOSS CURVE (val loss should fall from ~10.8 toward ~4 or lower)")
    print(f"{'step':>6} | {'train':>8} | {'val':>8} | {'lr':>9}")
    print("-" * 60)
    for h in history:
        print(f"{h['iter']:>6} | {h['train']:>8.4f} | {h['val']:>8.4f} | {h['lr']:>9.2e}")
    print("-" * 60)
    print(f"best val loss: {best_val_loss:.4f} | checkpoint: {tcfg.best_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
