"""
config.py — every knob for the model and training lives here.

WHY dataclasses? So there are no "magic numbers" sprinkled through the code.
When you read model.py or train.py, any number that matters comes from one of
these configs, so you can see the whole architecture/training setup at a glance
and change it in exactly one place.

Two configs:
  - GPTConfig   : the *shape* of the neural network (architecture).
  - TrainConfig : how we *train* it (optimizer, schedule, batch sizes, paths).
"""

from dataclasses import dataclass, field
from pathlib import Path


# The project root = the folder that contains this `src/` directory.
# Using pathlib (not string concatenation) keeps paths correct on Windows,
# macOS and Linux/Colab alike.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GPTConfig:
    """Architecture hyperparameters — these define the network's shape.

    The defaults below are the "proven" ~30M-parameter config that reaches
    ~1.7 validation loss on TinyStories in ~30 min on a free Colab T4.
    """

    # Size of the vocabulary = number of distinct tokens the model can read/emit.
    # 50257 is the GPT-2 BPE vocabulary size (50256 merges + 1 <|endoftext|>).
    # This MUST match the tokenizer, because the output layer produces exactly
    # one score (logit) per vocabulary entry.
    vocab_size: int = 50257

    # Number of Transformer blocks stacked on top of each other (depth).
    # More layers = more capacity to compose features, but more compute/memory.
    n_layer: int = 6

    # Number of attention heads inside each block. The n_embd dimension is split
    # evenly across heads, so each head works in a n_embd/n_head = 384/6 = 64-dim
    # subspace. Each head can learn a different "relationship" between tokens.
    n_head: int = 6

    # The embedding dimension a.k.a. the model width / "d_model".
    # Every token is represented as a vector of this many numbers as it flows
    # through the network. Must be divisible by n_head (384 / 6 = 64). ✓
    n_embd: int = 384

    # Context length = how many tokens the model can look at at once.
    # During training we feed sequences of this length; attention lets each
    # position attend to up to this many previous positions. Longer context =
    # more memory (attention cost grows ~quadratically with block_size).
    block_size: int = 256

    # Dropout probability (regularization). 0.0 = off. TinyStories is large
    # relative to this small model, so we don't overfit and can leave it off.
    dropout: float = 0.0

    # Whether Linear layers and LayerNorms use a bias term.
    # GPT-2 uses biases; modern nanoGPT-style models often drop them (bias=False)
    # because they add parameters for little benefit and train slightly faster.
    bias: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters — optimizer, LR schedule, batching, I/O paths."""

    # --- Batching ---
    # How many independent sequences we process in parallel per step.
    # Bigger batch = smoother gradients but more VRAM. 32 x 256 tokens fits a T4.
    batch_size: int = 32

    # --- Optimization length ---
    max_iters: int = 5000          # total training steps for the full run
    eval_interval: int = 250       # every N steps, measure loss + maybe checkpoint
    eval_iters: int = 100          # how many batches to average for a loss estimate

    # --- AdamW learning-rate schedule ---
    # We use linear warmup then cosine decay (implemented in utils.py, Phase 3).
    learning_rate: float = 1e-3    # peak LR after warmup
    min_lr: float = 1e-4           # floor LR at the end of cosine decay
    warmup_iters: int = 100        # ramp LR 0 -> peak over these first steps

    # --- AdamW regularization / stability ---
    weight_decay: float = 0.1      # L2-style pull toward 0 on matmul weights
    grad_clip: float = 1.0         # clip global grad norm to avoid exploding steps

    # AdamW momentum betas. beta2=0.95 (vs the usual 0.999) is the GPT-2/nanoGPT
    # choice — it reacts a bit faster, which helps on shorter runs like ours.
    beta1: float = 0.9
    beta2: float = 0.95

    # --- Reproducibility ---
    seed: int = 1337               # fixed seed so runs are repeatable

    # --- Filesystem paths (all via pathlib) ---
    data_dir: Path = field(default=PROJECT_ROOT / "data")
    checkpoint_dir: Path = field(default=PROJECT_ROOT / "checkpoints")
    samples_dir: Path = field(default=PROJECT_ROOT / "samples")

    @property
    def train_bin(self) -> Path:
        return self.data_dir / "train.bin"

    @property
    def val_bin(self) -> Path:
        return self.data_dir / "val.bin"

    @property
    def best_ckpt(self) -> Path:
        return self.checkpoint_dir / "best.pt"
