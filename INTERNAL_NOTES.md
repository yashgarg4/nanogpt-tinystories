# INTERNAL_NOTES.md — building a GPT from scratch, explained

This file is both a **build log** and a **textbook**. As we implement each piece
of a GPT language model by hand, this document explains *what it is*, *why it's
there*, and *what breaks without it* — in plain language first, then the math.

If you read this top to bottom after the project is done, it should teach you how
a transformer works from the inside.

> Status legend: ✅ built & explained · 🚧 placeholder, filled in a later phase

---

## 1. What this project is, and the "build it to understand it" goal

We are training a **~30-million-parameter GPT-2-style transformer** from random
initialization on the **TinyStories** dataset, so it learns to generate coherent
short children's stories.

The point is **not** to get a good story generator (a tiny model on tiny stories
is a toy). The point is **understanding**. We implement every core component —
tokenization, embeddings, causal self-attention, the transformer block, the
training loop, backprop, sampling — **by hand in pure PyTorch**, with no
`transformers` library, no `AutoModel`, and no pretrained weights.

**"From scratch" means:** the weights start as random numbers, and *we* write the
attention mechanism, the blocks, and the training loop ourselves. (We do reuse
GPT-2's *tokenizer* via `tiktoken`, because turning text into integers is a
solved, non-learning step and not the thing we're here to understand — the
**model** is.)

Target hardware: a **free Google Colab T4 (16GB)**. Everything is sized to fit
there; the "does it work?" steps also run on CPU.

---

## 2. The big picture: text → numbers → trained model → new text

Here is the entire pipeline in plain words. Every later section zooms into one of
these arrows.

```
  ┌─────────────┐   tokenize    ┌────────────┐   embed    ┌──────────────┐
  │  raw text   │ ────────────► │ token ids  │ ─────────► │  vectors     │
  │ "Once upon" │  (BPE, Phase1)│ [7454,2402]│ (Phase 2)  │ (meaning as  │
  └─────────────┘               └────────────┘            │  numbers)    │
                                                          └──────┬───────┘
                                                                 │  N transformer
                                                                 │  blocks: attention
                                                                 ▼  + MLP (Phase 2)
  ┌─────────────┐   sample      ┌────────────┐   logits   ┌──────────────┐
  │  new text   │ ◄──────────── │ next-token │ ◄───────── │ contextualized│
  │ (a story)   │  (Phase 4)    │ probs      │  (softmax) │  vectors      │
  └─────────────┘               └────────────┘            └──────────────┘
```

Step by step:

1. **Tokenize** (Phase 1). Split text into subword *tokens* and map each to an
   integer id. `"Once upon a time"` → `[7454, 2402, 257, 640]`.
2. **Embed** (Phase 2). Look up a learned vector for each token id, and add a
   vector encoding its *position*. Now each token is a point in a high-dimensional
   space where "meaning" and "where it sits" are both represented as numbers.
3. **Transformer blocks** (Phase 2). Each block lets every token **attend** to
   earlier tokens (gather context) and then **think** about what it gathered (an
   MLP). Stack several blocks so the model can build up increasingly abstract
   features.
4. **Predict** (Phase 2). A final linear layer turns each token's vector into
   **logits**: one score per vocabulary entry = "how likely is each possible next
   token?". A softmax turns logits into probabilities.
5. **Train** (Phase 3). Show the model real text, ask it to predict each next
   token, measure how wrong it is (**cross-entropy loss**), and nudge all the
   weights to be a little less wrong (**backprop + AdamW**). Repeat millions of
   times.
6. **Generate** (Phase 4). Feed a prompt, get next-token probabilities, **sample**
   one token, append it, and repeat — the model writes one token at a time.

The model is "just" learning one skill absurdly well: **given the tokens so far,
predict the next token.** Everything story-like emerges from doing that well.

---

## 3. Concepts, explained simply

### 3.1 Tokenization and BPE — why not just characters or words? ✅

A neural net only does arithmetic on numbers, so text must become integers first.
There are three ways to chop text into units ("tokens"):

| Scheme | Vocab size | Sequence length | "Unknown" words? |
|---|---|---|---|
| **Characters** | tiny (~100) | very long | never |
| **Whole words** | huge (millions) | short | yes — any unseen word is lost |
| **Subword (BPE)** | fixed, modest (50257) | medium | never |

- **Characters** (`O`,`n`,`c`,`e`): the model wastes effort learning to spell
  before it can learn meaning, and sequences get very long (attention cost grows
  with the *square* of length, so long sequences are expensive).
- **Whole words**: short sequences, but the vocabulary explodes and any word not
  seen in training (a typo, a name, a new word) becomes an **`<unk>`** — real
  information is thrown away.
- **Subword / BPE** (what GPT-2 and we use) is the sweet spot.

**BPE = Byte-Pair Encoding.** Intuition: start from raw bytes, then repeatedly
find the most frequent adjacent pair of tokens in a big corpus and **merge** it
into a new token. Do this ~50,000 times. The result:

- Common words become a **single token** (`" the"`, `" time"`), keeping sequences
  short.
- Rare words split into a few **reusable pieces** (`" tokenization"` →
  `" token" + "ization"`).
- Because the base units are bytes, **no word is ever "unknown"** — worst case it
  falls back to byte-level pieces.

GPT-2's vocabulary has **50257** entries: 50256 learned tokens + 1 special
**`<|endoftext|>`** (id **50256**) that marks a document boundary. We append it
between stories so the model can learn where a story ends (and later learns to
*stop* generating).

A subtlety worth noticing: BPE tokens usually include the **leading space**, so
`" time"` (with space) and `"time"` (no space) are different tokens. That's how a
space-delimited language is encoded efficiently.

We use `tiktoken` (a fast Rust tokenizer, *not* a neural net) so tokenization is
correct and fast; see `src/tokenizer.py`.

### 3.2 Embeddings — token embeddings AND position embeddings, why both? ✅

After tokenizing, we have integers like `7454`. But an integer id is meaningless
as a *quantity* — token 7454 isn't "more" than token 640. We need to turn each id
into a vector the network can compute with. That's an **embedding**.

**Token embedding (`wte`, "weights of token embeddings").**
Think of a big lookup table with `vocab_size` rows (50257) and `n_embd` columns
(384). Row *i* is the learned vector for token id *i*. "Look up the embedding" =
"grab that row". These 384 numbers per token **start random** and are **learned**
during training, so that tokens used in similar ways drift to similar vectors
(e.g. `" cat"` and `" dog"` end up near each other). This table is where the model
stores "what each token means".

**Position embedding (`wpe`, "weights of position embeddings").**
Here's the catch: the attention mechanism (Section 3.3) is **permutation-
invariant** — by itself it treats the input as an unordered *set* of tokens. It
literally can't tell `"dog bites man"` from `"man bites dog"`. But order is
everything in language! So we add a second learned table with `block_size` rows
(256, one per position) × `n_embd` columns (384). Row *p* is a learned vector that
means "I am at position *p*".

**Why add them?** For each token we compute:

```
x = token_embedding[id] + position_embedding[position]
```

Now a single vector carries **both** *what* the token is and *where* it sits. The
network can use both. (Addition works because the model has plenty of dimensions;
it learns to keep the two kinds of information usably separable.)

> Modern models often replace learned position embeddings with **RoPE** (rotary
> embeddings), which we may try as a Phase 5 stretch. GPT-2 uses the simple
> learned table, so that's our baseline.

Shapes to keep in mind (B = batch, T = sequence length):
`token ids (B,T) → embeddings (B,T,384) → ... → logits (B,T,50257)`.

### 3.3 Self-attention — the core idea, then the math 🚧 (Phase 2)

### 3.4 Multi-head attention — why several heads beat one 🚧 (Phase 2)

### 3.5 The MLP / feed-forward block and the 4× expansion 🚧 (Phase 2)

### 3.6 Residual connections + LayerNorm (pre-norm) 🚧 (Phase 2)

### 3.7 Weight tying — sharing input embedding & output projection 🚧 (Phase 2)

### 3.8 The training loop: forward, loss, backward, step 🚧 (Phase 3)

### 3.9 Learning-rate schedule (warmup + cosine) 🚧 (Phase 3)

### 3.10 Perplexity and how to read a loss curve 🚧 (Phase 3)

### 3.11 Sampling: temperature and top-k 🚧 (Phase 4)

---

## 4. Architecture diagram (full model) 🚧

Filled in during Phase 2 once the model is built. Sketch of the target:

```
              token ids (B, T)
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
 token embed (wte)          position embed (wpe)
   (B,T,384)                   (T,384)
      └─────────────┬─────────────┘
                    ▼   (add)
                dropout
                    ▼
         ┌───────────────────┐
         │  Transformer block│   × n_layer (6)
         │  pre-LN → attn → +│
         │  pre-LN → MLP  → +│
         └───────────────────┘
                    ▼
              final LayerNorm
                    ▼
          lm_head (Linear 384→50257)   ← weight-tied to wte
                    ▼
              logits (B, T, 50257)
```

---

## 5. Build log (append after every phase and every bug)

### Phase 1 — Tokenizer + data pipeline

**Goal:** TinyStories tokenized into `data/train.bin` / `data/val.bin`; a decoded
random chunk round-trips into readable story text.

**What was built**
- `src/config.py` — `GPTConfig` (architecture) and `TrainConfig` (training + paths)
  as dataclasses, so every hyperparameter lives in one place (no magic numbers).
- `src/tokenizer.py` — thin GPT-2 BPE wrapper over `tiktoken`: `encode`, `decode`,
  and `encode_with_eot` (appends the `<|endoftext|>` separator). Includes a
  round-trip self-test.
- `src/data.py` — downloads TinyStories via HuggingFace `datasets`, tokenizes
  every story (appending EOT), and writes both splits as flat **uint16** arrays
  via `np.memmap`, in shards so RAM stays bounded. A `verify()` step decodes a
  random chunk and prints token counts. Supports `--max-train-docs` for a fast
  end-to-end test before committing to the full ~2GB download.

**Key decisions**
- **uint16** for the .bin: GPT-2's max id (50256) fits in 2 bytes; halves file
  size vs uint32 / quarters vs int64, and memory-mapping lets us train on more
  data than fits in RAM.
- **EOT between stories** so document boundaries are learnable.
- **`encode_ordinary`** (not `encode`) on raw text, so literal substrings are
  never mis-parsed as special tokens; we add EOT ourselves, deliberately.
- Reuse GPT-2's tokenizer instead of training our own — tokenization isn't the
  learning target.

**Results (this run)**
- `train.bin` = **473,992,236 tokens** (948 MB) · `val.bin` = **4,765,918 tokens** (9.5 MB).
- Both stored as uint16; max token id = 50256 (< vocab 50257 ✓); EOT separators present.
- A random 256-token chunk decoded into coherent story text and even showed a
  `[END OF STORY]` boundary between two stories — full round-trip confirmed.

**Bugs / gotchas** _(append as they happen)_
- **Windows console `UnicodeEncodeError`.** Printing a `✓` crashed because Windows
  stdout defaults to the legacy cp1252 code page. Real story text can contain
  smart quotes / em-dashes and would crash the same way. Fix: call
  `sys.stdout.reconfigure(encoding="utf-8")` at each script's entry point (a
  no-op on platforms that already use UTF-8). *Lesson:* never assume the console
  is UTF-8 on Windows.
- **`load_dataset(split="validation")` still fetches everything.** HuggingFace
  downloads the whole repo's files and "generates" both splits regardless of the
  split you ask for, so the first call pays the full ~2GB / ~124s cost. It's
  cached afterward, so later runs are instant.
- **HF symlink warning on Windows.** Without Developer Mode/admin, the HF cache
  can't use symlinks and warns (uses a bit more disk). Harmless; silence with
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **The write phase is disk-I/O-bound, not CPU-bound.** Tokenizing 2.1M stories
  across 6 processes was fast; copying the ~948MB token stream into the
  memory-mapped file through 1024 shards took the bulk of the time (~1.2
  shards/sec). Expected for a ~1GB sequential write.

**Concepts learned this phase:** Tokenization & BPE (§3.1), Embeddings (§3.2, the
table the ids will index into next phase).

**Interview Q&A**
- *Why BPE over characters or words?* Characters make sequences too long (and
  attention is quadratic in length); words explode the vocabulary and create
  unknown-word failures. BPE keeps a fixed modest vocabulary, short-ish
  sequences, and never produces an unknown token (byte fallback).
- *What is the `<|endoftext|>` token for?* It separates documents so the model can
  learn where stories begin/end and, at generation time, learn to stop.
- *Why store tokens as uint16?* The vocabulary (50257) fits in 16 bits, so it's
  the smallest integer type that works — minimizing disk and memory, and enabling
  memory-mapped training on large corpora.
- *Why do we need position embeddings at all?* Self-attention is order-agnostic;
  without a position signal the model couldn't distinguish `"dog bites man"` from
  `"man bites dog"`.
