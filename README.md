# nanogpt-tinystories

**A ~30M-parameter GPT-2-style transformer, built from scratch in pure PyTorch,
trained on TinyStories to write coherent little children's stories.**

No `transformers` library, no `AutoModel`, no pretrained weights. Every core
component — tokenization, embeddings, causal self-attention, the transformer
block, the training loop, and sampling — is implemented and explained by hand.
The goal is **understanding transformers from the inside**.

> 🚧 Work in progress. This README grows as the project does; the full write-up
> (with a real generated story, a loss curve, and a sample gallery) lands in
> Phase 5. For the from-the-inside explanation of every component, see
> [`INTERNAL_NOTES.md`](INTERNAL_NOTES.md).

---

## What it will do

Train from random initialization until it can continue a prompt like
`"Once upon a time"` into a short, coherent story. Proven config reaches ~1.7
validation loss in ~30 minutes on a free Google Colab T4.

## From scratch means

The weights start as **random numbers**, and we write the attention mechanism,
the transformer blocks, and the training loop **ourselves**. We reuse GPT-2's
*tokenizer* (via `tiktoken`) because turning text into integers is a solved,
non-learning step — the **model** is what we're here to build and understand.

## Concepts you'll learn (see `INTERNAL_NOTES.md`)

Tokenization & BPE · token + position embeddings · scaled dot-product
self-attention · the causal mask · multi-head attention · the MLP / feed-forward
block · residual connections & pre-LayerNorm · weight tying · cross-entropy loss
· backprop & AdamW · warmup + cosine LR schedule · perplexity · temperature &
top-k sampling.

## Tech stack

Pure **PyTorch** for the model (the only deep-learning dependency) · `tiktoken`
(GPT-2 BPE) · `numpy` (memory-mapped `.bin` data) · `datasets` (download
TinyStories) · `tqdm` · `matplotlib`. **Not** used: `transformers` or any
prebuilt attention/GPT.

## Quick start (local)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Build the dataset (downloads TinyStories, tokenizes -> data/*.bin)
python -m src.data                   # or: --max-train-docs 20000 for a fast test

# 4. Verify the round-trip (decodes a random chunk from train.bin)
python -m src.data --verify-only
```

Training (Phase 3) and generation (Phase 4) commands land as those phases are
built. A one-click Colab notebook will live in `notebooks/`.

## Project structure

```
nanogpt-tinystories/
├── src/
│   ├── config.py       # GPTConfig + TrainConfig dataclasses
│   ├── tokenizer.py    # tiktoken GPT-2 BPE wrapper
│   ├── data.py         # download + tokenize -> data/train.bin, data/val.bin
│   ├── model.py        # GPT built from scratch            (Phase 2)
│   ├── train.py        # training loop                     (Phase 3)
│   ├── generate.py     # autoregressive sampling           (Phase 4)
│   └── utils.py        # param count, loss est., LR sched. (Phase 2–3)
├── notebooks/          # one-click Colab runner            (Phase 4)
├── data/               # train.bin / val.bin (gitignored)
├── checkpoints/        # saved models (gitignored)
├── samples/            # generated stories + loss curve
├── INTERNAL_NOTES.md   # the "textbook": every concept explained
├── requirements.txt
├── Makefile
└── README.md
```

## Acknowledgements

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
the [TinyStories](https://arxiv.org/abs/2305.07759) dataset/paper. This
implementation is written from scratch for learning.
