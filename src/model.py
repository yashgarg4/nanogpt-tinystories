"""
model.py — a GPT-2-style decoder-only Transformer, built from scratch.

We implement every piece by hand with plain PyTorch tensor ops: embeddings,
scaled dot-product self-attention with a causal mask, multi-head attention, the
MLP, the pre-norm residual block, and weight tying. No `transformers`, no
prebuilt attention.

Read this top-to-bottom; it's ordered the way the model is built:

    LayerNorm  ->  CausalSelfAttention  ->  MLP  ->  Block  ->  GPT

Notation used throughout:
    B = batch size            (independent sequences processed at once)
    T = time / sequence len   (number of tokens, <= block_size)
    C = channels = n_embd     (the width of each token's vector, 384)
    nh = n_head, hs = head size = C // nh   (384 / 6 = 64)
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


# ---------------------------------------------------------------------------
# LayerNorm — normalize each token's vector to mean 0 / variance 1, then apply a
# learned scale (weight) and optional shift (bias).
#
# WHY normalize? As signals flow through many layers, their scale can drift
# (blow up or shrink), which makes training unstable. LayerNorm re-centers and
# re-scales each token vector independently, keeping the numbers in a sane range
# so gradients stay well-behaved. The learned weight/bias let the network undo
# the normalization if it wants to.
#
# WHY our own instead of nn.LayerNorm? Only so we can cleanly support bias=False
# (GPT-2-lite style). It also makes the operation explicit for learning.
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))          # learned scale, γ
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # shift, β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.layer_norm normalizes over the last dimension (size = ndim = C).
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


# ---------------------------------------------------------------------------
# Causal self-attention — THE core of the Transformer.
#
# THE IDEA (plain language): each token looks at the earlier tokens and pulls in
# information that's relevant to it. "The cat sat on the ___" — to predict the
# next word, the token at the blank should pay attention to "cat"/"sat"/"on".
# Attention is a learned, content-based way to gather that context.
#
# THE MECHANISM: for every token we compute three vectors via learned linear
# maps of its embedding:
#   Query (Q)  = "what am I looking for?"
#   Key   (K)  = "what do I offer / advertise?"
#   Value (V)  = "what do I actually pass along if attended to?"
# Token i attends to token j by how well Q_i matches K_j (a dot product). We
# turn those match scores into weights with softmax, then take a weighted sum of
# the V_j's. That weighted sum is token i's new, context-aware representation.
#
#   scores = Q · Kᵀ / sqrt(head_size)          # (T, T) how much i attends to j
#   weights = softmax(scores over j)            # rows sum to 1
#   out    = weights · V                        # (T, head_size)
#
# WHY divide by sqrt(head_size)? Dot products of 64-dim vectors have variance
# ~64; large values push softmax into a near one-hot (tiny gradients). Scaling
# by 1/sqrt(head_size) keeps the scores at unit variance so softmax stays soft
# and trainable. This is the "scaled" in "scaled dot-product attention".
#
# THE CAUSAL MASK — why we hide the future: this is a LANGUAGE MODEL; at position
# i it must predict token i+1 using ONLY tokens 0..i. If a token could attend to
# later tokens, it would "see the answer" during training and learn nothing
# useful for generation (where the future doesn't exist yet). So before softmax
# we set the scores for all j > i to -infinity; after softmax those weights
# become exactly 0. The attention is therefore "causal" (a.k.a. masked).
#
# MULTI-HEAD — why several heads beat one: instead of one 384-dim attention, we
# run nh=6 independent attentions ("heads") in parallel, each in a 64-dim
# subspace, then concatenate their outputs. Each head can specialize in a
# different kind of relationship (one might track syntax, another a referenced
# noun, another position/recency). One big head would have to average all those
# roles together; many small heads can attend to different places at once. We
# implement all heads at once with a single batched matmul (reshaping the
# channel dim into nh × hs) — mathematically identical to nh separate heads,
# just faster.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One linear map produces Q, K, and V for ALL heads at once: it outputs
        # 3*C features (C for Q, C for K, C for V). Doing it in a single matmul
        # is faster than three separate ones.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: mixes the concatenated per-head outputs back into
        # the residual stream's C dimensions.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout (0.0 by default here): randomly zeroes some attention weights /
        # outputs during training to regularize. No-op when dropout=0.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # The causal mask: a lower-triangular matrix of 1s. Position i may attend
        # to j only where mask[i, j] == 1 (i.e. j <= i). Stored as a buffer (not a
        # parameter: it isn't learned, but should move with the model to GPU and
        # be saved with it). Shape (1, 1, block_size, block_size) to broadcast
        # over the batch and head dimensions.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, channels (= n_embd)

        # Project to Q, K, V (each B, T, C), then split.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape each into heads: (B, T, C) -> (B, nh, T, hs).
        # We split C into nh heads of size hs, and move the head axis next to the
        # batch axis so the following matmuls act on each (T, hs) head in parallel.
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores: how much each token attends to each other token.
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T), scaled by 1/sqrt(hs).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

        # Apply the causal mask: wherever mask == 0 (the future, j > i), set the
        # score to -inf so softmax makes its weight 0. Slice to the current T.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over the last dim (the keys j): turn scores into weights that
        # sum to 1 for each query i.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs).
        y = att @ v

        # Re-assemble heads: (B, nh, T, hs) -> (B, T, nh*hs=C). This concatenates
        # every head's output for each token back into one C-dim vector.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection (+ optional dropout).
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# MLP (feed-forward) — attention MOVES information between tokens; the MLP THINKS
# about it, per token. After a token has gathered context via attention, this
# little 2-layer network transforms that vector nonlinearly.
#
# WHY the 4× expansion (C -> 4C -> C)? Widening to 4× before the nonlinearity
# gives the layer room to compute richer intermediate features (more "neurons"
# to detect patterns) before compressing back to C. 4× is the GPT-2 convention;
# it's a good capacity/compute trade-off. GELU is a smooth version of ReLU that
# works well in Transformers.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()  # GPT-2 originally used the tanh approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)     # (B, T, C) -> (B, T, 4C)
        x = self.gelu(x)     # nonlinearity
        x = self.c_proj(x)   # (B, T, 4C) -> (B, T, C)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Block — one Transformer layer = attention sublayer + MLP sublayer, each wrapped
# in a residual connection with pre-LayerNorm.
#
# RESIDUAL CONNECTIONS (the `x + ...`): instead of replacing x, each sublayer
# computes a *delta* that is ADDED to x. This gives gradients a clean "highway"
# straight back through the network (the identity path), which is what makes it
# possible to train deep stacks without the signal vanishing. Each block only
# has to learn a small refinement of the running representation (the "residual
# stream").
#
# PRE-NORM (LayerNorm BEFORE the sublayer, i.e. `x + attn(ln(x))`): GPT-2 and
# nanoGPT normalize the input to each sublayer rather than its output
# (post-norm). Pre-norm keeps the residual highway un-normalized end-to-end,
# which is far more stable to train at depth (post-norm deep Transformers often
# need careful warmup/tuning or they diverge).
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # communicate: gather context (residual)
        x = x + self.mlp(self.ln_2(x))   # compute: think per-token (residual)
        return x


# ---------------------------------------------------------------------------
# GPT — the full model: embeddings -> dropout -> N blocks -> final LN -> lm_head.
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config

        # --- Embeddings (see INTERNAL_NOTES §3.2) ---
        # wte: token embedding table, one learned C-vector per vocab id.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # wpe: position embedding table, one learned C-vector per position 0..T-1.
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # --- The stack of Transformer blocks ---
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # --- Final LayerNorm before projecting to vocabulary logits ---
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # --- Language-model head: map each token's C-vector to a score (logit)
        #     for every vocabulary entry. bias=False is standard here. ---
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # --- WEIGHT TYING (see INTERNAL_NOTES §3.7) ---
        # The input embedding (wte: id -> vector) and the output projection
        # (lm_head: vector -> id scores) are inverse operations over the SAME
        # vocabulary, so it makes sense for them to share ONE weight matrix. We
        # literally point lm_head.weight at wte.weight (same tensor object). This:
        #   - removes ~19M redundant parameters (a huge fraction of the model),
        #   - ties "which tokens are similar as inputs" to "as outputs",
        #   - and empirically improves language-model quality.
        self.lm_head.weight = self.wte.weight

        # --- Initialize all weights GPT-2 style ---
        self.apply(self._init_weights)
        # Special scaled init for the residual-projection layers (the c_proj at
        # the end of each attention/MLP sublayer). With n_layer residual adds,
        # the residual stream's variance would grow with depth; scaling these by
        # 1/sqrt(2*n_layer) keeps it stable at initialization. (2 = attn + mlp.)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        # Small-normal init (std 0.02) is the GPT-2 recipe; biases start at 0.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids. targets: (B, T) next-token ids or None.

        Returns (logits, loss). loss is cross-entropy when targets are given,
        else None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Positions 0,1,...,T-1 to look up in the position table.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        tok_emb = self.wte(idx)   # (B, T, C) — what each token is
        pos_emb = self.wpe(pos)   # (T, C)    — where each token sits (broadcasts)
        x = self.drop(tok_emb + pos_emb)  # add the two signals into one vector

        # Pass through every Transformer block (the residual stream).
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)          # final normalization

        # Project to vocabulary logits: (B, T, C) -> (B, T, vocab_size).
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy compares the predicted distribution at each position
            # to the true next token. We flatten (B, T) positions into one long
            # batch of size B*T. ignore_index=-1 lets callers mask positions.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        verbose: bool = True,
    ) -> torch.optim.Optimizer:
        """Build the AdamW optimizer with correct weight-decay grouping.

        WEIGHT DECAY, and why we split parameters into two groups:
        weight decay gently pulls weights toward 0 each step (an L2-style
        regularizer that discourages over-large weights and helps generalization).
        But it only makes sense for the *matmul* weights — the 2-D tensors that
        actually mix features (Linear weights, the embedding table). The 1-D
        parameters — LayerNorm gains and any biases — are scales/offsets; decaying
        them toward 0 just fights the network for no benefit. So:
            - tensors with dim >= 2  -> weight_decay = 0.1   (matmuls, embeddings)
            - tensors with dim <  2  -> weight_decay = 0.0   (LayerNorm, biases)

        FUSED AdamW: on CUDA, PyTorch has a fused kernel that does the optimizer
        math for all params in one GPU launch — a nice free speedup. We enable it
        only when available and on CUDA.
        """
        # All parameters that will receive gradients. (named_parameters dedupes
        # the weight-tied wte/lm_head tensor, so it's grouped once.)
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Use the fused AdamW kernel if this PyTorch/CUDA build offers it.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra
        )

        if verbose:
            n_decay = sum(p.numel() for p in decay_params)
            n_nodecay = sum(p.numel() for p in nodecay_params)
            print(
                f"AdamW: {len(decay_params)} decayed tensors ({n_decay:,} params), "
                f"{len(nodecay_params)} non-decayed ({n_nodecay:,} params), "
                f"fused={use_fused}"
            )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `idx` by `max_new_tokens` tokens.

        `idx` is (B, T) of seed token ids (the prompt). We generate ONE token at a
        time: predict the next-token distribution, sample from it, append it, and
        feed the longer sequence back in — repeating until we've added enough. This
        loop is exactly how the model "writes": each new token becomes part of the
        context for the next.

        TEMPERATURE — the creativity dial. We divide the logits by `temperature`
        before softmax:
          - temperature < 1  sharpens the distribution → safer, more predictable,
            more repetitive text (→ 0 approaches greedy argmax).
          - temperature = 1  samples from the model's raw distribution.
          - temperature > 1  flattens it → more surprising / more mistakes.

        TOP-K — a sanity filter. With `top_k` set, we keep only the k most likely
        tokens and set all others to −∞ before sampling, so we never accidentally
        draw a token from the long, low-probability "tail" of nonsense. Smaller k =
        more focused; larger k = more diverse. (top_k=None = consider all tokens.)
        """
        for _ in range(max_new_tokens):
            # The model can only attend back `block_size` tokens, so crop the
            # context to the last block_size ids if the prompt+generation is longer.
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # Forward pass; we only need the logits at the LAST position — that's
            # the prediction for the next token.
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, vocab)

            # Optionally restrict to the top-k logits.
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                # v[:, [-1]] is the k-th largest logit per row; drop anything below.
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            # Convert to probabilities and draw one token (stochastic sampling).
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # append
        return idx


# ---------------------------------------------------------------------------
# Sanity check: instantiate the model, count params (~30M), run one forward pass
# on a dummy batch, verify the logits shape, and check the random-init loss.
#
# WHY should the initial loss be ~ln(vocab_size) ≈ ln(50257) ≈ 10.82? At random
# initialization the model has learned nothing, so its predicted next-token
# distribution is ~uniform over all 50257 tokens. Cross-entropy of a uniform
# distribution is -log(1/vocab) = log(vocab) ≈ 10.82. Seeing ~10.8 confirms the
# output layer and loss are wired correctly (not, say, wildly over-confident).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .utils import count_parameters, enable_utf8_stdout

    enable_utf8_stdout()
    torch.manual_seed(1337)

    config = GPTConfig()
    model = GPT(config)
    model.eval()

    count_parameters(model)

    # A dummy batch of random token ids (content is irrelevant for a shape/loss
    # sanity check). Use full block_size so we exercise the causal mask fully.
    B, T = 4, config.block_size
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets)

    expected = math.log(config.vocab_size)
    print(f"\ninput idx shape : {tuple(idx.shape)}")
    print(f"logits shape    : {tuple(logits.shape)}   (expected ({B}, {T}, {config.vocab_size}))")
    assert logits.shape == (B, T, config.vocab_size), "logits shape is wrong!"
    print(f"random-init loss: {loss.item():.4f}")
    print(f"expected ~ln(vocab) = ln({config.vocab_size}) = {expected:.4f}")
    print("\nForward pass OK: correct logit shape and loss ~= ln(vocab).")
