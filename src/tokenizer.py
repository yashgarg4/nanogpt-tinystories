"""
tokenizer.py — a thin wrapper around tiktoken's GPT-2 BPE tokenizer.

WHY do we even need a tokenizer?
--------------------------------
A neural network only does math on numbers. Text is a string of characters, so
before the model can read it we must turn text into a sequence of integers
("token ids"), and after the model writes, we turn ids back into text. That
two-way street is the tokenizer's whole job:

    "Once upon a time"  --encode-->  [7454, 2402, 257, 640]
    [7454, 2402, 257, 640]  --decode-->  "Once upon a time"

WHY BPE (subword) instead of characters or whole words?
-------------------------------------------------------
There are three natural ways to split text into tokens:

  1) Characters ('O','n','c','e', ...).
       + Tiny vocabulary (~100 symbols), never an "unknown" token.
       - Sequences become VERY long (one step per letter), so the model wastes
         capacity/compute learning to spell before it can learn meaning.

  2) Whole words ('Once','upon','a','time').
       + Short sequences.
       - Vocabulary explodes (millions of words), and any word not seen during
         training (typos, names, new words) becomes an "unknown" — information
         lost.

  3) Subwords via BPE = Byte-Pair Encoding (what GPT-2 uses, and what we use).
       Idea: start from raw bytes, then repeatedly merge the most frequent
       adjacent pair into a new token. Common words end up as a single token
       ("the", " time"); rare words split into a few reusable pieces
       ("tokenization" -> "token" + "ization"). You get the best of both:
         + Short-ish sequences (common stuff is one token).
         + A fixed, modest vocabulary (50257 for GPT-2).
         + NO unknown tokens ever — worst case a word falls back to bytes.

Tokenization is NOT the thing we're trying to learn in this project (the MODEL
is), so we happily reuse GPT-2's battle-tested BPE via `tiktoken` instead of
training our own. `tiktoken` is just a fast tokenizer; it contains no neural
network and no "GPT" — it only maps text <-> ids.

The special token <|endoftext|> (id 50256) marks a document boundary. We append
it after each story so the model can learn where one story ends and the next
begins (and later, at generation time, learn to *stop*).
"""

import tiktoken


class Tokenizer:
    """GPT-2 BPE tokenizer. encode(str) -> list[int]; decode(list[int]) -> str."""

    def __init__(self, encoding_name: str = "gpt2"):
        # `tiktoken.get_encoding("gpt2")` loads GPT-2's merge rules + vocabulary.
        # The first call downloads a small vocab file and caches it locally.
        self.enc = tiktoken.get_encoding(encoding_name)

        # The end-of-text / document-separator token id (50256 for GPT-2).
        # We expose it so data.py can append it between stories.
        self.eot_token: int = self.enc.eot_token

        # Number of distinct tokens. For GPT-2 this is 50257 and MUST match
        # GPTConfig.vocab_size (the model's output layer has one slot per token).
        self.vocab_size: int = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids.

        We use `encode_ordinary`, which treats the input as *plain text* and does
        NOT try to interpret substrings like "<|endoftext|>" as special tokens.
        That's what we want for raw dataset text: everything becomes ordinary
        tokens, and we add the EOT separator ourselves, deliberately.
        """
        return self.enc.encode_ordinary(text)

    def encode_with_eot(self, text: str) -> list[int]:
        """Same as encode() but appends the end-of-text separator token.
        Handy for building the training stream where stories are concatenated.
        """
        ids = self.enc.encode_ordinary(text)
        ids.append(self.eot_token)
        return ids

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text. Inverse of encode()."""
        return self.enc.decode(ids)


# A module-level singleton so callers (and multiprocessing workers) can just do
# `from .tokenizer import get_tokenizer`. Cheap to build, but no need to rebuild.
_TOKENIZER: Tokenizer | None = None


def get_tokenizer() -> Tokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer()
    return _TOKENIZER


if __name__ == "__main__":
    import sys

    # Windows consoles default to a legacy code page (e.g. cp1252) that can't
    # print many Unicode characters and will crash. Force UTF-8 so decoded text
    # (and any check marks) print safely everywhere. No-op if unsupported.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Quick self-test: prove the round-trip encode -> decode is lossless.
    tok = get_tokenizer()
    sample = "Once upon a time, there was a little robot who loved to read."
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"eot_token  = {tok.eot_token}")
    print(f"text       = {sample!r}")
    print(f"ids        = {ids}")
    print(f"decoded    = {back!r}")
    assert back == sample, "round-trip failed!"
    print("round-trip OK ✓")
