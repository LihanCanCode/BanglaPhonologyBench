# -*- coding: utf-8 -*-
"""Real-tokenizer adapter: HF tokenizer -> UTF-8 byte spans per token (Spec C.1).

Three tokenizer families are supported, each decoded back to the *raw bytes*
its tokens cover so that GTAD's byte-internal violations are measurable:

  1. byte-level BPE (GPT-2, Llama-3): token strings live in the GPT-2
     byte<->unicode alphabet (Ġ etc.); invert that map to recover bytes.
  2. SentencePiece (T5/banglat5, Gemma/TigerLLM): pieces use ▁ for space and
     <0xNN> byte-fallback pieces for out-of-vocabulary bytes.
  3. ByT5: ids 3..258 are literal bytes (id - 3); everything else is special.

The invariant is that token byte strings concatenate EXACTLY to the word's
UTF-8 encoding (after stripping BOS/EOS/specials and any leading-space /
metaspace artifact). Words where no decoder satisfies it raise
MisalignedTokenizationError and are quarantined by callers, never guessed at.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

__all__ = [
    "MisalignedTokenizationError",
    "real_token_byte_spans",
    "token_bytes_list",
    "load_tokenizers",
    "TOKENIZER_SPECS",
]


class MisalignedTokenizationError(ValueError):
    """No decoding of the token ids reproduces the word's UTF-8 bytes."""


# --- GPT-2 byte-level alphabet ------------------------------------------------

@lru_cache(maxsize=1)
def _gpt2_byte_decoder() -> Dict[str, int]:
    """Inverse of the standard GPT-2 bytes_to_unicode map (deterministic)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# --- per-family token -> bytes decoders ---------------------------------------

def _bytes_bytelevel(tokens: List[str]) -> Optional[List[bytes]]:
    dec = _gpt2_byte_decoder()
    out = []
    for t in tokens:
        if not all(ch in dec for ch in t):
            return None
        out.append(bytes(dec[ch] for ch in t))
    return out


def _bytes_sentencepiece(tokens: List[str]) -> Optional[List[bytes]]:
    out = []
    for t in tokens:
        if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            try:
                out.append(bytes([int(t[3:5], 16)]))
                continue
            except ValueError:
                return None
        out.append(t.replace("▁", " ").encode("utf-8"))
    return out


def _bytes_byt5(ids: List[int], offset: int = 3) -> List[bytes]:
    return [bytes([i - offset]) for i in ids]


def _is_byt5(tokenizer) -> bool:
    return "ByT5" in type(tokenizer).__name__


def token_bytes_list(word: str, tokenizer) -> Tuple[List[int], List[bytes]]:
    """Encode `word` (no specials) and return (ids, per-token raw bytes).

    Tries the family decoders in order and keeps the first whose concatenation
    matches the word's UTF-8 bytes, tolerating one leading space/metaspace
    artifact on the first token. Raises MisalignedTokenizationError otherwise.
    """
    target = word.encode("utf-8")
    ids = tokenizer.encode(word, add_special_tokens=False)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = [i for i in ids if i not in special]

    candidates: List[List[bytes]] = []
    if _is_byt5(tokenizer):
        candidates.append(_bytes_byt5(ids))
    else:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for fn in (_bytes_bytelevel, _bytes_sentencepiece):
            got = fn(tokens)
            if got is not None:
                candidates.append(got)

    for tb in candidates:
        cat = b"".join(tb)
        if cat == target:
            return ids, tb
        # leading space / metaspace artifact: first token starts with ' '
        if cat == b" " + target and tb and tb[0][:1] == b" ":
            tb = [tb[0][1:]] + tb[1:]
            if tb[0] == b"":                      # bare space token: drop it
                return ids[1:], tb[1:]
            return ids, tb
    raise MisalignedTokenizationError(
        f"cannot reproduce bytes of {word!r} with {type(tokenizer).__name__}")


def real_token_byte_spans(word: str, tokenizer) -> List[Tuple[int, int]]:
    """UTF-8 byte [start, end) spans of `word`'s tokens under `tokenizer`.

    Specials are stripped; spans partition [0, len(word.encode('utf-8'))).
    Raises MisalignedTokenizationError when the invariant cannot be met —
    callers should quarantine such words, not crash.
    """
    _, tb = token_bytes_list(word, tokenizer)
    spans, pos = [], 0
    for b in tb:
        if len(b) == 0:
            continue                               # zero-width artifact token
        spans.append((pos, pos + len(b)))
        pos += len(b)
    assert pos == len(word.encode("utf-8")), "span partition invariant"
    return spans


# --- tokenizer registry --------------------------------------------------------

# name -> list of HF repo candidates, tried in order (first that loads wins).
TOKENIZER_SPECS: Dict[str, List[str]] = {
    "llama3":   ["meta-llama/Llama-3.1-8B-Instruct",       # gated: needs HF_TOKEN
                 "NousResearch/Meta-Llama-3.1-8B-Instruct"],  # ungated mirror, same tokenizer
    "gpt2":     ["gpt2"],                                   # English-centric control
    "byt5":     ["google/byt5-small"],                      # byte-level control
    "tigerllm": ["md-nishat-008/TigerLLM-9B-it"],           # Bangla-centric (Gemma-2 SP)
    "banglat5": ["csebuetnlp/banglat5"],                    # Bangla-centric (SP)
}


def load_tokenizers(names: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, object]:
    """Load requested tokenizers; skip (with a message) any that fail."""
    from transformers import AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    out: Dict[str, object] = {}
    for name in names or list(TOKENIZER_SPECS):
        last_err = None
        for repo in TOKENIZER_SPECS[name]:
            try:
                out[name] = AutoTokenizer.from_pretrained(repo, token=token)
                if verbose:
                    print(f"[load_tokenizers] {name}: {repo}")
                break
            except Exception as e:                 # gated / missing / network
                last_err = e
        else:
            if verbose:
                print(f"[load_tokenizers] SKIP {name}: {last_err}")
    return out
