"""Byte-level BPE tokenizer: training + encoding/decoding.

Mirrors the GPT-2 pre-tokenization scheme so that, given the GPT-2 vocab/merges,
`Tokenizer.encode` reproduces `tiktoken`'s output exactly.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Iterable, Iterator

import regex as re

# GPT-2 pre-tokenization pattern (contractions / words / numbers / punctuation / whitespace).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)


def _split_on_special(text: str, special_tokens: list[str]) -> list[str]:
    """Split `text` on the special tokens, dropping them.

    Splitting first prevents BPE merges from crossing document boundaries.
    """
    if not special_tokens:
        return [text]
    # Longest-first so overlapping specials (e.g. "<|eot|><|eot|>") match greedily.
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in ordered)
    return re.split(pattern, text)


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[str]:
    """Count pre-token frequencies over `text`, ignoring special tokens."""
    counts: Counter[str] = Counter()
    for segment in _split_on_special(text, special_tokens):
        for match in _COMPILED_PAT.finditer(segment):
            counts[match.group()] += 1
    return counts


def bpe_train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns the id->bytes vocabulary and the ordered list of merges. Ties in
    pair frequency are broken by preferring the lexicographically greatest pair.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Vocabulary starts with the 256 single bytes, then the special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Each unique pre-token becomes a mutable list of byte tokens with a weight.
    word_counts = _count_pretokens(text, special_tokens)
    words: list[list[bytes]] = []
    weights: list[int] = []
    for word, count in word_counts.items():
        words.append([bytes([b]) for b in word.encode("utf-8")])
        weights.append(count)

    # Frequency of each adjacent pair, plus a reverse index of the words holding it.
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for idx, tokens in enumerate(words):
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += weights[idx]
            pair_to_words[pair].add(idx)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        merged = best[0] + best[1]
        vocab[len(vocab)] = merged

        # Apply the merge only in the words that actually contain it, updating
        # pair counts incrementally rather than rescanning the whole corpus.
        for idx in list(pair_to_words[best]):
            tokens = words[idx]
            weight = weights[idx]
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= weight
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(idx)

            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            words[idx] = new_tokens

            for pair in zip(new_tokens, new_tokens[1:]):
                pair_counts[pair] += weight
                pair_to_words[pair].add(idx)

        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)

    return vocab, merges


class Tokenizer:
    """A byte-level BPE tokenizer parameterized by a vocab, merges, and specials."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self._bytes_to_id: dict[bytes, int] = {token: idx for idx, token in vocab.items()}
        # Merge priority: earlier merges bind first (lower rank wins).
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        # Precompile the special-token splitter (longest-first, capturing so we keep them).
        self._special_pattern = None
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(idx): token.encode("utf-8") for token, idx in raw_vocab.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def _bpe(self, piece: str) -> list[int]:
        """BPE-encode a single pre-token string into token ids."""
        tokens = [bytes([b]) for b in piece.encode("utf-8")]
        while len(tokens) >= 2:
            # Pick the adjacent pair learned earliest; stop when none are known.
            best = min(
                zip(tokens, tokens[1:]),
                key=lambda p: self._merge_ranks.get(p, float("inf")),
            )
            if best not in self._merge_ranks:
                break
            merged = best[0] + best[1]
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self._bytes_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_set = set(self.special_tokens)
        chunks = self._special_pattern.split(text) if self._special_pattern else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                ids.append(self._bytes_to_id[chunk.encode("utf-8")])
            else:
                for match in _COMPILED_PAT.finditer(chunk):
                    ids.extend(self._bpe(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. file lines) for large inputs."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
