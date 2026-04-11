"""Sparse vector helpers for lightweight lexical retrieval."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_HASH_SPACE = 2**31 - 1


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def sparse_terms(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return [], []
    counts: dict[int, float] = {}
    length = max(len(tokens), 1)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % _HASH_SPACE
        counts[index] = counts.get(index, 0.0) + 1.0

    indices = sorted(counts.keys())
    values = [round((counts[index] / length) * (1.0 + math.log1p(counts[index])), 6) for index in indices]
    return indices, values
