"""Language-aware tokenization helpers for technical retrieval.

The project needs the same tokenization rules on both sides of BM25.  In
particular, software identifiers such as ``gpt-4``, ``python3.11`` and
``deep_learning`` must remain intact while Chinese spans still benefit from
jieba segmentation.
"""

from __future__ import annotations

import re

import jieba

_TECHNICAL_OR_CJK = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:(?:[._/-][A-Za-z0-9]+)|(?:\+\+)|#)*"
    r"|\d+(?:\.\d+)+"
    r"|\d+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_CJK_ONLY = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def tokenize_technical_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/English technical text deterministically.

    Punctuation is discarded, CJK spans are segmented with jieba, and common
    identifier punctuation (``-``, ``_``, ``.``, ``/``, ``++``, ``#``) is
    preserved inside Latin/numeric tokens.
    """

    tokens: list[str] = []
    for match in _TECHNICAL_OR_CJK.finditer(text):
        value = match.group(0)
        if _CJK_ONLY.fullmatch(value):
            tokens.extend(token.strip() for token in jieba.lcut(value) if token.strip())
        else:
            tokens.append(value)
    return tokens
