"""
Chunk long filing text into overlapping word windows for RAG.
"""

from __future__ import annotations

import re
from typing import List

# Target chunk size and overlap (in words)
CHUNK_WORDS = 500
OVERLAP_WORDS = 100


def clean_text(text: str) -> str:
    """
    Normalize whitespace and drop noisy lines (page numbers, table junk, short headers).
    """
    if not text or not text.strip():
        return ""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if len(line) < 20:
            continue
        if len(line) == 1:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        kept.append(line)

    # Collapse to single spaces between lines (paragraph-ish); trim overall whitespace
    collapsed = re.sub(r"[ \t]+", " ", "\n".join(kept))
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Split text into overlapping chunks of ~chunk_words words, overlapping by overlap_words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words.")

    step = chunk_words - overlap_words
    chunks: List[str] = []
    i = 0
    n = len(words)

    while i < n:
        piece = words[i : i + chunk_words]
        chunks.append(" ".join(piece))
        if len(piece) < chunk_words:
            break
        i += step

    return chunks


if __name__ == "__main__":
    sample = "This is a test sentence. " * 200
    chunks = chunk_text(sample)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0])} characters")
    print(f"Last chunk preview: {chunks[-1][:100]}")
