"""
Retrieve top-k similar chunks from the FAISS index for a query.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.embedder import embed_query, load_index

TOP_K = 6


def retrieve(query: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    """
    Embed the query, search the FAISS index for the top_k most similar chunks,
    and return those chunk texts with their similarity scores (inner product / cosine).
    """
    print("Searching index for relevant sections...")
    index, all_chunks = load_index()

    n = index.ntotal
    if n == 0:
        print("Retrieved 0 chunks with average similarity score of 0.000")
        return [], []

    k = min(top_k, n)
    q = np.asarray(embed_query(query), dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(q, k)

    scores: List[float] = []
    chunks: List[str] = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        chunks.append(all_chunks[idx])
        scores.append(float(dist))

    avg = float(np.mean(scores)) if scores else 0.0
    print(
        f"Retrieved {len(chunks)} chunks with average similarity score of {avg:.3f}"
    )

    return chunks, scores


def build_rag_context(retrieved_chunks: List[str]) -> str:
    """Join retrieved chunks with numbered excerpt headers for prompt injection."""
    if not retrieved_chunks:
        return ""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(f"--- Excerpt {i} ---\n{chunk}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    query = "What are the biggest supply chain risks this company faces?"
    chunks, scores = retrieve(query)
    print(f"\nTop {len(chunks)} relevant sections:")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"\n[{i+1}] Score: {score:.3f}")
        print(chunk[:200])

    context = build_rag_context(chunks)
    print("\nFormatted context:")
    print(context[:500])
