"""
Embed text chunks via OpenAI and persist a FAISS index + chunk list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
EMBED_BATCH = 128

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

INDEX_PATH = _BASE_DIR / "index.faiss"
CHUNKS_PATH = _BASE_DIR / "chunks.json"

_client: Optional[OpenAI] = None


def _ensure_client() -> OpenAI:
    global _client
    if _client is None:
        print("Loading embedding model...")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _client = OpenAI(api_key=key)
    return _client


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32)


def _embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Return L2-normalized embedding matrix (float32) for cosine / inner product search."""
    rows: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=MODEL_NAME, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        for d in ordered:
            rows.append(d.embedding)
    arr = np.asarray(rows, dtype=np.float32)
    return _l2_normalize_rows(arr)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string with the same model used for chunks (normalized)."""
    client = _ensure_client()
    resp = client.embeddings.create(model=MODEL_NAME, input=text)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return vec
    return (vec / n).astype(np.float32)


def embed_and_store(chunks: List[str]) -> None:
    """
    Embed chunks, build a FAISS inner-product index (cosine similarity on L2-normalized vectors),
    and save index + chunk list to disk.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")

    client = _ensure_client()
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = _embed_texts(client, chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving index to disk...")
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Done. Index contains {index.ntotal} vectors.")


def load_index() -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index and chunk list from disk."""
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: List[str] = json.load(f)
    return index, chunks


if __name__ == "__main__":
    sample_chunks = [
        "Rivian faces significant supply chain risk due to sole-source battery suppliers.",
        "The company is scaling its Normal Illinois plant to support Commercial Van production.",
        "Gross profit per vehicle remains negative as the company works toward unit economics.",
        "Management expects material cost reductions to drive margin improvement in fiscal 2025.",
        "Regulatory changes to EV tax credits represent a material risk to consumer demand.",
    ]
    embed_and_store(sample_chunks)
    index, chunks = load_index()
    print(f"Index loaded with {index.ntotal} vectors")
    print(f"First chunk: {chunks[0]}")
