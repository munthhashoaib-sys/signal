"""
Embed text chunks with SentenceTransformer and persist a FAISS index + chunk list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = _BASE_DIR / "index.faiss"
CHUNKS_PATH = _BASE_DIR / "chunks.json"

_model: Optional[SentenceTransformer] = None


def _ensure_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string with the same model used for chunks (normalized)."""
    model = _ensure_model()
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec[0], dtype=np.float32)


def embed_and_store(chunks: List[str]) -> None:
    """
    Embed chunks, build a FAISS inner-product index (cosine similarity on L2-normalized vectors),
    and save index + chunk list to disk.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")

    model = _ensure_model()
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

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
