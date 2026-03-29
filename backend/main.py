"""
FastAPI entrypoint for the Signal 10-K RAG pipeline.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.chunker import chunk_text
from backend.edgar import fetch_10k
from backend.embedder import embed_and_store
from backend.generator import generate_openers
from backend.retriever import build_rag_context, retrieve

app = FastAPI(title="Signal", description="10-K RAG cold email openers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    company_name: str
    prospect_name: str
    prospect_role: str
    ae_product: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    company = req.company_name.strip()
    if not company:
        raise HTTPException(status_code=400, detail="company_name is required.")

    try:
        print(f"Starting pipeline for {company}...")

        print("Fetching 10-K text from EDGAR...")
        filing_text = fetch_10k(company)

        print("Chunking filing text...")
        chunks = chunk_text(filing_text)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks produced from the filing. The document may be empty after cleaning.",
            )
        print(f"Created {len(chunks)} chunks.")

        print("Embedding chunks and updating FAISS index...")
        embed_and_store(chunks)

        query = (
            f"strategic risks, supply chain challenges, financial pressures, "
            f"and growth initiatives at {company}"
        )
        print(f"Retrieving context with query: {query!r}")

        retrieved_chunks, _scores = retrieve(query)
        if not retrieved_chunks:
            raise HTTPException(
                status_code=500,
                detail="Retrieval returned no chunks after indexing.",
            )

        rag_context = build_rag_context(retrieved_chunks)
        print("Built RAG context for generator.")

        print("Generating openers with Claude...")
        result = generate_openers(
            prospect_name=req.prospect_name,
            company_name=company,
            prospect_role=req.prospect_role,
            ae_product=req.ae_product,
            rag_context=rag_context,
        )

        print("Pipeline finished successfully.")
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {e!s}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
