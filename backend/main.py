import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.edgar import fetch_10k
from backend.chunker import chunk_text
from backend.embedder import embed_and_store
from backend.retriever import retrieve, build_rag_context
from backend.generator import generate_openers
from backend.judge import judge_chunks, needs_fallback
from backend.opener_judge import judge_openers
from backend.reviser import revise_openers

app = FastAPI()

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
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        print(f"\nStarting pipeline for {req.company_name}...")

        # Step 1: Fetch and chunk the 10-K
        print("Step 1: Fetching 10-K from EDGAR...")
        text = fetch_10k(req.company_name)
        if not text or len(text) < 500:
            raise HTTPException(status_code=400, detail="Could not extract text from filing.")

        print("Step 2: Chunking and embedding...")
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not chunk document.")
        embed_and_store(chunks)

        # Step 3: Retrieve top chunks
        print("Step 3: Retrieving top chunks...")
        query = f"strategic risks, supply chain challenges, financial pressures, and growth initiatives at {req.company_name}"
        retrieved_chunks, scores = retrieve(query, top_k=10)

        # Step 4: Judge chunk quality
        print("Step 4: Judging chunk quality...")
        approved_chunks, rejected_chunks, chunk_evals = judge_chunks(
            chunks=retrieved_chunks,
            prospect_name=req.prospect_name,
            prospect_role=req.prospect_role,
            prospect_company=req.company_name,
            ae_product=req.ae_product
        )

        # Step 5: Agentic fallback if not enough chunks passed
        if needs_fallback(approved_chunks):
            print("Step 5: Not enough high-quality chunks. Widening retrieval...")
            retrieved_chunks, scores = retrieve(query, top_k=20)
            approved_chunks, rejected_chunks, chunk_evals = judge_chunks(
                chunks=retrieved_chunks,
                prospect_name=req.prospect_name,
                prospect_role=req.prospect_role,
                prospect_company=req.company_name,
                ae_product=req.ae_product
            )
            if needs_fallback(approved_chunks):
                print("Step 5: Still insufficient. Proceeding with best available chunks.")
                approved_chunks = [{"text": c, "score": 5, "key_signal": None} for c in retrieved_chunks[:4]]

        # Step 6: Build context and generate openers
        print("Step 6: Generating openers with Claude...")
        context_chunks = [c["text"] if isinstance(c, dict) else c for c in approved_chunks[:6]]
        rag_context = build_rag_context(context_chunks)

        result = generate_openers(
            prospect_name=req.prospect_name,
            company_name=req.company_name,
            prospect_role=req.prospect_role,
            ae_product=req.ae_product,
            rag_context=rag_context
        )

        openers = result.get("openers", [])

        # Step 7: Judge opener quality
        print("Step 7: Judging opener quality...")
        approved_openers, flagged_openers, rejected_openers, opener_evals = judge_openers(
            openers=openers,
            prospect_name=req.prospect_name,
            prospect_role=req.prospect_role,
            prospect_company=req.company_name,
            ae_product=req.ae_product
        )

        # Step 8: Revise rejected openers
        final_openers = []
        final_openers.extend(approved_openers)
        final_openers.extend(flagged_openers)

        if rejected_openers:
            print(f"Step 8: Revising {len(rejected_openers)} rejected openers...")
            revised = revise_openers(
                rejected_openers=rejected_openers,
                approved_chunks=approved_chunks,
                prospect_name=req.prospect_name,
                prospect_role=req.prospect_role,
                prospect_company=req.company_name,
                ae_product=req.ae_product
            )
            final_openers.extend(revised)

        # Sort by score descending
        final_openers.sort(key=lambda x: x.get("score", 0), reverse=True)

        print(f"Pipeline complete. Returning {len(final_openers)} openers.")

        return {
            "openers": final_openers,
            "pipeline_summary": {
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_approved": len(approved_chunks),
                "openers_approved": len(approved_openers),
                "openers_flagged": len(flagged_openers),
                "openers_revised": len(rejected_openers)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
