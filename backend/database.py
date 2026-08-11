import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase = create_client(url, key) if url and key else None

def save_session(company_name, prospect_name, prospect_role, linkedin_url, ae_product, ae_company, ae_role):
    if not supabase:
        print("Supabase not configured. Skipping save.")
        return None, None
    try:
        company_result = supabase.table("companies").upsert(
            {"name": company_name},
            on_conflict="name"
        ).execute()
        company_id = company_result.data[0]["id"]
        session_result = supabase.table("sessions").insert({
            "company_id": company_id,
            "prospect_name": prospect_name,
            "prospect_role": prospect_role,
            "linkedin_url": linkedin_url,
            "ae_product": ae_product,
            "ae_company": ae_company,
            "ae_role": ae_role
        }).execute()
        session_id = session_result.data[0]["id"]
        print(f"Session saved: {session_id}")
        return company_id, session_id
    except Exception as e:
        print(f"Database save failed: {e}")
        return None, None

def save_chunks(session_id, approved_chunks):
    if not supabase or not session_id:
        return
    try:
        rows = []
        for chunk in approved_chunks:
            rows.append({
                "session_id": session_id,
                "text": chunk.get("text", "")[:500],
                "score": chunk.get("score", 0),
                "passed_judge": True,
                "section": "10-K"
            })
        if rows:
            supabase.table("chunks").insert(rows).execute()
            print(f"Saved {len(rows)} chunks to database.")
    except Exception as e:
        print(f"Chunk save failed: {e}")

def save_openers(session_id, openers):
    if not supabase or not session_id:
        return
    try:
        rows = []
        for opener in openers:
            rows.append({
                "session_id": session_id,
                "text": opener.get("text", ""),
                "signal_tag": opener.get("signal", ""),
                "rationale": opener.get("why", ""),
                "score": opener.get("score", 0),
                "verdict": opener.get("verdict", "APPROVE")
            })
        if rows:
            supabase.table("openers").insert(rows).execute()
            print(f"Saved {len(rows)} openers to database.")
    except Exception as e:
        print(f"Opener save failed: {e}")
