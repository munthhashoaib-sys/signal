"""
Fetch the most recent 10-K filing text from SEC EDGAR for a given company name.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import pdfplumber
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

USER_AGENT = "Mozilla/5.0 Signal-App contact@signal.com"

SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q={query}&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K"
)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _pad_cik(cik: str) -> str:
    """CIK as 10-digit zero-padded string (e.g. 320193 -> 0000320193)."""
    digits = re.sub(r"\D", "", cik)
    return digits.zfill(10)


def _cik_int_path(cik_padded: str) -> str:
    """CIK for archive path: no leading zeros."""
    return str(int(cik_padded))


def _accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def search_cik_candidates(company_name: str) -> List[Tuple[str, float]]:
    """
    Query EDGAR full-text search and return (CIK, score) pairs, best first.
    Deduplicates by CIK, keeping the highest score per CIK.
    """
    query = quote(company_name.strip())
    url = SEARCH_URL.format(query=query)
    print(f"Searching EDGAR for {company_name!r}...")
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    hits = data.get("hits", {}).get("hits", [])
    best: Dict[str, float] = {}
    for h in hits:
        src = h.get("_source") or {}
        ciks = src.get("ciks") or []
        if not ciks:
            continue
        cik = str(ciks[0]).strip()
        score = float(h.get("_score") or 0.0)
        if cik not in best or score > best[cik]:
            best[cik] = score
    ranked = sorted(best.items(), key=lambda x: -x[1])
    return ranked


def _fetch_submissions(cik_padded: str) -> Dict[str, Any]:
    url = SUBMISSIONS_URL.format(cik_padded=cik_padded)
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def _pick_cik_for_company(company_name: str, candidates: List[Tuple[str, float]]) -> str:
    """
    Choose the CIK whose SEC legal name best matches the user query.
    When several names contain the query (e.g. 'Apple'), prefer the shortest legal name.
    """
    q = company_name.strip().lower()
    if not q:
        raise ValueError("Company name must not be empty.")

    scored: List[Tuple[int, str, str]] = []
    for cik, _ in candidates:
        cik_padded = _pad_cik(cik)
        try:
            sub = _fetch_submissions(cik_padded)
        except requests.HTTPError:
            continue
        name = (sub.get("name") or "").strip()
        if not name:
            continue
        nl = name.lower()
        if q == nl:
            rank = 0
        elif nl.startswith(q + " ") or nl.startswith(q + ",") or nl.startswith(q + "."):
            rank = 1
        elif q in nl:
            rank = 2
        else:
            continue
        scored.append((rank, len(name), cik_padded))

    if not scored:
        raise ValueError(
            f"No SEC registrant matched {company_name!r} among search results. "
            "Try a more specific company name."
        )
    scored.sort(key=lambda x: (x[0], x[1]))
    chosen = scored[0][2]
    sub = _fetch_submissions(chosen)
    print(f"Found CIK: {chosen} ({sub.get('name', '')})")
    return chosen


def resolve_cik(company_name: str) -> str:
    candidates = search_cik_candidates(company_name)
    if not candidates:
        raise ValueError(f"No EDGAR hits for {company_name!r}.")
    return _pick_cik_for_company(company_name, candidates)


def find_most_recent_10k(submissions: Dict[str, Any]) -> Tuple[str, str]:
    """Return (accession_number, primary_document) for the latest 10-K."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms: List[str] = recent.get("form") or []
    accession: List[str] = recent.get("accessionNumber") or []
    primary: List[str] = recent.get("primaryDocument") or []
    for i, form in enumerate(forms):
        if form == "10-K" and i < len(accession) and i < len(primary):
            return accession[i], primary[i]
    raise ValueError("No 10-K filing found in recent submissions.")


def build_filing_url(cik_padded: str, accession: str, primary_document: str) -> str:
    cik_path = _cik_int_path(cik_padded)
    acc = _accession_no_dashes(accession)
    return f"{ARCHIVE_BASE}/{cik_path}/{acc}/{primary_document}"


def _extract_text_from_html(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_text_from_pdf(data: bytes) -> str:
    out: List[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out.append(t)
    return "\n\n".join(out)


def download_filing_text(url: str) -> str:
    print(f"Downloading filing: {url}")
    r = requests.get(url, headers=_headers(), timeout=120)
    r.raise_for_status()
    content_type = (r.headers.get("Content-Type") or "").lower()
    body = r.content
    path_lower = url.lower()

    if "pdf" in content_type or path_lower.endswith(".pdf"):
        print("Extracting text from PDF...")
        return _extract_text_from_pdf(body)

    if "html" in content_type or path_lower.endswith((".htm", ".html")):
        print("Extracting text from HTML...")
        return _extract_text_from_html(body)

    # Fallback: try HTML parse, then PDF
    try:
        text = _extract_text_from_html(body)
        if len(text.strip()) > 200:
            return text
    except Exception:
        pass
    try:
        return _extract_text_from_pdf(body)
    except Exception as e:
        raise ValueError(f"Could not parse filing as HTML or PDF: {e}") from e


def fetch_10k(company_name: str) -> str:
    """
    Resolve company name to a CIK, locate the most recent 10-K, download it,
    and return plain text.
    """
    cik_padded = resolve_cik(company_name)
    print("Fetching most recent 10-K...")
    submissions = _fetch_submissions(cik_padded)
    accession, primary = find_most_recent_10k(submissions)
    url = build_filing_url(cik_padded, accession, primary)
    text = download_filing_text(url)
    print(f"Done. Extracted {len(text)} characters.")
    return text


if __name__ == "__main__":
    text = fetch_10k("Apple")
    print(f"Total characters extracted: {len(text)}")
    print(text[:500])
