# Signal

AI-powered cold email opener generator for B2B sales reps. Signal reads a company's SEC 10-K annual filing and generates five hyper-specific cold email openers grounded in real filing disclosures — replacing 30 minutes of manual research with a 60-second automated pipeline.

**Live product:** https://signal-kappa-two.vercel.app  
**Backend API:** https://signal-project.onrender.com/health

---

## The Problem

Sales reps are expected to personalize outreach at scale but have no practical way to do it. Generic AI tools produce openers prospects delete immediately. Manual research takes 30 minutes per prospect and does not scale. Signal closes this gap by treating the 10-K as a structured data source and extracting the specific risks, strategic bets, and financial tensions a prospect is actually accountable for.

---

## Architecture

Signal is a production-grade RAG pipeline with three evaluation layers.
