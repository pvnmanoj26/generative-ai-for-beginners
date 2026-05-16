import os
import json
import re
import uuid
import asyncio
import httpx
import numpy as np
import faiss
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import anthropic
from dotenv import load_dotenv
from upstash_vector import Index
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime, timezone

load_dotenv()

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Clinical AI API",
    description="Healthcare AI — summarization, care gaps, semantic search",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

claude     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
BQ_PROJECT = "healthcare-ai-manoj"
BQ_TABLE   = f"{BQ_PROJECT}.healthcare_ai.patients"
bq_client  = bigquery.Client(project=BQ_PROJECT)

vertexai.init(project=BQ_PROJECT, location="us-central1")
embedder = TextEmbeddingModel.from_pretrained("text-embedding-004")

notes_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

# ─────────────────────────────────────────────
# GUIDELINES
# ─────────────────────────────────────────────
GUIDELINES = [
    {"condition": "diabetes",     "text": "Diabetes: HbA1c every 3 months if uncontrolled (>8%), every 6 months if stable."},
    {"condition": "diabetes",     "text": "Diabetes: Annual diabetic eye exam required."},
    {"condition": "diabetes",     "text": "Diabetes: Annual diabetic foot exam including monofilament test."},
    {"condition": "diabetes",     "text": "Diabetes: Urine microalbumin/creatinine ratio checked annually."},
    {"condition": "diabetes",     "text": "Diabetes: BP target <130/80 mmHg."},
    {"condition": "diabetes",     "text": "Diabetes: Statin therapy recommended aged 40-75."},
    {"condition": "diabetes",     "text": "Diabetes: ACE inhibitor or ARB if microalbuminuria present."},
    {"condition": "heart_failure","text": "Heart Failure: ACE inhibitor or ARB for all HFrEF (EF <40%)."},
    {"condition": "heart_failure","text": "Heart Failure: Beta-blocker for all stable HFrEF."},
    {"condition": "heart_failure","text": "Heart Failure: Echocardiogram at diagnosis."},
    {"condition": "heart_failure","text": "Heart Failure: 30-day follow-up post-discharge."},
    {"condition": "hypertension", "text": "Hypertension: BP target <130/80 mmHg."},
    {"condition": "hypertension", "text": "Hypertension: Annual renal function monitoring."},
    {"condition": "hypertension", "text": "Hypertension: Lifestyle modifications including DASH diet."},
    {"condition": "ckd",          "text": "CKD: eGFR and urine albumin-creatinine ratio every 3-6 months."},
    {"condition": "ckd",          "text": "CKD: Nephrology referral when eGFR <30."},
    {"condition": "ckd",          "text": "CKD: Avoid NSAIDs and nephrotoxic medications."},
    {"condition": "asthma",       "text": "Asthma: Annual spirometry recommended."},
    {"condition": "asthma",       "text": "Asthma: Inhaled corticosteroid first-line for persistent asthma."},
    {"condition": "dyslipidaemia","text": "Dyslipidaemia: Fasting lipid panel annually."},
    {"condition": "dyslipidaemia","text": "Dyslipidaemia: LDL target <70 mg/dL for very high CV risk."},
    {"condition": "general",      "text": "General: Smoking cessation counseling for all active smokers."},
    {"condition": "general",      "text": "General: Annual flu vaccination for chronic disease patients."},
]

GUIDELINE_TEXTS      = [g["text"] for g in GUIDELINES]
GUIDELINE_CONDITIONS = [g["condition"] for g in GUIDELINES]

CONDITION_MAP = {
    "diabetes":           "diabetes",
    "diabetic":           "diabetes",
    "hba1c":              "diabetes",
    "heart failure":      "heart_failure",
    "chf":                "heart_failure",
    "hfref":              "heart_failure",
    "hypertension":       "hypertension",
    "high blood pressure":"hypertension",
    "ckd":                "ckd",
    "chronic kidney":     "ckd",
    "renal":              "ckd",
    "asthma":             "asthma",
    "dyslipidemia":       "dyslipidaemia",
    "dyslipidaemia":      "dyslipidaemia",
    "cholesterol":        "dyslipidaemia",
}

print("Embedding guidelines...")
guideline_embeddings = np.array(
    [embedder.get_embeddings([g])[0].values for g in GUIDELINE_TEXTS],
    dtype="float32"
)
guideline_index = faiss.IndexFlatL2(guideline_embeddings.shape[1])
guideline_index.add(guideline_embeddings)
print(f"✅ {len(GUIDELINES)} guidelines indexed")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_embedding(text: str) -> list:
    return embedder.get_embeddings([text[:3000]])[0].values

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
def get_claude_response(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text.strip()

def clean_json(text: str) -> str:
    text = re.sub(r"^```json", "", text.strip())
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 512) -> list:
    words        = text.split()
    approx_words = int(chunk_size / 1.3)
    if len(words) <= approx_words:
        return [text]
    return [" ".join(words[i:i+approx_words]) for i in range(0, len(words), approx_words)]

def index_note(url: str, text: str, patient_id: str = None) -> int:
    chunks  = chunk_text(text, chunk_size=512)
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id":       f"{abs(hash(url))}_{i}",
            "vector":   get_embedding(chunk),
            "data":     chunk,
            "metadata": {"url": url, "chunk": i, "patient_id": patient_id or ""}
        })
    notes_index.upsert(vectors=vectors)
    return len(chunks)

def detect_conditions(summary: dict) -> set:
    detected   = {"general"}
    search_txt = summary.get("primary_diagnosis", "").lower()
    for c in summary.get("comorbidities", []):
        search_txt += " " + c.lower()
    for keyword, tag in CONDITION_MAP.items():
        if keyword in search_txt:
            detected.add(tag)
    return detected

def retrieve_relevant_guidelines(query: str, summary: dict = None, k: int = 12) -> list:
    query_vec = np.array([get_embedding(query)], dtype="float32")
    D, I      = guideline_index.search(query_vec, min(k * 2, len(GUIDELINES)))
    if summary:
        detected = detect_conditions(summary)
        filtered = [GUIDELINE_TEXTS[i] for i in I[0] if GUIDELINE_CONDITIONS[i] in detected][:k]
        if len(filtered) < 3:
            filtered += [GUIDELINE_TEXTS[i] for i in I[0] if GUIDELINE_TEXTS[i] not in filtered][:k]
        return filtered[:k]
    return [GUIDELINE_TEXTS[i] for i in I[0][:k]]

def save_patient(summary: dict, source_note: str) -> str:
    patient_id = str(uuid.uuid4())[:8].upper()
    flag_count = len(summary.get("risk_flags", []))
    risk_level = "HIGH" if flag_count >= 3 else "MEDIUM" if flag_count >= 1 else "LOW"
    row = {
        "patient_id":        patient_id,
        "primary_diagnosis": summary.get("primary_diagnosis", ""),
        "procedure":         summary.get("procedure", ""),
        "comorbidities":     json.dumps(summary.get("comorbidities", [])),
        "medications":       json.dumps(summary.get("medications", [])),
        "key_findings":      json.dumps(summary.get("key_findings", [])),
        "risk_flags":        json.dumps(summary.get("risk_flags", [])),
        "follow_up_actions": json.dumps(summary.get("follow_up_actions", [])),
        "risk_level":        risk_level,
        "source_note":       source_note,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    errors = bq_client.insert_rows_json(BQ_TABLE, [row])
    if errors:
        print(f"❌ BigQuery error: {errors}")
    else:
        print(f"✅ Patient {patient_id} saved to BigQuery")
    return patient_id

async def scrape_url_async(client, url: str):
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.select_one("div.col-lg-9.mainContent")
        if main:
            text = main.get_text(separator=" ").strip()
            text = re.sub(r"\s+", " ", text)
            for stop in ["About This Sample:", "Legal & Usage Notice", "Related Samples"]:
                if stop in text:
                    text = text[:text.index(stop)]
            if len(text.split()) > 50:
                return url, text, None
        return url, None, "No clinical content found"
    except Exception as e:
        return url, None, str(e)

async def scrape_urls_async(urls: list):
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True
    ) as client:
        results = await asyncio.gather(*[scrape_url_async(client, url) for url in urls])
    return results


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class NoteRequest(BaseModel):
    note:       str
    max_tokens: Optional[int] = 1500

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class IngestRequest(BaseModel):
    urls:           List[str]
    auto_summarize: Optional[bool] = False

class IngestResult(BaseModel):
    url:        str
    success:    bool
    chunks:     int           = 0
    patient_id: Optional[str] = None
    error:      Optional[str] = None

class IngestResponse(BaseModel):
    results:      List[IngestResult] = []
    total_chunks: int                = 0

class CareGap(BaseModel):
    gap:            str
    guideline:      str
    recommendation: str
    priority:       str = "MEDIUM"

class SummaryResponse(BaseModel):
    patient_id:        Optional[str] = None
    primary_diagnosis: str           = ""
    comorbidities:     List[str]     = []
    medications:       List[str]     = []
    key_findings:      List[str]     = []
    risk_flags:        List[str]     = []
    follow_up_actions: List[str]     = []
    risk_level:        str           = "LOW"
    out_of_scope:      bool          = False
    reason:            Optional[str] = None

class CareGapsResponse(BaseModel):
    gaps:    List[CareGap] = []
    summary: str           = ""

class SearchResult(BaseModel):
    text:       str
    patient_id: Optional[str]   = None
    url:        str              = ""
    score:      Optional[float] = None

class SearchResponse(BaseModel):
    query:   str
    results: List[SearchResult] = []

class AskResponse(BaseModel):
    question: str
    answer:   str
    sources:  List[str] = []

class PatientRecord(BaseModel):
    patient_id:        str
    primary_diagnosis: str
    risk_level:        str
    created_at:        str
    comorbidities:     Optional[str] = None
    medications:       Optional[str] = None
    risk_flags:        Optional[str] = None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Clinical AI API",
        "version": "1.0.0",
        "docs":    "/docs",
        "endpoints": {
            "health":    "GET  /health",
            "patients":  "GET  /patients",
            "patient":   "GET  /patients/{id}",
            "summarize": "POST /summarize",
            "caregaps":  "POST /caregaps",
            "search":    "POST /search",
            "ask":       "POST /ask",
            "ingest":    "POST /ingest"
        }
    }

@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/summarize", response_model=SummaryResponse)
def summarize(req: NoteRequest):
    user_prompt = f"""
First determine if IN SCOPE for chronic disease management.
IN SCOPE: diabetes, hypertension, heart failure, CKD, asthma, dyslipidaemia.
OUT OF SCOPE: surgical procedures, burns, dermatology, acute injuries.

If OUT OF SCOPE: {{"out_of_scope": true, "reason": "explanation"}}

If IN SCOPE:
{{
  "out_of_scope": false,
  "primary_diagnosis": "specific diagnosis",
  "procedure": "procedure if any else empty string",
  "comorbidities": ["diagnosed conditions only"],
  "medications": ["each medication individually"],
  "key_findings": ["lab values, abnormal vitals"],
  "risk_flags": ["urgent concerns"],
  "follow_up_actions": ["next steps"]
}}

Clinical Note:
{req.note}
"""
    raw  = get_claude_response("Return valid JSON only — no markdown.", user_prompt, req.max_tokens)
    data = json.loads(clean_json(raw))

    if data.get("out_of_scope"):
        return SummaryResponse(out_of_scope=True, reason=data.get("reason"))

    flag_count = len(data.get("risk_flags", []))
    risk_level = "HIGH" if flag_count >= 3 else "MEDIUM" if flag_count >= 1 else "LOW"
    patient_id = save_patient(data, req.note)

    notes_index.upsert(vectors=[{
        "id":       f"patient_{patient_id}",
        "vector":   get_embedding(req.note),
        "data":     req.note,
        "metadata": {
            "patient_id":        patient_id,
            "primary_diagnosis": data.get("primary_diagnosis", ""),
            "risk_level":        risk_level,
            "url":               "api_entry",
            "chunk":             0
        }
    }])

    return SummaryResponse(
        patient_id=patient_id,
        primary_diagnosis=data.get("primary_diagnosis", ""),
        comorbidities=data.get("comorbidities", []),
        medications=data.get("medications", []),
        key_findings=data.get("key_findings", []),
        risk_flags=data.get("risk_flags", []),
        follow_up_actions=data.get("follow_up_actions", []),
        risk_level=risk_level
    )

@app.post("/caregaps", response_model=CareGapsResponse)
def caregaps(req: NoteRequest):
    try:
        quick_raw     = get_claude_response(
            "Return valid JSON only.",
            f'Extract: {{"primary_diagnosis": "...", "comorbidities": ["..."]}}\nNote: {req.note[:1000]}',
            max_tokens=300
        )
        quick_summary = json.loads(clean_json(quick_raw))
    except Exception:
        quick_summary = {"primary_diagnosis": req.note[:200], "comorbidities": []}

    guidelines     = retrieve_relevant_guidelines(req.note, summary=quick_summary, k=12)
    guidelines_txt = "\n".join([f"- {g}" for g in guidelines])

    user_prompt = f"""
Review note against guidelines. Return JSON:
{{
  "gaps": [{{"gap": "name", "guideline": "guideline", "recommendation": "action", "priority": "HIGH/MEDIUM/LOW"}}],
  "summary": "one sentence assessment"
}}

Only flag genuine missing items. Don't flag things already documented.

GUIDELINES:
{guidelines_txt}

NOTE: {req.note}
"""
    raw  = get_claude_response("Return valid JSON only.", user_prompt, req.max_tokens)
    data = json.loads(clean_json(raw))
    return CareGapsResponse(
        gaps=[CareGap(**g) for g in data.get("gaps", [])],
        summary=data.get("summary", "")
    )

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    scraped = asyncio.run(scrape_urls_async(req.urls))
    results = []
    total   = 0

    for url, text, error in scraped:
        if not text:
            results.append(IngestResult(url=url, success=False, error=error))
            continue

        chunks_added = index_note(url, text)
        total       += chunks_added
        patient_id   = None

        if req.auto_summarize:
            try:
                raw = get_claude_response(
                    "Return valid JSON only.",
                    f"""Extract as JSON:
{{
  "primary_diagnosis": "...",
  "comorbidities": [],
  "medications": [],
  "key_findings": [],
  "risk_flags": [],
  "follow_up_actions": []
}}
Note: {text[:3000]}"""
                )
                summary    = json.loads(clean_json(raw))
                patient_id = save_patient(summary, text)
                flag_count = len(summary.get("risk_flags", []))
                risk_level = "HIGH" if flag_count >= 3 else "MEDIUM" if flag_count >= 1 else "LOW"
                notes_index.upsert(vectors=[{
                    "id":       f"patient_{patient_id}",
                    "vector":   get_embedding(text[:1000]),
                    "data":     text[:1000],
                    "metadata": {
                        "patient_id":        patient_id,
                        "primary_diagnosis": summary.get("primary_diagnosis", ""),
                        "risk_level":        risk_level,
                        "url":               url,
                        "chunk":             0
                    }
                }])
            except Exception as e:
                print(f"Auto-summarize failed: {e}")

        results.append(IngestResult(
            url=url, success=True,
            chunks=chunks_added, patient_id=patient_id
        ))

    return IngestResponse(results=results, total_chunks=total)

@app.post("/search", response_model=SearchResponse)
def search(req: QueryRequest):
    results = notes_index.query(
        vector=get_embedding(req.query),
        top_k=req.top_k,
        include_metadata=True,
        include_data=True
    )
    return SearchResponse(
        query=req.query,
        results=[SearchResult(
            text=r.data or "",
            patient_id=r.metadata.get("patient_id"),
            url=r.metadata.get("url", ""),
            score=r.score
        ) for r in results]
    )

@app.post("/ask", response_model=AskResponse)
def ask(req: QueryRequest):
    results = notes_index.query(
        vector=get_embedding(req.query),
        top_k=6,
        include_metadata=True,
        include_data=True
    )
    if not results:
        return AskResponse(question=req.query, answer="No relevant notes found.", sources=[])

    context = "\n\n---\n\n".join([f"[Source {i+1}]\n{r.data}" for i, r in enumerate(results)])
    sources  = [r.metadata.get("url", "") for r in results]

    answer = get_claude_response(
        "Answer based ONLY on context. Cite sources (Source 1, 2 etc). Be clinically precise.",
        f"CONTEXT:\n{context}\n\nQUESTION: {req.query}"
    )
    return AskResponse(question=req.query, answer=answer, sources=sources)

@app.get("/patients", response_model=List[PatientRecord])
def get_patients(risk_level: Optional[str] = None, limit: int = 50):
    if risk_level:
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("risk_level", "STRING", risk_level)
        ])
        rows = list(bq_client.query(
            f"""SELECT patient_id, primary_diagnosis, risk_level, comorbidities,
               medications, risk_flags,
               FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) as created_at
               FROM `{BQ_TABLE}` WHERE risk_level = @risk_level
               ORDER BY created_at DESC LIMIT {limit}""",
            job_config=job_config
        ).result())
    else:
        rows = list(bq_client.query(
            f"""SELECT patient_id, primary_diagnosis, risk_level, comorbidities,
               medications, risk_flags,
               FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) as created_at
               FROM `{BQ_TABLE}` ORDER BY created_at DESC LIMIT {limit}"""
        ).result())
    return [PatientRecord(**dict(row)) for row in rows]

@app.get("/patients/{patient_id}", response_model=PatientRecord)
def get_patient(patient_id: str):
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id)
    ])
    rows = list(bq_client.query(
        f"""SELECT patient_id, primary_diagnosis, risk_level, comorbidities,
           medications, risk_flags,
           FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) as created_at
           FROM `{BQ_TABLE}` WHERE patient_id = @patient_id""",
        job_config=job_config
    ).result())
    if not rows:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return PatientRecord(**dict(rows[0]))


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)