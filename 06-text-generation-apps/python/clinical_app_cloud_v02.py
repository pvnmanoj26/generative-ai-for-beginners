import os
import json
import re
import uuid
import requests
import tempfile
import numpy as np
import faiss
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string, redirect, url_for, session
import anthropic
from dotenv import load_dotenv
from upstash_vector import Index
from google.cloud import bigquery
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, validator
from typing import List, Optional
from pathlib import Path
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Import ADK Ingestion functions
from adk_agents.ingestion import (
    profile_csv_file,
    propose_mapping_with_ai,
    ingest_csv_with_mapping,
    write_demographics_to_bigquery,
    write_events_to_bigquery,
    EVENT_TYPE_TO_FIELD
)

# ─────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────
load_dotenv()

# Set up secret key for session tracking
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "clinical-ai-secret-key-12345")
app.jinja_env.filters["fromjson"] = json.loads

# ─────────────────────────────────────────────
# DATA DIRECTORY
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "..", "..", "data")
os.makedirs(_DATA_DIR, exist_ok=True)

# BigQuery config
BQ_PROJECT = "healthcare-ai-manoj"
BQ_DATASET = "healthcare_ai"
BQ_TABLE   = f"{BQ_PROJECT}.{BQ_DATASET}.patient_summaries"
bq_client  = bigquery.Client(project=BQ_PROJECT)

claude   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

vertexai.init(project="healthcare-ai-manoj", location="us-central1")
embedder = TextEmbeddingModel.from_pretrained("text-embedding-004")

def get_embedding(text):
    """Get embedding vector — Vertex AI text-embedding-004"""
    result = embedder.get_embeddings([text[:3000]])
    return result[0].values

# Upstash - serverless
notes_collection = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

# ─────────────────────────────────────────────
# PYDANTIC MODELS — validate Claude's output
# ─────────────────────────────────────────────

class ClinicalSummary(BaseModel):
    out_of_scope:      bool          = False
    reason:            Optional[str] = None
    primary_diagnosis: str           = ""
    procedure:         Optional[str] = ""
    comorbidities:     List[str]     = []
    medications:       List[str]     = []
    key_findings:      List[str]     = []
    risk_flags:        List[str]     = []
    follow_up_actions: List[str]     = []

    @validator("medications", pre=True)
    def medications_must_be_list(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    @validator("comorbidities", "risk_flags", "key_findings",
               "follow_up_actions", pre=True)
    def lists_must_be_lists(cls, v):
        if isinstance(v, str):
            return [v] if v else []
        return v

    @validator("medications", each_item=True)
    def no_surgical_supplies(cls, v):
        surgical_keywords = [
            "monocryl", "nylon", "suture", "xylocaine", "lidocaine",
            "epinephrine", "silk", "vicryl", "prolene", "staple",
            "betadine", "chlorhexidine"
        ]
        for keyword in surgical_keywords:
            if keyword in v.lower():
                raise ValueError(f"Surgical supply in medications: {v}")
        return v

    @validator("comorbidities", each_item=True)
    def no_symptoms_as_comorbidities(cls, v):
        symptom_keywords = [
            "fatigue", "blurred vision", "nocturia", "polyuria",
            "polydipsia", "hemoptysis", "syncope", "dizziness",
            "shortness of breath", "chest pain", "nausea"
        ]
        for keyword in symptom_keywords:
            if keyword in v.lower():
                raise ValueError(f"Symptom in comorbidities: {v}")
        return v


class CareGap(BaseModel):
    gap:            str
    guideline:      str
    recommendation: str
    priority:       str = "MEDIUM"

    @validator("priority")
    def priority_must_be_valid(cls, v):
        if v.upper() not in {"HIGH", "MEDIUM", "LOW"}:
            return "MEDIUM"
        return v.upper()


class CareGapResult(BaseModel):
    gaps:    List[CareGap] = []
    summary: str           = ""

# ─────────────────────────────────────────────
# CLINICAL GUIDELINES KNOWLEDGE BASE
# ─────────────────────────────────────────────
GUIDELINES = [
    # ── DIABETES ──────────────────────────────
    {"condition": "diabetes", "text": "Diabetes: HbA1c should be measured every 3 months if uncontrolled (>8%), every 6 months if stable."},
    {"condition": "diabetes", "text": "Diabetes: Annual diabetic eye exam (fundoscopy/retinal screening) is required for all diabetic patients."},
    {"condition": "diabetes", "text": "Diabetes: Annual diabetic foot exam including monofilament sensation test is required."},
    {"condition": "diabetes", "text": "Diabetes: Urine microalbumin/creatinine ratio should be checked annually to screen for nephropathy."},
    {"condition": "diabetes", "text": "Diabetes: Blood pressure target for diabetic patients is <130/80 mmHg."},
    {"condition": "diabetes", "text": "Diabetes: Statin therapy is recommended for all diabetic patients aged 40-75."},
    {"condition": "diabetes", "text": "Diabetes: ACE inhibitor or ARB is recommended if microalbuminuria is present."},
    {"condition": "diabetes", "text": "Diabetes: Referral to endocrinology if HbA1c remains >9% despite treatment."},
    {"condition": "diabetes", "text": "Diabetes: Diabetes self-management education program enrollment is recommended at diagnosis."},
    {"condition": "diabetes", "text": "Diabetes: Annual flu vaccination is recommended for all diabetic patients."},
    {"condition": "diabetes", "text": "Diabetes: Pneumococcal vaccination recommended for all diabetic patients."},

    # ── HEART FAILURE ──────────────────────────
    {"condition": "heart_failure", "text": "Heart Failure: ACE inhibitor or ARB therapy is recommended for all HFrEF patients (EF <40%)."},
    {"condition": "heart_failure", "text": "Heart Failure: Beta-blocker therapy is recommended for all stable HFrEF patients."},
    {"condition": "heart_failure", "text": "Heart Failure: Aldosterone antagonist recommended for HFrEF patients with EF <35%."},
    {"condition": "heart_failure", "text": "Heart Failure: BNP or NT-proBNP should be measured to assess disease severity."},
    {"condition": "heart_failure", "text": "Heart Failure: Echocardiogram recommended to assess ejection fraction at diagnosis and after treatment changes."},
    {"condition": "heart_failure", "text": "Heart Failure: Daily weight monitoring and fluid restriction education is required."},
    {"condition": "heart_failure", "text": "Heart Failure: Cardiology referral is recommended for newly diagnosed heart failure."},
    {"condition": "heart_failure", "text": "Heart Failure: Annual flu and pneumococcal vaccination recommended for heart failure patients."},
    {"condition": "heart_failure", "text": "Heart Failure: Sodium restriction to <2g/day is recommended."},
    {"condition": "heart_failure", "text": "Heart Failure: 30-day readmission follow-up appointment is required post-discharge."},

    # ── HYPERTENSION ──────────────────────────
    {"condition": "hypertension", "text": "Hypertension: Blood pressure target is <130/80 mmHg for most patients per ACC/AHA guidelines."},
    {"condition": "hypertension", "text": "Hypertension: If blood pressure is above target despite medication, intensify antihypertensive therapy or add a second agent."},
    {"condition": "hypertension", "text": "Hypertension: First-line agents include thiazide diuretics, CCBs, ACE inhibitors, or ARBs."},
    {"condition": "hypertension", "text": "Hypertension: Annual renal function (eGFR, creatinine) and electrolyte monitoring required."},
    {"condition": "hypertension", "text": "Hypertension: Lifestyle modifications including DASH diet and exercise counseling are required."},
    {"condition": "hypertension", "text": "Hypertension: EKG recommended to assess for LVH in newly diagnosed hypertensive patients."},
    {"condition": "hypertension", "text": "Hypertension: Cardiovascular risk assessment (10-year ASCVD risk) should be calculated and documented."},

    # ── CKD ───────────────────────────────────
    {"condition": "ckd", "text": "CKD: eGFR and urine albumin-creatinine ratio should be monitored every 3-6 months."},
    {"condition": "ckd", "text": "CKD: Blood pressure target for CKD patients is <130/80 mmHg."},
    {"condition": "ckd", "text": "CKD: ACE inhibitor or ARB recommended for CKD patients with proteinuria."},
    {"condition": "ckd", "text": "CKD: Nephrology referral recommended when eGFR falls below 30 mL/min."},
    {"condition": "ckd", "text": "CKD: Anemia workup (CBC, iron studies, ESA consideration) recommended for CKD patients."},
    {"condition": "ckd", "text": "CKD: Dietary protein restriction and phosphate management counseling recommended."},
    {"condition": "ckd", "text": "CKD: Avoid NSAIDs, aminoglycosides, and nephrotoxic contrast agents — document counseling."},
    {"condition": "ckd", "text": "CKD: Secondary hyperparathyroidism — monitor PTH every 3-6 months in stage III-V CKD."},
    {"condition": "ckd", "text": "CKD: Hepatitis B vaccination recommended for all CKD patients not yet immune."},
    {"condition": "ckd", "text": "CKD: Flu and pneumococcal vaccination recommended for all CKD patients."},

    # ── ASTHMA ────────────────────────────────
    {"condition": "asthma", "text": "Asthma: Annual spirometry/pulmonary function test recommended to assess control."},
    {"condition": "asthma", "text": "Asthma: Inhaled corticosteroid is first-line controller therapy for persistent asthma."},
    {"condition": "asthma", "text": "Asthma: Asthma action plan should be documented and provided to patient."},
    {"condition": "asthma", "text": "Asthma: Referral to pulmonology if symptoms uncontrolled on moderate-dose ICS."},

    # ── DYSLIPIDAEMIA ─────────────────────────
    {"condition": "dyslipidaemia", "text": "Dyslipidaemia: Fasting lipid panel should be checked annually."},
    {"condition": "dyslipidaemia", "text": "Dyslipidaemia: Statin therapy recommended for patients with cardiovascular risk >10% (10-year ASCVD risk)."},
    {"condition": "dyslipidaemia", "text": "Dyslipidaemia: LDL target <70 mg/dL for very high cardiovascular risk patients."},
    {"condition": "dyslipidaemia", "text": "Dyslipidaemia: Lifestyle counseling on diet and exercise is required alongside pharmacotherapy."},

    # ── GENERAL / PREVENTIVE ──────────────────
    {"condition": "general", "text": "General: Smoking cessation counseling and pharmacotherapy is strongly recommended for all active smokers at every clinical encounter."},
    {"condition": "general", "text": "General: Annual influenza vaccination recommended for all patients with chronic conditions."},
    {"condition": "general", "text": "General: Pneumococcal vaccination (PCV15/PPSV23) recommended for immunocompromised patients and those with chronic disease."},
    {"condition": "general", "text": "General: BMI should be documented and obesity management counseling provided if BMI >30."},
]

# Separate texts and condition tags for retrieval
GUIDELINE_TEXTS      = [g["text"] for g in GUIDELINES]
GUIDELINE_CONDITIONS = [g["condition"] for g in GUIDELINES]

# Embed guidelines at startup (batched to avoid 50 sequential API calls)
print("Embedding clinical guidelines knowledge base...")
try:
    # Batch request all embeddings in 1 API call
    result = embedder.get_embeddings(GUIDELINE_TEXTS)
    guideline_embeddings = np.array([r.values for r in result], dtype="float32")
except Exception as e:
    print(f"⚠️ Batched embedding failed: {e}. Falling back to sequential calls...")
    # Fallback to sequential if batch fails
    guideline_embeddings = np.array(
        [get_embedding(g) for g in GUIDELINE_TEXTS], dtype="float32"
    )
guideline_index = faiss.IndexFlatL2(guideline_embeddings.shape[1])
guideline_index.add(guideline_embeddings)
print(f"✅ {len(GUIDELINES)} guidelines embedded and indexed.")

# ─────────────────────────────────────────────
# BigQuery Table Init
# ─────────────────────────────────────────────

def init_db():
    schema = [
        bigquery.SchemaField("patient_id",        "STRING"),
        bigquery.SchemaField("primary_diagnosis",  "STRING"),
        bigquery.SchemaField("procedure",          "STRING"),
        bigquery.SchemaField("comorbidities",      "STRING"),
        bigquery.SchemaField("medications",        "STRING"),
        bigquery.SchemaField("key_findings",       "STRING"),
        bigquery.SchemaField("risk_flags",         "STRING"),
        bigquery.SchemaField("follow_up_actions",  "STRING"),
        bigquery.SchemaField("risk_level",         "STRING"),
        bigquery.SchemaField("source_note",        "STRING"),
        bigquery.SchemaField("created_at",         "TIMESTAMP"),
    ]
    table = bigquery.Table(BQ_TABLE, schema=schema)
    bq_client.create_table(table, exists_ok=True)
    print(f"✅ BigQuery table initialised: {BQ_TABLE}")

def save_patient(summary, source_note):
    from datetime import datetime, timezone
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
        print(f"❌ BigQuery insert error: {errors}")
    else:
        print(f"✅ Patient {patient_id} saved to BigQuery")

    return patient_id

def get_all_patients():
    query = f"""
        SELECT patient_id, primary_diagnosis, procedure, comorbidities,
               medications, risk_flags, risk_level,
               FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) as created_at
        FROM `{BQ_TABLE}`
        ORDER BY created_at DESC
    """
    rows = bq_client.query(query).result()
    return [dict(row) for row in rows]

def get_patient_by_id(patient_id):
    query = f"""
        SELECT *
        FROM `{BQ_TABLE}`
        WHERE patient_id = @patient_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id)
        ]
    )
    rows = list(bq_client.query(query, job_config=job_config).result())
    return dict(rows[0]) if rows else None

def get_patients_by_risk(risk_level):
    query = f"""
        SELECT patient_id, primary_diagnosis, procedure, risk_flags,
               risk_level,
               FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) as created_at
        FROM `{BQ_TABLE}`
        WHERE risk_level = @risk_level
        ORDER BY created_at DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("risk_level", "STRING", risk_level)
        ]
    )
    rows = bq_client.query(query, job_config=job_config).result()
    return [dict(row) for row in rows]

def get_patient_stats():
    query = f"""
        SELECT
            COUNT(*) as total,
            COUNTIF(risk_level = 'HIGH')   as high,
            COUNTIF(risk_level = 'MEDIUM') as medium,
            COUNTIF(risk_level = 'LOW')    as low
        FROM `{BQ_TABLE}`
    """
    try:
        row = list(bq_client.query(query).result())[0]
        return {
            "total":  row.total,
            "high":   row.high,
            "medium": row.medium,
            "low":    row.low
        }
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"total": 0, "high": 0, "medium": 0, "low": 0}

init_db()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def get_claude_response(system_prompt, user_prompt, max_tokens=1500, temperature=0.0):
    try:
        response = claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Claude API error (will retry): {e}")
        raise

def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```",     "", text)
    text = re.sub(r"```$",     "", text)
    return text.strip()

def chunk_text(text, chunk_size=512):
    words        = text.split()
    approx_words = int(chunk_size / 1.3)
    if len(words) <= approx_words:
        return [text]
    return [
        " ".join(words[i:i+approx_words])
        for i in range(0, len(words), approx_words)
    ]

# Maps clinical keywords to condition categories
CONDITION_MAP = {
    "diabetes":           "diabetes",
    "diabetic":           "diabetes",
    "hyperglycemia":      "diabetes",
    "hba1c":              "diabetes",
    "heart failure":      "heart_failure",
    "chf":                "heart_failure",
    "cardiomyopathy":     "heart_failure",
    "hfref":              "heart_failure",
    "hypertension":       "hypertension",
    "high blood pressure":"hypertension",
    "ckd":                "ckd",
    "chronic kidney":     "ckd",
    "renal insufficiency":"ckd",
    "nephropathy":        "ckd",
    "kidney disease":     "ckd",
    "asthma":             "asthma",
    "dyslipidemia":       "dyslipidaemia",
    "dyslipidaemia":      "dyslipidaemia",
    "hyperlipidemia":     "dyslipidaemia",
    "cholesterol":        "dyslipidaemia",
}

def detect_conditions(summary):
    """Detect which condition categories apply to this patient"""
    detected = set(["general"])  # always include general guidelines
    search_text = summary.get("primary_diagnosis", "").lower()
    for c in summary.get("comorbidities", []):
        search_text += " " + c.lower()
    for keyword, tag in CONDITION_MAP.items():
        if keyword in search_text:
            detected.add(tag)
    return detected

def retrieve_relevant_guidelines(query, summary=None, k=12):
    """Filter guidelines by patient's actual conditions."""
    query_vec = np.array([get_embedding(query)], dtype="float32")
    D, I = guideline_index.search(query_vec, k=min(k*2, len(GUIDELINES)))

    if summary:
        detected_conditions = detect_conditions(summary)
        filtered = []
        for idx in I[0]:
            guideline_condition = GUIDELINE_CONDITIONS[idx]
            if guideline_condition in detected_conditions:
                filtered.append(GUIDELINE_TEXTS[idx])
            if len(filtered) >= k:
                break
        if len(filtered) < 3:
            for idx in I[0]:
                if GUIDELINE_TEXTS[idx] not in filtered:
                    filtered.append(GUIDELINE_TEXTS[idx])
                if len(filtered) >= k:
                    break
        return filtered
    else:
        return [GUIDELINE_TEXTS[i] for i in I[0][:k]]

async def scrape_url_async(client, url):
    """Async version of scraper"""
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        main = soup.select_one("div.col-lg-9.mainContent")
        if main:
            text = main.get_text(separator=" ").strip()
            text = re.sub(r"\s+", " ", text)
            match = re.search(r"Sample Name:", text)
            if not match:
                match = re.search(
                    r"(REASON FOR VISIT|CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|"
                    r"SUBJECTIVE|PREOPERATIVE DIAGNOSIS|ADMISSION DIAGNOSIS|"
                    r"CONSULTATION|DISCHARGE SUMMARY|PROCEDURE)", text
                )
            if match:
                text = text[match.start():]
            noise = [
                "Intended for: Medical transcription students, transcriptionists, and educators practicing clinical documentation formats in General Medicine.",
                "Discover more", "News", "Newspapers",
                "Secure transcription solutions", "Medical transcription software",
                "Nasal Sprays", "Drugs & Medications", "Health Conditions",
            ]
            for n in noise:
                text = text.replace(n, "")
            for stop in ["About This Sample:", "Legal & Usage Notice",
                         "Related Samples", "Keywords:", "Go Back to"]:
                if stop in text:
                    text = text[:text.index(stop)]
                    break
            text = re.sub(r"\s+", " ", text).strip()
            if len(text.split()) > 50:
                return url, text, None

        return url, None, "No clinical content found"

    except Exception as e:
        return url, None, str(e)

async def scrape_urls_async(urls):
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True
    ) as client:
        tasks = [scrape_url_async(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

def index_note(url, text, patient_id=None):
    chunks  = chunk_text(text, chunk_size=512)
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_id  = f"{abs(hash(url))}_{i}"
        embedding = get_embedding(chunk)
        vectors.append({
            "id":       chunk_id,
            "vector":   embedding,
            "data":     chunk,
            "metadata": {"url": url, "chunk": i, "patient_id": patient_id or ""}
        })
    notes_collection.upsert(vectors=vectors)
    return len(chunks)

def search_notes(query, top_k=5):
    query_vec = get_embedding(query)
    results   = notes_collection.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        include_data=True
    )
    if not results:
        return []
    return [(r.data or "", r.metadata) for r in results]

def base_context(**kwargs):
    stats = get_patient_stats()
    defaults = dict(
        summary=None, gaps=None,
        guideline_count=len(GUIDELINES),
        notes_count=notes_collection.info().vector_count,
        ingest_results=None,
        search_results=None,
        search_query=None,
        note_text=None,
        patient_id=None,
        patients=None,
        stats=stats,
        active_tab="summarize",
        ask_question=None,
        ask_answer=None,
        ask_sources=None,
        ask_chunks=None,
        out_of_scope_reason=None,
        csv_ingest_message=None,
        csv_ingest_error=None,
        copilot_query=None,
        copilot_response=None
    )
    defaults.update(kwargs)
    return defaults

# ─────────────────────────────────────────────
# FLASK ROUTING
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(TEMPLATE, **base_context())

@app.route("/patients")
def patients():
    risk = request.args.get("risk")
    if risk:
        patients_list = get_patients_by_risk(risk)
    else:
        patients_list = get_all_patients()
    return render_template_string(TEMPLATE, **base_context(patients=patients_list, active_tab="patients"))

@app.route("/patient/<patient_id>")
def patient_detail(patient_id):
    p = get_patient_by_id(patient_id)
    if not p:
        return "Patient not found", 404
    p["comorbidities"] = json.loads(p.get("comorbidities") or "[]")
    p["medications"] = json.loads(p.get("medications") or "[]")
    p["key_findings"] = json.loads(p.get("key_findings") or "[]")
    p["risk_flags"] = json.loads(p.get("risk_flags") or "[]")
    p["follow_up_actions"] = json.loads(p.get("follow_up_actions") or "[]")
    return render_template_string(PATIENT_DETAIL_TEMPLATE, patient=p)

@app.route("/summarize", methods=["POST"])
def run_summarizer():
    note = request.form.get("clinical_note", "")
    temp = float(request.form.get("temperature", 0.0))
    max_t = int(request.form.get("max_tokens", 1500))

    if not note.strip():
        return render_template_string(TEMPLATE, **base_context(note_text=note))

    system_prompt = (
        "You are an expert clinical summariser. Output valid, parsed JSON matching this structure: "
        '{"out_of_scope": bool, "reason": str or null, "primary_diagnosis": str, "procedure": str, '
        '"comorbidities": [str], "medications": [str], "key_findings": [str], "risk_flags": [str], "follow_up_actions": [str]}. '
        "Extract only facts present in the note. Do not list symptoms as comorbidities. Do not list surgical supplies as medications."
    )

    raw_json = get_claude_response(system_prompt, note, max_tokens=max_t, temperature=temp)
    cleaned = clean_json(raw_json)

    try:
        summary_dict = json.loads(cleaned)
        validated = ClinicalSummary(**summary_dict)
    except Exception as e:
        print(f"Validation failed: {e}")
        return render_template_string(TEMPLATE, **base_context(note_text=note, out_of_scope_reason=f"Failed validation: {e}"))

    if validated.out_of_scope:
        return render_template_string(TEMPLATE, **base_context(note_text=note, out_of_scope_reason=validated.reason))

    patient_id = save_patient(validated.model_dump(), note)
    return render_template_string(TEMPLATE, **base_context(summary=validated.model_dump(), patient_id=patient_id, note_text=note))

@app.route("/caregaps", methods=["POST"])
def run_caregaps():
    note = request.form.get("clinical_note", "")
    temp = float(request.form.get("temperature", 0.0))
    max_t = int(request.form.get("max_tokens", 1500))

    if not note.strip():
        return render_template_string(TEMPLATE, **base_context(note_text=note, active_tab="gaps"))

    # Determine conditions for guideline retrieval
    temp_summary = {"primary_diagnosis": note, "comorbidities": []}
    guidelines = retrieve_relevant_guidelines(note, temp_summary, k=10)
    guidelines_context = "\n".join([f"- {g}" for g in guidelines])

    system_prompt = (
        "You are a clinical care gap analyst. Compare the patient note against these guidelines:\n"
        f"{guidelines_context}\n\n"
        "Output valid JSON matching this schema: "
        '{"gaps": [{"gap": "string", "guideline": "string", "recommendation": "string", "priority": "HIGH|MEDIUM|LOW"}], '
        '"summary": "string"}. Only output gaps that are explicitly indicated as missing/incomplete in the note.'
    )

    raw_json = get_claude_response(system_prompt, note, max_tokens=max_t, temperature=temp)
    cleaned = clean_json(raw_json)

    try:
        result_dict = json.loads(cleaned)
        validated = CareGapResult(**result_dict)
    except Exception as e:
        return render_template_string(TEMPLATE, **base_context(note_text=note, active_tab="gaps", out_of_scope_reason=f"Parsing error: {e}"))

    return render_template_string(TEMPLATE, **base_context(gaps=validated.model_dump(), note_text=note, active_tab="gaps"))

@app.route("/ingest", methods=["POST"])
def run_ingest():
    urls_raw = request.form.get("urls", "")
    auto_summarize = "auto_summarize" in request.form
    urls = [u.strip() for u in urls_raw.split("\n") if u.strip()]

    if not urls:
        return redirect(url_for("index"))

    results = asyncio.run(scrape_urls_async(urls))
    ingest_results = []

    for url, text, err in results:
        if err:
            ingest_results.append({"url": url, "success": False, "error": err})
            continue

        patient_id = None
        if auto_summarize:
            system_prompt = (
                "You are an expert clinical summariser. Output valid, parsed JSON matching this structure: "
                '{"out_of_scope": bool, "reason": str or null, "primary_diagnosis": str, "procedure": str, '
                '"comorbidities": [str], "medications": [str], "key_findings": [str], "risk_flags": [str], "follow_up_actions": [str]}. '
            )
            try:
                raw_json = get_claude_response(system_prompt, text, max_tokens=1500, temperature=0.0)
                summary_dict = json.loads(clean_json(raw_json))
                validated = ClinicalSummary(**summary_dict)
                if not validated.out_of_scope:
                    patient_id = save_patient(validated.model_dump(), text)
            except Exception as e:
                print(f"Auto-summarize failed for ingested note: {e}")

        try:
            chunks = index_note(url, text, patient_id)
            ingest_results.append({"url": url, "success": True, "chunks": chunks, "patient_id": patient_id})
        except Exception as e:
            ingest_results.append({"url": url, "success": False, "error": str(e)})

    return render_template_string(TEMPLATE, **base_context(ingest_results=ingest_results, active_tab="ingest"))

@app.route("/search", methods=["POST"])
def run_search():
    query = request.form.get("query", "")
    if not query.strip():
        return render_template_string(TEMPLATE, **base_context(active_tab="search"))
    results = search_notes(query, top_k=5)
    return render_template_string(TEMPLATE, **base_context(search_results=results, search_query=query, active_tab="search"))

@app.route("/ask", methods=["POST"])
def run_ask():
    question = request.form.get("question", "")
    if not question.strip():
        return render_template_string(TEMPLATE, **base_context(active_tab="ask"))

    chunks = search_notes(question, top_k=5)
    if not chunks:
        return render_template_string(TEMPLATE, **base_context(
            ask_question=question,
            ask_answer="I couldn't find any relevant clinical notes in the system to answer your question.",
            active_tab="ask"
        ))

    context_str = ""
    sources = []
    seen_urls = {}
    source_counter = 1

    for doc, meta in chunks:
        url = meta.get("url", "Unknown Source")
        if url not in seen_urls:
            seen_urls[url] = source_counter
            sources.append((source_counter, url))
            source_counter += 1
        num = seen_urls[url]
        context_str += f"\n[Source {num}]: {doc}\n"

    system_prompt = (
        "You are a clinical Q&A assistant. Answer the user's question accurately using only the provided notes chunks.\n"
        "Cite the Source numbers (e.g. [Source 1]) in your answer where appropriate. If the context does not contain the answer, "
        "say 'I cannot find that information in the clinical notes.'"
    )

    user_prompt = f"Context clinical chunks:\n{context_str}\n\nQuestion: {question}"
    answer = get_claude_response(system_prompt, user_prompt, max_tokens=1500, temperature=0.0)

    return render_template_string(TEMPLATE, **base_context(
        ask_question=question,
        ask_answer=answer,
        ask_sources=sources,
        ask_chunks=chunks,
        active_tab="ask"
    ))


# ─────────────────────────────────────────────
# NEW ROUTING: CLINICAL COPILOT (AGENT)
# ─────────────────────────────────────────────

@app.route("/copilot", methods=["GET", "POST"])
def run_copilot():
    if request.method == "GET":
        return render_template_string(TEMPLATE, **base_context(active_tab="copilot"))
        
    message = request.form.get("message", "")
    if not message.strip():
        return render_template_string(TEMPLATE, **base_context(active_tab="copilot"))

    try:
        from adk_agents.agent import root_agent
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        async def run_agent(msg):
            runner = InMemoryRunner(agent=root_agent, app_name="clinical_copilot")
            session = await runner.session_service.create_session(
                app_name="clinical_copilot",
                user_id="web_user"
            )
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=msg)]
            )
            parts = []
            async for event in runner.run_async(
                user_id="web_user",
                session_id=session.id,
                new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
            return "\n".join(parts) if parts else "No response from Copilot."

        response_text = asyncio.run(run_agent(message))

    except Exception as e:
        import traceback
        response_text = f"Error running clinical orchestrator: {e}\n\n{traceback.format_exc()}"

    return render_template_string(
        TEMPLATE, 
        **base_context(
            copilot_query=message,
            copilot_response=response_text,
            active_tab="copilot"
        )
    )


# ─────────────────────────────────────────────
# NEW ROUTING: CSV INGESTION (HITL WEB UI)
# ─────────────────────────────────────────────

@app.route("/csv-ingest")
def csv_ingest_page():
    return render_template_string(TEMPLATE, **base_context(active_tab="csv_ingest"))

@app.route("/csv-ingest/upload", methods=["POST"])
def csv_ingest_upload():
    if "csv_file" not in request.files:
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error="No file part in request"
        ))
    
    file = request.files["csv_file"]
    if file.filename == "":
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error="No selected file"
        ))

    if not file.filename.endswith(".csv"):
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error="File must be a CSV (.csv)"
        ))

    # Save to a temporary file inside a unique directory to preserve the original filename stem
    temp_subdir = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}")
    os.makedirs(temp_subdir, exist_ok=True)
    temp_path = os.path.join(temp_subdir, file.filename)
    file.save(temp_path)

    try:
        # Profile the CSV file
        profile = profile_csv_file(temp_path)
        
        # Propose mapping using AI
        mapping = propose_mapping_with_ai(profile)
        
        # Detect category based on mapped target fields
        has_demographics = any(
            str(item.get("target_field") or "").startswith("demographics.") 
            for item in mapping.get("mappings", [])
        )
        category = "patients" if has_demographics else "events"
        
        # If event category, try to refine event type using CSV name
        if category == "events":
            # Check if csv stem corresponds to a known category, e.g. conditions
            stem = Path(file.filename).stem.lower()
            if stem in EVENT_TYPE_TO_FIELD:
                category = stem

        # Save mapping to session (so we don't have to regenerate it)
        session["temp_csv_path"] = temp_path
        session["temp_csv_name"] = file.filename
        session["proposed_mapping"] = mapping
        session["proposed_category"] = category

        # Render preview page
        return render_template_string(MAPPING_PREVIEW_TEMPLATE, 
                                     filename=file.filename,
                                     category=category,
                                     mapping=mapping)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            try:
                os.rmdir(os.path.dirname(temp_path))
            except Exception:
                pass
        print(f"Error profiling CSV: {e}")
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error=f"Error profiling/mapping CSV: {str(e)}"
        ))

@app.route("/csv-ingest/confirm", methods=["POST"])
def csv_ingest_confirm():
    temp_path = session.get("temp_csv_path")
    filename = session.get("temp_csv_name")
    mapping = session.get("proposed_mapping")
    category = session.get("proposed_category")
    action = request.form.get("action", "reject")

    if not temp_path or not os.path.exists(temp_path):
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error="Session expired or file not found. Please upload again."
        ))

    if action == "reject":
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            try:
                os.rmdir(os.path.dirname(temp_path))
            except Exception:
                pass
        session.pop("temp_csv_path", None)
        session.pop("temp_csv_name", None)
        session.pop("proposed_mapping", None)
        session.pop("proposed_category", None)
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_message="Ingestion cancelled by user. Proposed mapping discarded."
        ))

    try:
        # Set approved flag to pass the ADK safety check
        mapping["approved"] = True
        
        # User approved, ingest CSV using the mapping
        records = ingest_csv_with_mapping(temp_path, mapping)
        
        # Write to BigQuery depending on category
        if category == "patients":
            results = write_demographics_to_bigquery(records, BQ_PROJECT, BQ_DATASET)
        else:
            results = write_events_to_bigquery(records, category, BQ_PROJECT, BQ_DATASET)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            try:
                os.rmdir(os.path.dirname(temp_path))
            except Exception:
                pass
        session.pop("temp_csv_path", None)
        session.pop("temp_csv_name", None)
        session.pop("proposed_mapping", None)
        session.pop("proposed_category", None)

        rows_loaded = results.get("rows_attempted", 0)
        errors = results.get("errors", [])
        
        if errors:
            return render_template_string(TEMPLATE, **base_context(
                active_tab="csv_ingest",
                csv_ingest_error=f"Ingested {rows_loaded} rows, but encountered database errors: {errors}"
            ))
        else:
            return render_template_string(TEMPLATE, **base_context(
                active_tab="csv_ingest",
                csv_ingest_message=f"🎉 Successfully ingested {rows_loaded} rows from '{filename}' into table '{results.get('table')}'!"
            ))

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            try:
                os.rmdir(os.path.dirname(temp_path))
            except Exception:
                pass
        print(f"Error executing ingestion: {e}")
        return render_template_string(TEMPLATE, **base_context(
            active_tab="csv_ingest",
            csv_ingest_error=f"Ingestion execution failed: {str(e)}"
        ))


PATIENT_DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Patient Details — {{ patient.patient_id }}</title>
  <style>
    body   { font-family:Arial,sans-serif; max-width:960px; margin:40px auto; padding:0 20px; background:#f5f7fa; }
    h1     { color:#1a365d; }
    h2     { color:#2c5282; border-bottom:2px solid #bee3f8; padding-bottom:8px; }
    .card  { background:white; border-radius:8px; padding:20px; margin:20px 0; border-left:4px solid #553c9a; }
    .card-red { border-left-color:#c53030; }
    .card-yellow { border-left-color:#b7791f; }
    .card-green { border-left-color:#276749; }
    .btn   { display:inline-block; background:#2b6cb0; color:white; padding:10px 24px; border:none; border-radius:6px; cursor:pointer; font-size:15px; text-decoration:none; font-weight:600; }
    .btn:hover { background:#2c5282; }
    .tag   { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:bold; margin:2px; }
    .tag-high   { background:#fed7d7; color:#c53030; }
    .tag-medium { background:#fefcbf; color:#744210; }
    .tag-low    { background:#c6f6d5; color:#22543d; }
    .section { margin:16px 0; }
    .label { font-weight:bold; color:#4a5568; font-size:13px; text-transform:uppercase; letter-spacing:0.5px; display:block; margin-bottom:4px; }
    pre { background:#edf2f7; padding:16px; border-radius:6px; overflow-x:auto; font-family:monospace; font-size:13px; white-space:pre-wrap; }
  </style>
</head>
<body>

  <div style="margin-top:20px;">
    <a href="/patients" class="btn">⬅️ Back to Patient Directory</a>
  </div>

  <div class="card {% if patient.risk_level == 'HIGH' %}card-red{% elif patient.risk_level == 'MEDIUM' %}card-yellow{% else %}card-green{% endif %}">
    <h1>🧑‍⚕️ Patient ID: {{ patient.patient_id }}</h1>
    <div style="margin-bottom:20px;">
      <span class="tag {% if patient.risk_level == 'HIGH' %}tag-high{% elif patient.risk_level == 'MEDIUM' %}tag-medium{% else %}tag-low{% endif %}">
        {{ patient.risk_level }} RISK LEVEL
      </span>
      <span style="color:#718096; font-size:13px; margin-left:12px;">Created: {{ patient.created_at }}</span>
    </div>

    <div class="section">
      <span class="label">Primary Diagnosis</span>
      <p style="font-size:16px; font-weight:600; margin:0; color:#2d3748;">{{ patient.primary_diagnosis or '—' }}</p>
    </div>

    <div class="section">
      <span class="label">Procedure</span>
      <p style="font-size:15px; margin:0; color:#2d3748;">{{ patient.procedure or '—' }}</p>
    </div>

    <div class="section">
      <span class="label">Comorbidities</span>
      <ul style="margin:4px 0; padding-left:20px;">
        {% for c in patient.comorbidities %}
          <li>{{ c }}</li>
        {% else %}
          <li style="color:#718096; list-style:none; margin-left:-20px;">None documented</li>
        {% endfor %}
      </ul>
    </div>

    <div class="section">
      <span class="label">Medications</span>
      <ul style="margin:4px 0; padding-left:20px;">
        {% for m in patient.medications %}
          <li>{{ m }}</li>
        {% else %}
          <li style="color:#718096; list-style:none; margin-left:-20px;">None documented</li>
        {% endfor %}
      </ul>
    </div>

    <div class="section">
      <span class="label">Key Findings</span>
      <ul style="margin:4px 0; padding-left:20px;">
        {% for f in patient.key_findings %}
          <li>{{ f }}</li>
        {% else %}
          <li style="color:#718096; list-style:none; margin-left:-20px;">None documented</li>
        {% endfor %}
      </ul>
    </div>

    <div class="section">
      <span class="label">Risk Flags</span>
      {% for r in patient.risk_flags %}
        <span class="tag tag-high">⚠️ {{ r }}</span>
      {% else %}
        <span style="color:#276749; font-weight:600;">✅ No risk flags detected</span>
      {% endfor %}
    </div>

    <div class="section">
      <span class="label">Follow-Up Actions</span>
      <ul style="margin:4px 0; padding-left:20px;">
        {% for a in patient.follow_up_actions %}
          <li>{{ a }}</li>
        {% else %}
          <li style="color:#718096; list-style:none; margin-left:-20px;">None documented</li>
        {% endfor %}
      </ul>
    </div>

    <div class="section" style="margin-top:32px;">
      <span class="label">Raw Clinical Note / Source Text</span>
      <pre>{{ patient.source_note or '—' }}</pre>
    </div>
  </div>

</body>
</html>
"""


MAPPING_PREVIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Review Schema Mapping — {{ filename }}</title>
  <style>
    body   { font-family:Arial,sans-serif; max-width:960px; margin:40px auto; padding:0 20px; background:#f5f7fa; }
    h1     { color:#1a365d; }
    h2     { color:#2c5282; }
    .card  { background:white; border-radius:8px; padding:20px; margin:20px 0; border-left:4px solid #553c9a; }
    table  { width:100%; border-collapse:collapse; font-size:13px; margin:16px 0; }
    th     { background:#edf2f7; padding:12px; text-align:left; font-weight:600; }
    td     { padding:12px; border-bottom:1px solid #e2e8f0; }
    .btn   { color:white; padding:10px 24px; border:none; border-radius:6px; cursor:pointer; font-size:15px; margin-right:12px; font-weight:600; }
    .btn-green  { background:#276749; } .btn-green:hover  { background:#22543d; }
    .btn-red    { background:#c53030; } .btn-red:hover    { background:#9b2c2c; }
    .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:bold; }
    .badge-high { background:#c6f6d5; color:#22543d; }
    .badge-low  { background:#fed7d7; color:#c53030; }
    .sample { font-family: monospace; font-size:11px; color:#4a5568; background:#edf2f7; padding:2px 6px; border-radius:4px; }
  </style>
</head>
<body>

  <h1>📥 Human-in-the-Loop Mapping Approval</h1>
  <p style="color:#4a5568;">
    Review the AI-generated schema matches for <b>{{ filename }}</b>. 
    Category detected: <span class="badge" style="background:#e9d8fd; color:#553c9a; font-size:13px; padding:4px 12px;">{{ category | upper }}</span>
  </p>

  <div class="card">
    <h2>Proposed Column Mapping</h2>
    <table>
      <thead>
        <tr>
          <th>Source Column (CSV)</th>
          <th>Mapped Target Field</th>
          <th>Confidence</th>
          <th>Type</th>
          <th>Sample Values</th>
        </tr>
      </thead>
      <tbody>
        {% for item in mapping.mappings %}
        <tr>
          <td><b>{{ item.source_column }}</b></td>
          <td>
            {% if item.target_field %}
              <code style="color:#553c9a; font-weight:600;">{{ item.target_field }}</code>
            {% else %}
              <span style="color:#a0aec0; font-style:italic;">(Ignored / Unmapped)</span>
            {% endif %}
          </td>
          <td>
            {% if item.confidence is not none %}
              {% set score = item.confidence | float %}
              <span class="badge {% if score >= 0.8 %}badge-high{% else %}badge-low{% endif %}">
                {{ (score * 100) | int }}
              </span>
            {% else %}
              —
            {% endif %}
          </td>
          <td><span style="color:#718096; font-size:12px;">{{ item.reason or '—' }}</span></td>
          <td>
            {% if item.sample_values %}
              {% for val in item.sample_values[:3] %}
                <span class="sample">{{ val }}</span>
              {% endfor %}
            {% else %}
              —
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <div style="margin-top:24px; display:flex;">
      <form method="post" action="/csv-ingest/confirm">
        <input type="hidden" name="action" value="approve">
        <button type="submit" class="btn btn-green">✅ Approve &amp; Ingest to BigQuery</button>
      </form>
      <form method="post" action="/csv-ingest/confirm">
        <input type="hidden" name="action" value="reject">
        <button type="submit" class="btn btn-red">❌ Reject &amp; Discard</button>
      </form>
    </div>
  </div>

</body>
</html>
"""


# ─────────────────────────────────────────────
# MAIN HTML TEMPLATE
# ─────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Clinical AI Assistant</title>
  <style>
    body   { font-family:Arial,sans-serif; max-width:960px; margin:40px auto; padding:0 20px; background:#f5f7fa; }
    h1     { color:#1a365d; }
    h2     { color:#2c5282; border-bottom:2px solid #bee3f8; padding-bottom:8px; }
    textarea         { width:100%; height:200px; padding:10px; border:1px solid #cbd5e0; border-radius:6px; font-size:14px; }
    input[type=text] { width:100%; padding:10px; border:1px solid #cbd5e0; border-radius:6px; font-size:14px; }
    .btn        { background:#2b6cb0; color:white; padding:10px 24px; border:none; border-radius:6px; cursor:pointer; font-size:15px; margin-right:8px; }
    .btn:hover  { background:#2c5282; }
    .btn-green  { background:#276749; } .btn-green:hover  { background:#22543d; }
    .btn-purple { background:#553c9a; } .btn-purple:hover { background:#44337a; }
    .btn-orange { background:#c05621; } .btn-orange:hover { background:#9c4221; }
    .btn-red    { background:#c53030; } .btn-red:hover    { background:#9b2c2c; }
    .btn-yellow { background:#b7791f; } .btn-yellow:hover { background:#975a16; }
    .btn-sm     { padding:6px 14px; font-size:12px; }
    .card       { background:white; border-radius:8px; padding:20px; margin:20px 0; border-left:4px solid #2b6cb0; }
    .card-red   { border-left-color:#c53030; }
    .card-purple{ border-left-color:#553c9a; }
    .tag        { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:bold; margin:2px; }
    .tag-high   { background:#fed7d7; color:#c53030; }
    .tag-medium { background:#fefcbf; color:#744210; }
    .tag-low    { background:#c6f6d5; color:#22543d; }
    ul { padding-left:20px; } li { margin:6px 0; }
    .section    { margin:12px 0; }
    .label      { font-weight:bold; color:#4a5568; font-size:13px; text-transform:uppercase; letter-spacing:0.5px; }
    .tabs       { display:flex; margin-bottom:20px; flex-wrap:wrap; }
    .tab        { padding:10px 18px; cursor:pointer; border:1px solid #cbd5e0; background:#edf2f7; font-weight:500; font-size:13px; }
    .tab.active { background:#2b6cb0; color:white; border-color:#2b6cb0; }
    .tab:first-child { border-radius:6px 0 0 6px; }
    .tab:last-child  { border-radius:0 6px 6px 0; }
    .form-panel { display:none; }
    .form-panel.active { display:block; }
    .controls   { display:flex; gap:32px; margin-bottom:16px; background:#edf2f7; padding:14px; border-radius:8px; flex-wrap:wrap; }
    .control-group label { font-weight:600; font-size:13px; }
    .control-group input[type=range] { width:200px; display:block; margin:4px 0; }
    .hint       { font-size:11px; color:#718096; }
    .count-badge{ background:#ebf8ff; color:#2b6cb0; padding:4px 12px; border-radius:12px; font-weight:600; font-size:13px; }
    .stat-row   { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
    .stat-box   { padding:12px 18px; border-radius:8px; text-align:center; min-width:80px; }
    .stat-box .num { font-size:24px; font-weight:700; }
    .stat-box .lbl { font-size:11px; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th    { background:#edf2f7; padding:10px; text-align:left; }
    td    { padding:10px; border-bottom:1px solid #e2e8f0; }
    .source-badge { font-size:11px; color:#718096; margin-bottom:6px; }
    .note-text    { font-size:13px; color:#2d3748; line-height:1.6; }
    .ingest-row   { padding:8px 0; border-bottom:1px solid #e2e8f0; font-size:13px; }
    .banner { border:1px solid; border-radius:8px; padding:12px 16px; margin-bottom:16px; font-weight:600; }
    .banner-green { background:#c6f6d5; border-color:#68d391; color:#22543d; }
    .banner-red { background:#fed7d7; border-color:#fc8181; color:#9b2c2c; }
  </style>
  <script>
    function switchTab(tab) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.form-panel').forEach(p => p.classList.remove('active'));
      document.getElementById('tab-' + tab).classList.add('active');
      document.getElementById('panel-' + tab).classList.add('active');
    }
  </script>
</head>
<body>

  <h1>🏥 Clinical AI Assistant</h1>
  <p style="color:#718096; margin-bottom:12px;">
    Powered by Claude + RAG · BigQuery patient store · Upstash vector search
  </p>

  <!-- STATS BAR -->
  <div class="stat-row">
    <div class="stat-box" style="background:#ebf8ff; color:#2b6cb0;">
      <div class="num">{{ stats.total }}</div><div class="lbl">Patients</div>
    </div>
    <div class="stat-box" style="background:#fff5f5; color:#c53030;">
      <div class="num">{{ stats.high }}</div><div class="lbl">High Risk</div>
    </div>
    <div class="stat-box" style="background:#fffff0; color:#b7791f;">
      <div class="num">{{ stats.medium }}</div><div class="lbl">Medium Risk</div>
    </div>
    <div class="stat-box" style="background:#f0fff4; color:#276749;">
      <div class="num">{{ stats.low }}</div><div class="lbl">Low Risk</div>
    </div>
    <div class="stat-box" style="background:#faf5ff; color:#553c9a;">
      <div class="num">{{ notes_count }}</div><div class="lbl">Note Chunks</div>
    </div>
    <div class="stat-box" style="background:#fffaf0; color:#c05621;">
      <div class="num">{{ guideline_count }}</div><div class="lbl">Guidelines</div>
    </div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <div class="tab {% if active_tab == 'summarize' %}active{% endif %}" id="tab-summarize" onclick="switchTab('summarize')">📋 Summarizer</div>
    <div class="tab {% if active_tab == 'gaps' %}active{% endif %}"      id="tab-gaps"      onclick="switchTab('gaps')">🔍 Care Gaps</div>
    <div class="tab {% if active_tab == 'ingest' %}active{% endif %}"    id="tab-ingest"    onclick="switchTab('ingest')">📥 Ingest Notes</div>
    <div class="tab {% if active_tab == 'csv_ingest' %}active{% endif %}" id="tab-csv_ingest" onclick="switchTab('csv_ingest')">📊 CSV Ingestion</div>
    <div class="tab {% if active_tab == 'copilot' %}active{% endif %}"    id="tab-copilot"    onclick="switchTab('copilot')">🤖 Clinical Copilot</div>
    <div class="tab {% if active_tab == 'search' %}active{% endif %}"    id="tab-search"    onclick="switchTab('search')">🔎 Search Notes</div>
    <div class="tab {% if active_tab == 'patients' %}active{% endif %}"  id="tab-patients"  onclick="switchTab('patients')">🗂️ Patients</div>
    <div class="tab {% if active_tab == 'ask' %}active{% endif %}"       id="tab-ask"       onclick="switchTab('ask')">💬 Ask Notes</div>
  </div>

  <!-- ── BANNERS ────────────────────────────────────────────── -->
  {% if csv_ingest_message %}
    <div class="banner banner-green">{{ csv_ingest_message }}</div>
  {% endif %}
  {% if csv_ingest_error %}
    <div class="banner banner-red">{{ csv_ingest_error }}</div>
  {% endif %}

  <!-- ── SUMMARIZER ─────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'summarize' %}active{% endif %}" id="panel-summarize">
    <form method="post" action="/summarize">
      <p style="color:#4a5568;">
        Paste any clinical note or discharge summary.
        Structured data is automatically saved to the patient database.
      </p>
      <textarea name="clinical_note" placeholder="Paste clinical note here...">{{ note_text or '' }}</textarea>
      <br><br>
      <div class="controls">
        <div class="control-group">
          <label>🌡️ Temperature: <span id="ts">0.0</span></label>
          <input type="range" name="temperature" min="0" max="1" step="0.1" value="0.0"
            oninput="document.getElementById('ts').textContent=this.value">
          <div class="hint">0.0 = precise &nbsp;|&nbsp; 1.0 = creative</div>
        </div>
        <div class="control-group">
          <label>📏 Max Tokens: <span id="ms">1500</span></label>
          <input type="range" name="max_tokens" min="200" max="2000" step="100" value="1500"
            oninput="document.getElementById('ms').textContent=this.value">
          <div class="hint">200 = brief &nbsp;|&nbsp; 2000 = detailed</div>
        </div>
      </div>
      <button type="submit" class="btn">📋 Generate Summary + Save Patient</button>
    </form>
    
    {% if summary %}
    <div class="card">
      <h2>📋 Generated Summary</h2>
      <p style="color:#718096; font-size:13px;">Saved as Patient ID: <b>{{ patient_id }}</b></p>
      <div class="section"><span class="label">Primary Diagnosis:</span> {{ summary.primary_diagnosis }}</div>
      <div class="section"><span class="label">Procedure:</span> {{ summary.procedure or '—' }}</div>
      <div class="section">
        <span class="label">Comorbidities:</span>
        <ul>{% for c in summary.comorbidities %}<li>{{ c }}</li>{% endfor %}</ul>
      </div>
      <div class="section">
        <span class="label">Medications:</span>
        <ul>{% for m in summary.medications %}<li>{{ m }}</li>{% endfor %}</ul>
      </div>
      <div class="section">
        <span class="label">Key Findings:</span>
        <ul>{% for f in summary.key_findings %}<li>{{ f }}</li>{% endfor %}</ul>
      </div>
      <div class="section">
        <span class="label">Risk Flags:</span>
        {% for r in summary.risk_flags %}
          <span class="tag tag-high">⚠️ {{ r }}</span>
        {% else %}
          <span style="color:#276749;">✅ No risk flags detected</span>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if out_of_scope_reason %}
    <div class="card card-red">
      <h2>⚠️ Summary Rejected</h2>
      <p style="color:#c53030;">{{ out_of_scope_reason }}</p>
    </div>
    {% endif %}
  </div>

  <!-- ── CARE GAPS ───────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'gaps' %}active{% endif %}" id="panel-gaps">
    <form method="post" action="/caregaps">
      <p style="color:#4a5568;">
        Paste a patient's clinical note. Checked against {{ guideline_count }} embedded clinical guidelines.
      </p>
      <textarea name="clinical_note" placeholder="Paste patient clinical note here...">{{ note_text or '' }}</textarea>
      <br><br>
      <button type="submit" class="btn btn-green">🔍 Identify Care Gaps</button>
    </form>
    
    {% if gaps %}
    <div class="card card-purple">
      <h2>🔍 Clinical Care Gaps</h2>
      {% for gap in gaps.gaps %}
      <div style="margin:16px 0; padding:12px; background:#faf5ff; border-radius:6px; border-left:3px solid #553c9a;">
        <span class="tag {% if gap.priority == 'HIGH' %}tag-high{% elif gap.priority == 'MEDIUM' %}tag-medium{% else %}tag-low{% endif %}">
          {{ gap.priority }} Priority
        </span>
        <div style="margin-top:6px;"><b>Gap:</b> {{ gap.gap }}</div>
        <div><b>Based on Guideline:</b> <i>{{ gap.guideline }}</i></div>
        <div style="color:#2d3748; margin-top:4px;"><b>Recommendation:</b> {{ gap.recommendation }}</div>
      </div>
      {% endfor %}
      <div style="margin-top:16px; border-top:1px solid #e2e8f0; padding-top:12px;">
        <h3>Synthesis / Summary</h3>
        <p style="font-size:14px; line-height:1.6;">{{ gaps.summary }}</p>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- ── INGEST ──────────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'ingest' %}active{% endif %}" id="panel-ingest">
    <form method="post" action="/ingest">
      <p style="color:#4a5568;">
        Paste one or more URLs — one per line. Works with MTSamples or any page with clinical text.<br>
      </p>
      <textarea name="urls" placeholder="https://mtsamples.com/..."></textarea>
      <br><br>
      <div style="margin-bottom:16px; padding:12px; background:#edf2f7; border-radius:8px; display:flex; align-items:center; gap:10px;">
        <input type="checkbox" name="auto_summarize" id="auto_summarize" style="width:16px; height:16px;">
        <label for="auto_summarize" style="font-size:14px; font-weight:500; cursor:pointer;">
          🤖 Auto-summarize &amp; create patient records
        </label>
      </div>
      <button type="submit" class="btn btn-purple">📥 Scrape &amp; Index Notes</button>
    </form>
    
    {% if ingest_results %}
    <div class="card card-purple" style="margin-top:20px;">
      <h2>📥 Ingest Results</h2>
      {% for r in ingest_results %}
      <div class="ingest-row">
        {% if r.success %}
          ✅ <b>{{ r.url[:70] }}</b> — {{ r.chunks }} chunks indexed
          {% if r.patient_id %}
            | 🧑⚕️ <a href="/patient/{{ r.patient_id }}" style="color:#553c9a;">Patient {{ r.patient_id }}</a>
          {% endif %}
        {% else %}
          ❌ <b>{{ r.url[:70] }}</b> — {{ r.error or 'Failed' }}
        {% endif %}
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </div>

  <!-- ── CSV INGESTION ────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'csv_ingest' %}active{% endif %}" id="panel-csv_ingest">
    <form method="post" action="/csv-ingest/upload" enctype="multipart/form-data">
      <h2>📊 Ingest CSV Datasets</h2>
      <p style="color:#4a5568; margin-bottom:18px;">
        Upload CSV files (like demographics <code>patients.csv</code> or event-based <code>conditions.csv</code>). 
        The system will auto-detect the category, suggest AI column mappings, and wait for your approval before inserting records into BigQuery.
      </p>
      
      <div style="background:white; border:2px dashed #cbd5e0; border-radius:8px; padding:32px; text-align:center; margin-bottom:20px;">
        <input type="file" name="csv_file" accept=".csv" required style="font-size:15px; margin-bottom:12px;"><br>
        <span class="hint">Upload file format: CSV (.csv) only</span>
      </div>

      <button type="submit" class="btn btn-purple">Upload &amp; Propose Mapping</button>
    </form>
  </div>

  <!-- ── COPILOT ────────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'copilot' %}active{% endif %}" id="panel-copilot">
    <form method="post" action="/copilot">
      <h2>🤖 Clinical Copilot (AI Agent Orchestrator)</h2>
      <p style="color:#4a5568; margin-bottom:18px;">
        Chat with the clinical orchestrator. You can ask analytical questions (e.g. <i>"How many patients have high risk level?"</i>), 
        request care gap analysis, search notes, or upload and ask it to ingest files.
      </p>
      <input type="text" name="message" placeholder="Ask the Copilot anything..." value="{{ copilot_query or '' }}">
      <br><br>
      <button type="submit" class="btn btn-purple">Send to Copilot</button>
    </form>
    
    {% if copilot_response %}
    <div class="card card-purple" style="margin-top:20px;">
      <h2>🤖 Copilot Response</h2>
      <div style="font-size:14px; color:#2d3748; line-height:1.8; white-space:pre-wrap;">{{ copilot_response }}</div>
    </div>
    {% endif %}
  </div>

  <!-- ── SEARCH ──────────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'search' %}active{% endif %}" id="panel-search">
    <form method="post" action="/search">
      <p style="color:#4a5568;">Semantic search across all indexed notes.</p>
      <input type="text" name="query" placeholder="e.g. patients with uncontrolled diabetes" value="{{ search_query or '' }}">
      <br><br>
      <button type="submit" class="btn btn-orange">🔎 Search</button>
    </form>
    
    {% if search_results %}
    <div class="card" style="margin-top:20px;">
      <h2>🔎 Search Results</h2>
      {% for doc, meta in search_results %}
      <div style="margin:16px 0; padding:14px; background:#f7fafc; border-radius:6px; border-left:3px solid #2b6cb0;">
        <div class="source-badge">
          Source: {{ meta.url }}
          {% if meta.patient_id %} | Patient: <a href="/patient/{{ meta.patient_id }}" style="color:#2b6cb0;">{{ meta.patient_id }}</a>{% endif %}
        </div>
        <div class="note-text">{{ doc }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </div>

  <!-- ── ASK ─────────────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'ask' %}active{% endif %}" id="panel-ask">
    <form method="post" action="/ask">
      <input type="text" name="question" placeholder="Ask anything about your clinical notes..." value="{{ ask_question or '' }}">
      <br><br>
      <button type="submit" class="btn" style="background:#0d6efd;">💬 Ask</button>
    </form>

    {% if ask_answer %}
    <div class="card" style="margin-top:20px; border-left-color:#0d6efd;">
      <h2>💬 Answer</h2>
      <div style="font-size:14px; color:#2d3748; line-height:1.8; white-space:pre-wrap;">{{ ask_answer }}</div>
    </div>
    {% endif %}
  </div>

  <!-- ── PATIENTS ────────────────────────────────────────────── -->
  <div class="form-panel {% if active_tab == 'patients' %}active{% endif %}" id="panel-patients">
    <div style="display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap;">
      <a href="/patients" style="text-decoration:none;"><button class="btn btn-sm">All</button></a>
      <a href="/patients?risk=HIGH" style="text-decoration:none;"><button class="btn btn-sm btn-red">High Risk</button></a>
      <a href="/patients?risk=MEDIUM" style="text-decoration:none;"><button class="btn btn-sm btn-yellow">Medium Risk</button></a>
      <a href="/patients?risk=LOW" style="text-decoration:none;"><button class="btn btn-sm btn-green">Low Risk</button></a>
    </div>
    
    {% if patients %}
    <div class="card">
      <h2>🗂️ Patient Records</h2>
      <table>
        <thead>
          <tr>
            <th>Patient ID</th><th>Primary Diagnosis</th><th>Procedure</th><th>Risk Flags</th><th>Risk Level</th><th>Action</th>
          </tr>
        </thead>
        <tbody>
          {% for p in patients %}
          <tr>
            <td style="font-weight:600; color:#2b6cb0;">{{ p.patient_id }}</td>
            <td>{{ p.primary_diagnosis[:55] }}</td>
            <td>{{ p.procedure or '—' }}</td>
            <td>
              {% set flags = p.risk_flags | fromjson %}
              {% for f in flags[:2] %}<span class="tag tag-high" style="font-size:10px;">{{ f[:25] }}</span>{% endfor %}
            </td>
            <td>
              <span class="tag {% if p.risk_level == 'HIGH' %}tag-high{% elif p.risk_level == 'MEDIUM' %}tag-medium{% else %}tag-low{% endif %}">
                {{ p.risk_level }}
              </span>
            </td>
            <td><a href="/patient/{{ p.patient_id }}"><button class="btn btn-sm btn-purple" style="padding:4px 10px;">Details</button></a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
  </div>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)