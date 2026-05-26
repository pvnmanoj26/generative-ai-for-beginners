import os
import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from adk_agents.ingestion import (
    detect_csv_category,
    propose_and_preview_mapping,
    prompt_user_approval,
    ingest_csv_data,
    run_bigquery_query,    # ← Add this
)

load_dotenv()

API_BASE = "https://clinical-ai-api-230808425514.us-central1.run.app"
REQUEST_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


def _api_error(exc: Exception, path: str) -> dict:
    """Return tool-safe errors so ADK does not cancel the agent node."""
    error = {
        "error": type(exc).__name__,
        "message": str(exc),
        "path": path,
        "retryable": isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)),
    }
    if isinstance(exc, httpx.HTTPStatusError):
        error["status_code"] = exc.response.status_code
        error["response"] = exc.response.text[:500]
    return error


def _request(method: str, path: str, **kwargs) -> dict:
    try:
        response = httpx.request(
            method,
            f"{API_BASE}{path}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return _api_error(exc, path)


def _is_error(result: dict) -> bool:
    return isinstance(result, dict) and "error" in result


def _safe_limit(limit: int, default: int = 5, maximum: int = 10) -> int:
    try:
        return min(max(int(limit), 1), maximum)
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
def list_patients(risk_level: str = None, limit: int = 10) -> dict:
    """
    List patients from BigQuery.
    Optionally filter by risk_level: HIGH, MEDIUM, or LOW.
    Returns patient IDs and diagnoses without clinical notes.
    """
    params = {"limit": _safe_limit(limit, default=10)}
    if risk_level:
        params["risk_level"] = risk_level.upper()
    patients = _request("GET", "/patients", params=params)
    if _is_error(patients):
        return patients
    if isinstance(patients, dict):
        patients = patients.get("patients", patients)
    if isinstance(patients, list):
        for p in patients:
            p.pop("source_note", None)
    return {"patients": patients}


def get_population_gaps(risk_level: str = "HIGH", limit: int = 5) -> dict:
    """
    Get care gaps for multiple patients at once.
    Prefer this over analyzing patients one by one.
    Returns prioritised gaps sorted by severity.
    """
    params   = {"limit": _safe_limit(limit)}
    if risk_level:
        params["risk_level"] = risk_level.upper()
    patients = _request("GET", "/patients", params=params)

    if _is_error(patients):
        return patients

    if isinstance(patients, dict):
        patients = patients.get("patients", [])

    results = []
    errors = []
    for p in patients:
        pid  = p.get("patient_id")
        if not pid:
            continue
        full = _request("GET", f"/patients/{pid}")
        if _is_error(full):
            errors.append({"patient_id": pid, **full})
            continue
        note = full.get("source_note", "")
        if not note:
            continue
        gaps = _request("POST", "/caregaps", json={"note": note})
        if _is_error(gaps):
            errors.append({"patient_id": pid, **gaps})
            continue
        results.append({
            "patient_id":        pid,
            "primary_diagnosis": p.get("primary_diagnosis", ""),
            "risk_level":        p.get("risk_level", ""),
            "gap_count":         len(gaps.get("gaps", [])),
            "gaps":              gaps.get("gaps", []),
            "summary":           gaps.get("summary", "")
        })

    results.sort(key=lambda x: x["gap_count"], reverse=True)
    return {
        "population_gaps": results,
        "total_patients":  len(results),
        "total_gaps":      sum(r["gap_count"] for r in results),
        "errors":          errors,
        "critical_gaps":   sum(
            1 for r in results
            for g in r["gaps"]
            if g.get("priority") == "HIGH"
        )
    }


def search_clinical_notes(query: str, top_k: int = 5) -> dict:
    """
    Semantic search over all indexed clinical notes.
    Use for finding notes about specific conditions or symptoms.
    """
    return _request("POST", "/search",
                    json={"query": query, "top_k": _safe_limit(top_k)})


def ask_clinical_question(question: str) -> dict:
    """
    Ask a clinical question using RAG over indexed notes.
    Best for cross-patient questions or protocol queries.
    """
    return _request("POST", "/ask", json={"query": question})


# ─────────────────────────────────────────────
# SUB-AGENTS
# ─────────────────────────────────────────────
model = LiteLlm(model="anthropic/claude-haiku-4-5")

care_gap_analyst = Agent(
    name="care_gap_analyst",
    model=model,
    description="Analyses patient populations to identify and prioritise care gaps.",
    instruction="""You are a clinical care gap analyst.
Use get_population_gaps to analyse multiple patients efficiently.
Always sort gaps by priority — HIGH first.
Be concise and cite patient IDs.""",
    tools=[list_patients, get_population_gaps],
)

clinical_search = Agent(
    name="clinical_search",
    model=model,
    description="Searches indexed clinical notes and answers clinical questions.",
    instruction="""You are a clinical knowledge assistant.
Use search_clinical_notes for finding relevant notes.
Use ask_clinical_question for synthesised answers across notes.
Be precise and cite sources.""",
    tools=[search_clinical_notes, ask_clinical_question],
)

# ─────────────────────────────────────────────
# INGESTION SUB-AGENT
# ─────────────────────────────────────────────
ingestion_agent = Agent(
    name="ingestion_agent",
    model=model,
    description="Handles clinical file ingestion, category detection, schema mapping, and BigQuery loading.",
    instruction="""You are a clinical data ingestion specialist.
When given a file path:
1. Call detect_csv_category to identify the clinical category.
2. Call propose_and_preview_mapping to generate mappings and get the markdown table preview.
3. Call prompt_user_approval with the preview markdown to get explicit human authorization.
4. Call ingest_csv_data with the file path, category, and approval status to execute the load.
If the user rejects the mapping, do not ingest and report that ingestion was cancelled.""",
    tools=[detect_csv_category, propose_and_preview_mapping, prompt_user_approval, ingest_csv_data],
)

# New analytics agent
analytics_agent = Agent(
    name="analytics_agent",
    model=model,
    description="Runs SQL queries against the BigQuery healthcare dataset to answer analytical questions.",
    instruction="""You are a clinical data analyst.
You have access to the BigQuery dataset: healthcare-ai-manoj.healthcare_ai
Available tables:
- patients (patient_id, first_name, last_name, birthdate, gender, race, ethnicity, address, city, state, zip)
- conditions (patient_id, code, description, start, stop, status)
- observations (patient_id, code, description, start, value, unit)
- medications (patient_id, code, description, start, stop)
When asked an analytical question:
1. Write a SQL query against the correct table(s).
2. Call run_bigquery_query with the SQL.
3. Summarize the results clearly.""",
    tools=[run_bigquery_query],
)


# ─────────────────────────────────────────────
# ROOT ORCHESTRATOR
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# ROOT ORCHESTRATOR
# ─────────────────────────────────────────────
root_agent = Agent(
    name="clinical_orchestrator",
    model=model,
    description="Orchestrates clinical AI agents to ingest data, answer questions, and analyze care gaps.",
    instruction="""You are a clinical AI orchestrator.

Route tasks to the right sub-agent:
- Ingesting, loading, or mapping raw clinical data files  → ingestion_agent
- Analytics, SQL queries, counting, aggregations → analytics_agent
- Care gaps, risk analysis, patient lists                 → care_gap_analyst
- Searching notes, clinical questions                     → clinical_search

Synthesise results into a clear clinical report.""",
    sub_agents=[ingestion_agent, analytics_agent,care_gap_analyst, clinical_search],
)
