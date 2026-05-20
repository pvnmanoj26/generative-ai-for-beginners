"""
Optimised ReAct care-gap agent.

Optimisations:
- Uses Haiku (20x cheaper than Sonnet)
- source_note stripped from patient results (60% token reduction)
- analyze_care_gaps takes patient_id not full note
- population-level tool to avoid per-patient loops
- max iterations capped at 5

Usage:
    python care_gap_agent.py
    python care_gap_agent.py "What are the most common care gaps in HIGH risk patients?"
"""

import os
import sys
import json
import httpx
import anthropic
from dotenv import load_dotenv

load_dotenv()

API_BASE       = "https://clinical-ai-api-230808425514.us-central1.run.app"
MODEL          = "claude-haiku-4-5"
MAX_TOKENS     = 2048
MAX_ITERATIONS = 5

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────
def _get(path: str, params: dict = None) -> dict:
    r = httpx.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{API_BASE}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────
def list_patients(risk_level: str = None, limit: int = 10) -> dict:
    """GET /patients — strips source_note to save tokens."""
    params = {"limit": limit}
    if risk_level:
        params["risk_level"] = risk_level.upper()
    patients = _get("/patients", params=params)
    # Strip source_note — biggest token waste
    if isinstance(patients, list):
        for p in patients:
            p.pop("source_note", None)
    return {"patients": patients}


def get_patient(patient_id: str) -> dict:
    """GET /patients/{id} — strips source_note to save tokens."""
    result = _get(f"/patients/{patient_id}")
    result.pop("source_note", None)
    return result


def analyze_care_gaps(patient_id: str) -> dict:
    """
    Analyze care gaps for a patient by their ID.
    Fetches note internally — never sends full note to Claude context.
    """
    # Fetch full patient including source_note internally
    patient = _get(f"/patients/{patient_id}")
    note    = patient.get("source_note", "")
    if not note:
        return {"gaps": [], "summary": "No clinical note available for this patient"}
    result = _post("/caregaps", {"note": note})
    # Add patient context to result
    result["patient_id"]        = patient_id
    result["primary_diagnosis"] = patient.get("primary_diagnosis", "")
    result["risk_level"]        = patient.get("risk_level", "")
    return result


def get_population_gaps(risk_level: str = "HIGH", limit: int = 5) -> dict:
    """
    Get care gaps for multiple patients at once.
    Use this instead of checking patients one by one — much more efficient.
    Returns gaps for all patients in a single call.
    """
    params   = {"limit": limit}
    if risk_level:
        params["risk_level"] = risk_level.upper()
    patients = _get("/patients", params=params)

    if isinstance(patients, dict):
        patients = patients.get("patients", [])

    results = []
    for p in patients:
        patient_id = p.get("patient_id")
        if not patient_id:
            continue
        # Fetch full patient with source_note
        full_patient = _get(f"/patients/{patient_id}")
        note         = full_patient.get("source_note", "")
        if not note:
            continue
        gaps = _post("/caregaps", {"note": note})
        results.append({
            "patient_id":        patient_id,
            "primary_diagnosis": p.get("primary_diagnosis", ""),
            "risk_level":        p.get("risk_level", ""),
            "gap_count":         len(gaps.get("gaps", [])),
            "gaps":              gaps.get("gaps", []),
            "summary":           gaps.get("summary", "")
        })

    # Sort by gap count descending — most gaps first
    results.sort(key=lambda x: x["gap_count"], reverse=True)

    return {
        "population_gaps":  results,
        "total_patients":   len(results),
        "total_gaps":       sum(r["gap_count"] for r in results),
        "critical_gaps":    sum(
            1 for r in results
            for g in r["gaps"]
            if g.get("priority") == "HIGH"
        )
    }


def ask_clinical(question: str) -> dict:
    """POST /ask — RAG-based clinical Q&A over indexed notes."""
    return _post("/ask", {"query": question})


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name":        "list_patients",
        "description": "List patients from BigQuery. Filter by risk_level (HIGH, MEDIUM, LOW). Use this to get patient IDs before deeper analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type":        "string",
                    "enum":        ["HIGH", "MEDIUM", "LOW"],
                    "description": "Filter by risk level"
                },
                "limit": {
                    "type":        "integer",
                    "description": "Max patients to return (default 10)",
                    "default":     10
                }
            }
        }
    },
    {
        "name":        "get_patient",
        "description": "Get details for a single patient by ID. Does not include source note.",
        "input_schema": {
            "type":     "object",
            "properties": {
                "patient_id": {
                    "type":        "string",
                    "description": "8-character patient ID e.g. B2922341"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name":        "analyze_care_gaps",
        "description": "Analyze care gaps for a single patient by their ID. Fetches clinical note internally.",
        "input_schema": {
            "type":     "object",
            "properties": {
                "patient_id": {
                    "type":        "string",
                    "description": "Patient ID to analyze"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name":        "get_population_gaps",
        "description": "Get care gaps for multiple patients at once. PREFER THIS over calling analyze_care_gaps in a loop. Returns all gaps in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type":        "string",
                    "enum":        ["HIGH", "MEDIUM", "LOW"],
                    "description": "Filter by risk level (default HIGH)"
                },
                "limit": {
                    "type":        "integer",
                    "description": "Number of patients to analyse (default 5)",
                    "default":     5
                }
            }
        }
    },
    {
        "name":        "ask_clinical",
        "description": "Ask a clinical question answered via RAG over all indexed notes. Best for cross-patient questions.",
        "input_schema": {
            "type":     "object",
            "properties": {
                "question": {
                    "type":        "string",
                    "description": "The clinical question to answer"
                }
            },
            "required": ["question"]
        }
    }
]

TOOL_FNS = {
    "list_patients":      lambda inp: list_patients(**inp),
    "get_patient":        lambda inp: get_patient(**inp),
    "analyze_care_gaps":  lambda inp: analyze_care_gaps(**inp),
    "get_population_gaps":lambda inp: get_population_gaps(**inp),
    "ask_clinical":       lambda inp: ask_clinical(**inp),
}


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────
SYSTEM = """You are a clinical care-gap analyst. Use tools to answer the user's request.

Rules:
1. ALWAYS prefer get_population_gaps over calling analyze_care_gaps in a loop
2. Be concise — don't over-explain steps
3. When you have enough data, give the final answer immediately
4. Final answer format:
   - Summary (2-3 sentences)
   - Prioritised care gaps (HIGH first, with patient IDs)
   - Top 3 recommended actions

Be clinically precise. Cite patient IDs."""


# ─────────────────────────────────────────────
# ReAct agent loop
# ─────────────────────────────────────────────
def run_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    print(f"\nTask: {task}\n{'─'*60}")
    total_input_tokens  = 0
    total_output_tokens = 0

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Track token usage
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        print(f"\n[Iter {iteration+1}] tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

        tool_uses   = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # Print reasoning
        for block in text_blocks:
            if block.text.strip():
                print(f"\n[Thought] {block.text.strip()[:200]}")

        # Final answer
        if response.stop_reason == "end_turn" or not tool_uses:
            final = "\n".join(b.text for b in text_blocks if b.type == "text")
            print(f"\n{'─'*60}")
            print(f"[Total tokens] {total_input_tokens} input / {total_output_tokens} output")
            print(f"[Estimated cost] ~${(total_input_tokens * 0.00000025 + total_output_tokens * 0.00000125):.4f}")
            print(f"\n[Final Answer]\n{final}")
            return final

        # Execute tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for tool_use in tool_uses:
            print(f"\n[Action] {tool_use.name}({json.dumps(tool_use.input)})")
            try:
                result = TOOL_FNS[tool_use.name](tool_use.input)
                # Truncate output for display only
                output = json.dumps(result, indent=2)
                print(f"[Observation] {output[:300]}{'...' if len(output) > 300 else ''}")
            except Exception as exc:
                output = json.dumps({"error": str(exc)})
                print(f"[Error] {exc}")

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_use.id,
                "content":     output,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Agent reached maximum iterations."


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    default_task = (
        "What are the most common care gaps across HIGH risk patients? "
        "Give me the top 3 most critical gaps that need immediate attention."
    )
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_task
    run_agent(task)