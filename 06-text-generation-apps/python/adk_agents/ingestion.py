from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from adk_agents.models import ClinicalEvent, ClinicalPatientRecord, PatientDemographics

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_BASE_DIR = Path(__file__).resolve().parent
MAPPINGS_DIR = _BASE_DIR / "mappings"
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SCHEMA = {
    "patient_id": "Stable patient identifier such as patient_id, member_id, MRN, or person_id.",
    "demographics.first_name": "Patient first or given name.",
    "demographics.last_name": "Patient last or family name.",
    "demographics.birth_date": "Patient date of birth.",
    "demographics.gender": "Patient administrative gender or sex.",
    "demographics.race": "Patient race.",
    "demographics.ethnicity": "Patient ethnicity.",
    "demographics.address": "Patient street address.",
    "demographics.city": "Patient city.",
    "demographics.state": "Patient state.",
    "demographics.zip_code": "Patient ZIP or postal code.",
    "event.event_type": "Clinical event category such as condition, medication, encounter, observation, procedure, allergy, immunization.",
    "event.code": "Clinical code such as SNOMED, LOINC, RxNorm, ICD, CPT, or internal code.",
    "event.description": "Human readable event description.",
    "event.start_date": "Event start, onset, authored, or service date.",
    "event.end_date": "Event end, stop, resolved, or discharge date.",
    "event.encounter_id": "Encounter, visit, or claim context identifier.",
    "event.status": "Clinical or workflow status.",
    "event.value": "Observation result, medication amount, claim amount, or other measured value.",
    "event.unit": "Unit for event.value.",
}


COMMON_ALIASES = {
    "patient_id": ["patient", "patient_id", "patientid", "person", "person_id", "member", "member_id", "mrn", "id"],
    "demographics.first_name": ["first", "firstname", "first_name", "given", "given_name"],
    "demographics.last_name": ["last", "lastname", "last_name", "family", "family_name", "surname"],
    "demographics.birth_date": ["birthdate", "birth_date", "date_of_birth", "dob", "birth"],
    "demographics.gender": ["gender", "sex"],
    "demographics.race": ["race"],
    "demographics.ethnicity": ["ethnicity"],
    "demographics.address": ["address", "street", "street_address"],
    "demographics.city": ["city"],
    "demographics.state": ["state"],
    "demographics.zip_code": ["zip", "zipcode", "zip_code", "postal", "postal_code"],
    "event.event_type": ["type", "category", "event_type", "class"],
    "event.code": ["code", "snomed", "loinc", "rxnorm", "icd", "cpt", "diagnosis_code", "procedure_code"],
    "event.description": ["description", "desc", "name", "reason", "display", "text"],
    "event.start_date": ["start", "start_date", "date", "onset", "from", "service_date", "authoredon"],
    "event.end_date": ["stop", "end", "end_date", "resolved", "to", "discharge_date"],
    "event.encounter_id": ["encounter", "encounter_id", "visit", "visit_id", "claim", "claim_id"],
    "event.status": ["status", "clinical_status"],
    "event.value": ["value", "result", "result_value", "amount", "quantity"],
    "event.unit": ["unit", "units"],
}

# Helper mapping to route event types to correct lists in ClinicalPatientRecord
EVENT_TYPE_TO_FIELD = {
    "allergy": "allergies",
    "allergies": "allergies",
    "careplan": "careplans",
    "careplans": "careplans",
    "condition": "conditions",
    "conditions": "conditions",
    "device": "devices",
    "devices": "devices",
    "encounter": "encounters",
    "encounters": "encounters",
    "imaging_study": "imaging_studies",
    "imaging_studies": "imaging_studies",
    "immunization": "immunizations",
    "immunizations": "immunizations",
    "medication": "medications",
    "medications": "medications",
    "observation": "observations",
    "observations": "observations",
    "procedure": "procedures",
    "procedures": "procedures",
    "supply": "supplies",
    "supplies": "supplies",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def get_claude_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
    model = model or os.getenv("INGESTION_MAPPING_MODEL", "claude-haiku-4-5")
    response = claude.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += block.text

    return text.strip()


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@dataclass
class ColumnProfile:
    name: str
    normalized_name: str
    non_null_count: int
    null_count: int
    distinct_count: int
    sample_values: list[str]
    detected_types: list[str]
    semantic_hints: list[str]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().strip())


def clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_date(value: Any) -> date | datetime | None:
    text = clean_value(value)
    if not text:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(text[:19], fmt)
            return parsed if "H" in fmt else parsed.date()
        except ValueError:
            continue
    return None


def detect_value_type(value: str) -> str:
    if parse_date(value):
        return "date"
    if re.fullmatch(r"-?\d+", value):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", value):
        return "decimal"
    if value.lower() in {"true", "false", "yes", "no", "y", "n"}:
        return "boolean"
    return "text"


def semantic_hints(values: list[str]) -> list[str]:
    joined = " ".join(values[:30]).lower()
    hints: list[str] = []

    if any(re.fullmatch(r"\d{4}-\d", v) for v in values):
        hints.append("loinc_like_code")
    if any(re.fullmatch(r"\d{5,18}", v) for v in values):
        hints.append("numeric_clinical_code_or_identifier")
    if any(v.lower() in {"male", "female", "m", "f"} for v in values):
        hints.append("gender_like")
    if any(term in joined for term in ["diabetes", "hypertension", "asthma", "kidney", "failure"]):
        hints.append("condition_description_like")
    if any(term in joined for term in ["mg", "tablet", "capsule", "injection", "rxnorm"]):
        hints.append("medication_like")
    if any(term in joined for term in ["mmhg", "kg", "cm", "mg/dl", "%"]):
        hints.append("observation_value_or_unit_like")

    return hints


def profile_csv_file(csv_path: str | Path, sample_size: int = 50) -> dict[str, Any]:
    csv_path = Path(csv_path)
    sampled_rows: list[dict[str, str]] = []
    column_values: dict[str, list[str]] = defaultdict(list)
    null_counts: Counter[str] = Counter()

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        for row_index, row in enumerate(reader):
            if row_index < sample_size:
                sampled_rows.append(row)

            for column in columns:
                value = clean_value(row.get(column))
                if value is None:
                    null_counts[column] += 1
                elif len(column_values[column]) < sample_size:
                    column_values[column].append(value)

            if row_index + 1 >= sample_size:
                break

    profiles = []
    for column in columns:
        values = column_values[column]
        type_counts = Counter(detect_value_type(v) for v in values)
        profiles.append(
            ColumnProfile(
                name=column,
                normalized_name=normalize_name(column),
                non_null_count=len(values),
                null_count=null_counts[column],
                distinct_count=len(set(values)),
                sample_values=values[:10],
                detected_types=[name for name, _ in type_counts.most_common()],
                semantic_hints=semantic_hints(values),
            ).__dict__
        )

    return {
        "file_name": csv_path.name,
        "path": str(csv_path),
        "sample_size": len(sampled_rows),
        "columns": profiles,
        "target_schema": TARGET_SCHEMA,
    }


def profile_clinical_folder(folder_path: str | Path, sample_size: int = 50) -> dict[str, Any]:
    folder_path = Path(folder_path)
    return {
        "folder": str(folder_path),
        "files": [
            profile_csv_file(csv_path, sample_size=sample_size)
            for csv_path in sorted(folder_path.glob("*.csv"))
        ],
    }


def fuzzy_score(source_name: str, aliases: list[str]) -> float:
    source = normalize_name(source_name)
    return max(SequenceMatcher(None, source, normalize_name(alias)).ratio() for alias in aliases)


def propose_mapping_with_rules(profile: dict[str, Any]) -> dict[str, Any]:
    mappings = []

    for column in profile["columns"]:
        best_target = None
        best_score = 0.0

        for target, aliases in COMMON_ALIASES.items():
            score = fuzzy_score(column["name"], aliases)
            if score > best_score:
                best_target = target
                best_score = score

        if best_target and best_score >= 0.82:
            confidence = round(best_score, 2)
            reason = "Column name closely matches known aliases."
        else:
            best_target = None
            confidence = 0.0
            reason = "No reliable rule-based match."

        mappings.append(
            {
                "source_column": column["name"],
                "target_field": best_target,
                "confidence": confidence,
                "requires_review": confidence < 0.9,
                "reason": reason,
                "sample_values": column["sample_values"][:5],
                "semantic_hints": column["semantic_hints"],
            }
        )

    return {
        "file_name": profile["file_name"],
        "mapping_source": "rules",
        "mappings": mappings,
    }


def propose_mapping_with_ai(profile: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """
    Uses Claude to propose source-to-target mapping from both column names and sample values.

    Requires ANTHROPIC_API_KEY. Falls back to rule mapping if the key is unavailable.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        proposal = propose_mapping_with_rules(profile)
        proposal["mapping_source"] = "rules_no_anthropic_key"
        return proposal

    system_prompt = """You are a healthcare data ingestion mapping assistant.
You map messy source CSV columns into a canonical clinical data model.
Always respond with valid JSON only. No markdown. No explanation outside JSON."""

    user_prompt = f"""
You are a healthcare data ingestion mapping assistant.

Map each source CSV column to the best target field from TARGET_SCHEMA.
Use both column names and sample values. Do not guess when ambiguous.

Return only valid JSON with this shape:
{{
  "file_name": "...",
  "mapping_source": "ai",
  "mappings": [
    {{
      "source_column": "...",
      "target_field": "patient_id | demographics.first_name | event.code | null",
      "confidence": 0.0,
      "requires_review": true,
      "reason": "short explanation",
      "sample_values": ["..."],
      "semantic_hints": ["..."]
    }}
  ],
  "warnings": ["..."]
}}

Rules:
- If a column is ambiguous, use null for target_field.
- Set requires_review=true for confidence below 0.90.
- Use sample values to distinguish patient identifiers, diagnosis codes, medication codes, observation codes, dates, names, and values.
- Never invent target fields outside TARGET_SCHEMA.
- Prefer null over a confident-looking but unsafe mapping.

TARGET_SCHEMA:
{json.dumps(TARGET_SCHEMA, indent=2)}

CSV_PROFILE:
{json.dumps(profile, indent=2)}
"""

    try:
        text = get_claude_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=4096,
            temperature=0.0,
            model=model,
        )
    except Exception as exc:
        proposal = propose_mapping_with_rules(profile)
        proposal["mapping_source"] = "rules_claude_error"
        proposal["claude_error"] = str(exc)
        return proposal

    text = clean_json(text)

    try:
        proposal = json.loads(text)
    except json.JSONDecodeError:
        proposal = propose_mapping_with_rules(profile)
        proposal["mapping_source"] = "rules_ai_invalid_json"
        proposal["ai_raw_response"] = text[:2000]
        return proposal

    proposal["mapping_source"] = proposal.get("mapping_source") or "ai"
    return proposal


def mapping_path(file_name: str, mappings_dir: str | Path | None = None) -> Path:
    mappings_dir = Path(mappings_dir) if mappings_dir else MAPPINGS_DIR
    mappings_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file_name).name.replace(".csv", "")
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", safe_name)

    return mappings_dir / f"{safe_name}.mapping.json"


def save_mapping(
    mapping: dict[str, Any],
    output_path: str | Path | None = None,
    mappings_dir: str | Path | None = None,
) -> Path:
    if output_path is None:
        output_path = mapping_path(mapping.get("file_name", "mapping"), mappings_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    return output_path


def load_mapping(mapping_file: str | Path) -> dict[str, Any]:
    mapping_file = Path(mapping_file)

    if not mapping_file.exists():
        mapping_file = MAPPINGS_DIR / mapping_file

    return json.loads(mapping_file.read_text(encoding="utf-8"))


def mapping_preview_markdown(mapping: dict[str, Any]) -> str:
    lines = [
        f"# Mapping Preview: {mapping.get('file_name', 'unknown')}",
        "",
        "| Source Column | Target Field | Confidence | Review? | Sample Values | Reason |",
        "|---|---|---:|---|---|---|",
    ]

    for item in mapping.get("mappings", []):
        samples = ", ".join(str(v) for v in item.get("sample_values", [])[:3])
        review = "YES" if item.get("requires_review") else "NO"
        lines.append(
            "| {source} | {target} | {confidence} | {review} | {samples} | {reason} |".format(
                source=item.get("source_column", ""),
                target=item.get("target_field") or "UNMAPPED",
                confidence=item.get("confidence", 0),
                review=review,
                samples=samples.replace("|", "/"),
                reason=str(item.get("reason", "")).replace("|", "/"),
            )
        )

    warnings = mapping.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


def ensure_approved(mapping: dict[str, Any]) -> None:
    if not mapping.get("approved"):
        raise ValueError(
            "Mapping is not approved. Review the preview, edit the JSON, then set "
            '"approved": true before ingestion.'
        )


def model_fields(model_cls: type) -> set[str]:
    return set(getattr(model_cls, "model_fields", {}).keys())


def make_patient_record(patient_id: str) -> ClinicalPatientRecord:
    # ClinicalPatientRecord requires a PatientDemographics instance in the 'patient' field
    return ClinicalPatientRecord(
        patient=PatientDemographics(patient_id=patient_id)
    )


def apply_demographic(record: ClinicalPatientRecord, field: str, value: Any) -> None:
    demographics = record.patient
    demo_field = field.split(".", 1)[1]

    # Map target schema mapping fields to PatientDemographics fields
    field_mapping = {
        "birth_date": "birthdate",
        "zip_code": "zip"
    }
    target_attr = field_mapping.get(demo_field, demo_field)

    if target_attr == "birthdate":
        value = parse_date(value)

    # Normalize gender to Literal["M", "F", "UNKNOWN"]
    elif target_attr == "gender":
        if value:
            norm_val = str(value).strip().upper()
            if norm_val in {"M", "F"}:
                value = norm_val
            elif norm_val in {"MALE"}:
                value = "M"
            elif norm_val in {"FEMALE"}:
                value = "F"
            else:
                value = "UNKNOWN"
        else:
            value = "UNKNOWN"

    if hasattr(demographics, target_attr):
        setattr(demographics, target_attr, value)


def make_event(values: dict[str, Any], raw: dict[str, Any], file_name: str) -> ClinicalEvent:
    event_values = {
        "source_file": file_name,
        "metadata": raw,
    }

    # Map target schema event fields to ClinicalEvent fields
    field_mapping = {
        "start_date": "start",
        "end_date": "stop",
    }

    for field, value in values.items():
        event_field = field.split(".", 1)[1]
        target_field = field_mapping.get(event_field, event_field)

        if target_field in {"start", "stop"}:
            value = parse_date(value)

        event_values[target_field] = value

    # Ensure required description is present
    if "description" not in event_values or event_values["description"] is None:
        event_values["description"] = event_values.get("code") or "Unknown Event"

    allowed = model_fields(ClinicalEvent)
    return ClinicalEvent(**{k: v for k, v in event_values.items() if k in allowed and v is not None})


def attach_event(record: ClinicalPatientRecord, event: ClinicalEvent, event_type: str) -> None:
    # Route the event to the correct list field in the record based on category/file stem
    key = str(event_type).strip().lower()
    field_name = EVENT_TYPE_TO_FIELD.get(key, "observations")

    if hasattr(record, field_name):
        current_list = getattr(record, field_name)
        current_list.append(event)


def ingest_csv_with_mapping(csv_path: str | Path, mapping: dict[str, Any]) -> list[ClinicalPatientRecord]:
    ensure_approved(mapping)
    csv_path = Path(csv_path)

    mapped_fields = {
        item["source_column"]: item["target_field"]
        for item in mapping.get("mappings", [])
        if item.get("target_field")
    }

    records: dict[str, ClinicalPatientRecord] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            patient_id = None
            event_values: dict[str, Any] = {}

            for source_column, target_field in mapped_fields.items():
                value = clean_value(row.get(source_column))
                if value is None:
                    continue

                if target_field == "patient_id":
                    patient_id = value
                elif target_field.startswith("event."):
                    event_values[target_field] = value

            if not patient_id:
                patient_id = f"{csv_path.stem}_row_{row_number}"

            record = records.setdefault(patient_id, make_patient_record(patient_id))

            for source_column, target_field in mapped_fields.items():
                value = clean_value(row.get(source_column))
                if value is None:
                    continue
                if target_field.startswith("demographics."):
                    apply_demographic(record, target_field, value)

            if event_values:
                event_type = event_values.get("event.event_type") or csv_path.stem
                event = make_event(event_values, dict(row), csv_path.name)
                attach_event(record, event, event_type)

    return list(records.values())


def create_relational_tables(project_id: str, dataset_id: str) -> None:
    """Initializes the relational BigQuery tables with correct schemas if they do not exist."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    print(f"Initializing BigQuery tables in dataset: {dataset_id}...")

    # 1. Patients demographics table schema
    patients_table_ref = f"{project_id}.{dataset_id}.patients"
    patients_schema = [
        bigquery.SchemaField("patient_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("first_name", "STRING"),
        bigquery.SchemaField("last_name", "STRING"),
        bigquery.SchemaField("birthdate", "DATE"),
        bigquery.SchemaField("deathdate", "DATE"),
        bigquery.SchemaField("gender", "STRING"),
        bigquery.SchemaField("race", "STRING"),
        bigquery.SchemaField("ethnicity", "STRING"),
        bigquery.SchemaField("address", "STRING"),
        bigquery.SchemaField("city", "STRING"),
        bigquery.SchemaField("state", "STRING"),
        bigquery.SchemaField("zip", "STRING"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]
    patients_table = bigquery.Table(patients_table_ref, schema=patients_schema)
    client.create_table(patients_table, exists_ok=True)
    print(f"✅ Demographics table is ready: {patients_table_ref}")

    # 2. Conditions event table schema
    conditions_table_ref = f"{project_id}.{dataset_id}.conditions"
    conditions_schema = [
        bigquery.SchemaField("patient_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("source_file", "STRING"),
        bigquery.SchemaField("source_id", "STRING"),
        bigquery.SchemaField("encounter_id", "STRING"),
        bigquery.SchemaField("code", "STRING"),
        bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("start", "DATE"),
        bigquery.SchemaField("stop", "DATE"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("value", "STRING"),
        bigquery.SchemaField("unit", "STRING"),
        bigquery.SchemaField("metadata", "STRING"),  # Serialized raw JSON string
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]
    conditions_table = bigquery.Table(conditions_table_ref, schema=conditions_schema)
    client.create_table(conditions_table, exists_ok=True)
    print(f"✅ Conditions table is ready: {conditions_table_ref}")


def write_demographics_to_bigquery(
    records: list[ClinicalPatientRecord],
    project_id: str,
    dataset_id: str,
    table_id: str = "patients",
) -> dict[str, Any]:
    """
    Writes flat patient demographics to the `patients` table in BigQuery.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    rows = []
    for record in records:
        payload = record.patient.model_dump(mode="json")
        payload["ingested_at"] = datetime.utcnow().isoformat()
        rows.append(payload)

    if not rows:
        return {"table": table_ref, "rows_attempted": 0, "errors": []}

    errors = client.insert_rows_json(table_ref, rows)
    return {
        "table": table_ref,
        "rows_attempted": len(rows),
        "errors": errors,
    }


def write_events_to_bigquery(
    records: list[ClinicalPatientRecord],
    event_type: str,
    project_id: str,
    dataset_id: str,
    table_id: str | None = None,
) -> dict[str, Any]:
    """
    Extracts events of a specific category (e.g. 'conditions') and inserts them 
    into their corresponding separate table in BigQuery, linked by patient_id.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    
    # Map event_type (like 'condition' or 'medications') to correct record field list
    key = str(event_type).strip().lower()
    field_name = EVENT_TYPE_TO_FIELD.get(key, "observations")
    
    # Default BQ table name to the field name (e.g., 'conditions')
    table_id = table_id or field_name
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    rows = []
    for record in records:
        patient_id = record.patient.patient_id
        if hasattr(record, field_name):
            events = getattr(record, field_name)
            for event in events:
                payload = event.model_dump(mode="json")
                # Link this event row back to the patient ID
                payload["patient_id"] = patient_id
                payload["ingested_at"] = datetime.utcnow().isoformat()
                
                # Convert metadata dict to JSON string if table column is STRING type
                if "metadata" in payload and isinstance(payload["metadata"], dict):
                    payload["metadata"] = json.dumps(payload["metadata"])
                    
                rows.append(payload)

    if not rows:
        return {
            "table": table_ref,
            "rows_attempted": 0,
            "errors": [],
            "message": f"No events found in records for category: {field_name}"
        }

    errors = client.insert_rows_json(table_ref, rows)
    return {
        "table": table_ref,
        "rows_attempted": len(rows),
        "errors": errors,
    }

def detect_csv_category(file_path: str) -> dict:
    """
    Analyzes the column headers of a CSV file to detect its clinical category.
    """
    from pathlib import Path
    import csv

    path = Path(file_path)
    if not path.exists():
        return {"error": "File not found", "file_path": file_path}

    # Read headers
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = [h.lower().strip() for h in next(reader)]

    # Simple heuristic classification based on common unique columns
    if any(h in headers for h in ["ssn", "first", "last", "maiden", "birthplace"]):
        category = "patients"
    elif any(h in headers for h in ["reasoncode", "reasondescription", "encounter"]):
        # Narrow down by file name or other headers
        name = path.stem.lower()
        if "condition" in name:
            category = "conditions"
        elif "medication" in name:
            category = "medications"
        elif "procedure" in name:
            category = "procedures"
        else:
            category = "events"
    else:
        category = "events"

    return {
        "file_name": path.name,
        "detected_category": category,
        "headers": headers[:10]  # Return first 10 headers for context
    }

def propose_and_preview_mapping(file_path: str, category: str) -> dict:
    """
    Generates mapping using AI and returns the schema proposal and a Markdown preview table.
    """
    # 1. Profile the CSV
    profile = profile_csv_file(file_path)
    
    # 2. Run the AI mapper (using your existing function)
    mapping_proposal = propose_mapping_with_ai(profile)
    
    # 3. Generate Markdown preview using your existing helper
    preview_md = mapping_preview_markdown(mapping_proposal)
    
    # Save mapping draft to the mappings directory
    draft_path = save_mapping(mapping_proposal)
    
    return {
        "mapping_path": str(draft_path),
        "preview_markdown": preview_md,
        "proposal": mapping_proposal
    }

def prompt_user_approval(preview_markdown: str) -> bool:
    """
    Displays the mapping preview table and prompts the user in the CLI for approval.
    """
    print("\n=== PROPOSED SCHEMA MAPPING PREVIEW ===")
    print(preview_markdown)
    print("=======================================")
    
    while True:
        response = input("\nDo you approve this mapping for ingestion? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        print("Please enter 'yes' or 'no'.")


def ingest_csv_data(file_path: str, category: str, approved: bool) -> dict:
    """
    If approved, runs ingestion and streams rows into BigQuery.
    """
    if not approved:
        return {"status": "cancelled", "message": "User did not approve the mapping."}
        
    # 1. Load the saved mapping (marked approved)
    mapping = load_mapping(mapping_path(file_path))
    mapping["approved"] = True
    save_mapping(mapping)
    
    # 2. Parse and map to canonical Pydantic records
    records = ingest_csv_with_mapping(file_path, mapping)
    
    # 3. Write to BigQuery (e.g. patients or events)
    project_id = "healthcare-ai-manoj"
    dataset_id = "healthcare_ai"
    
    if category == "patients":
        # Create table if missing (using Client API)
        result = write_demographics_to_bigquery(records, project_id, dataset_id)
    else:
        result = write_events_to_bigquery(records, category, project_id, dataset_id)
        
    return {
        "status": "success",
        "rows_ingested": result.get("rows_attempted", 0),
        "table": result.get("table"),
        "errors": result.get("errors")
    }

def run_bigquery_query(sql: str) -> dict:
    """
    Runs a SQL query against the healthcare BigQuery dataset and returns results.
    Use this to answer analytical questions about patients and clinical events.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project="healthcare-ai-manoj")
    try:
        query_job = client.query(sql)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return {
            "status": "success",
            "row_count": len(rows),
            "results": rows[:50]  # cap at 50 rows for safety
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}