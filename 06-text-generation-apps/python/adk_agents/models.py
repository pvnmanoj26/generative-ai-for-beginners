from __future__ import annotations

from datetime import date as Date, datetime as DateTime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
Gender = Literal["M", "F", "UNKNOWN"]


class CsvRow(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class PatientRow(CsvRow):
    id: str = Field(alias="Id")
    birthdate: Date | None = Field(default=None, alias="BIRTHDATE")
    deathdate: Date | None = Field(default=None, alias="DEATHDATE")
    ssn: str | None = Field(default=None, alias="SSN")
    drivers: str | None = Field(default=None, alias="DRIVERS")
    passport: str | None = Field(default=None, alias="PASSPORT")
    prefix: str | None = Field(default=None, alias="PREFIX")
    first: str | None = Field(default=None, alias="FIRST")
    last: str | None = Field(default=None, alias="LAST")
    suffix: str | None = Field(default=None, alias="SUFFIX")
    maiden: str | None = Field(default=None, alias="MAIDEN")
    marital: str | None = Field(default=None, alias="MARITAL")
    race: str | None = Field(default=None, alias="RACE")
    ethnicity: str | None = Field(default=None, alias="ETHNICITY")
    gender: str | None = Field(default=None, alias="GENDER")
    birthplace: str | None = Field(default=None, alias="BIRTHPLACE")
    address: str | None = Field(default=None, alias="ADDRESS")
    city: str | None = Field(default=None, alias="CITY")
    state: str | None = Field(default=None, alias="STATE")
    county: str | None = Field(default=None, alias="COUNTY")
    zip: str | None = Field(default=None, alias="ZIP")
    lat: float | None = Field(default=None, alias="LAT")
    lon: float | None = Field(default=None, alias="LON")
    healthcare_expenses: Decimal | None = Field(default=None, alias="HEALTHCARE_EXPENSES")
    healthcare_coverage: Decimal | None = Field(default=None, alias="HEALTHCARE_COVERAGE")
    income: Decimal | None = Field(default=None, alias="INCOME")


class EncounterRow(CsvRow):
    id: str = Field(alias="Id")
    start: DateTime | None = Field(default=None, alias="START")
    stop: DateTime | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    organization: str | None = Field(default=None, alias="ORGANIZATION")
    provider: str | None = Field(default=None, alias="PROVIDER")
    payer: str | None = Field(default=None, alias="PAYER")
    encounter_class: str | None = Field(default=None, alias="ENCOUNTERCLASS")
    code: str | None = Field(default=None, alias="CODE")
    description: str | None = Field(default=None, alias="DESCRIPTION")
    base_encounter_cost: Decimal | None = Field(default=None, alias="BASE_ENCOUNTER_COST")
    total_claim_cost: Decimal | None = Field(default=None, alias="TOTAL_CLAIM_COST")
    payer_coverage: Decimal | None = Field(default=None, alias="PAYER_COVERAGE")
    reason_code: str | None = Field(default=None, alias="REASONCODE")
    reason_description: str | None = Field(default=None, alias="REASONDESCRIPTION")


class ConditionRow(CsvRow):
    start: Date | None = Field(default=None, alias="START")
    stop: Date | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")


class ObservationRow(CsvRow):
    date: DateTime | None = Field(default=None, alias="DATE")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    category: str | None = Field(default=None, alias="CATEGORY")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    value: str | None = Field(default=None, alias="VALUE")
    units: str | None = Field(default=None, alias="UNITS")
    type: str | None = Field(default=None, alias="TYPE")


class MedicationRow(CsvRow):
    start: DateTime | None = Field(default=None, alias="START")
    stop: DateTime | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    payer: str | None = Field(default=None, alias="PAYER")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    base_cost: Decimal | None = Field(default=None, alias="BASE_COST")
    payer_coverage: Decimal | None = Field(default=None, alias="PAYER_COVERAGE")
    dispenses: int | None = Field(default=None, alias="DISPENSES")
    total_cost: Decimal | None = Field(default=None, alias="TOTALCOST")
    reason_code: str | None = Field(default=None, alias="REASONCODE")
    reason_description: str | None = Field(default=None, alias="REASONDESCRIPTION")


class AllergyRow(CsvRow):
    start: Date | None = Field(default=None, alias="START")
    stop: Date | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")


class CarePlanRow(CsvRow):
    id: str = Field(alias="Id")
    start: Date | None = Field(default=None, alias="START")
    stop: Date | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    reason_code: str | None = Field(default=None, alias="REASONCODE")
    reason_description: str | None = Field(default=None, alias="REASONDESCRIPTION")


class ProcedureRow(CsvRow):
    start: DateTime | None = Field(default=None, alias="START")
    stop: DateTime | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    base_cost: Decimal | None = Field(default=None, alias="BASE_COST")
    reason_code: str | None = Field(default=None, alias="REASONCODE")
    reason_description: str | None = Field(default=None, alias="REASONDESCRIPTION")


class ImmunizationRow(CsvRow):
    date: DateTime | None = Field(default=None, alias="DATE")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    base_cost: Decimal | None = Field(default=None, alias="BASE_COST")


class DeviceRow(CsvRow):
    start: DateTime | None = Field(default=None, alias="START")
    stop: DateTime | None = Field(default=None, alias="STOP")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    udi: str | None = Field(default=None, alias="UDI")


class ImagingStudyRow(CsvRow):
    id: str = Field(alias="Id")
    date: DateTime | None = Field(default=None, alias="DATE")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    series_uid: str | None = Field(default=None, alias="SERIES_UID")
    bodysite_code: str | None = Field(default=None, alias="BODYSITE_CODE")
    bodysite_description: str | None = Field(default=None, alias="BODYSITE_DESCRIPTION")
    modality_code: str | None = Field(default=None, alias="MODALITY_CODE")
    modality_description: str | None = Field(default=None, alias="MODALITY_DESCRIPTION")
    instance_uid: str | None = Field(default=None, alias="INSTANCE_UID")
    sop_code: str | None = Field(default=None, alias="SOP_CODE")
    sop_description: str | None = Field(default=None, alias="SOP_DESCRIPTION")


class OrganizationRow(CsvRow):
    id: str = Field(alias="Id")
    name: str = Field(alias="NAME")
    address: str | None = Field(default=None, alias="ADDRESS")
    city: str | None = Field(default=None, alias="CITY")
    state: str | None = Field(default=None, alias="STATE")
    zip: str | None = Field(default=None, alias="ZIP")
    lat: float | None = Field(default=None, alias="LAT")
    lon: float | None = Field(default=None, alias="LON")
    phone: str | None = Field(default=None, alias="PHONE")
    revenue: Decimal | None = Field(default=None, alias="REVENUE")
    utilization: int | None = Field(default=None, alias="UTILIZATION")


class ProviderRow(CsvRow):
    id: str = Field(alias="Id")
    organization: str | None = Field(default=None, alias="ORGANIZATION")
    name: str = Field(alias="NAME")
    gender: str | None = Field(default=None, alias="GENDER")
    speciality: str | None = Field(default=None, alias="SPECIALITY")
    address: str | None = Field(default=None, alias="ADDRESS")
    city: str | None = Field(default=None, alias="CITY")
    state: str | None = Field(default=None, alias="STATE")
    zip: str | None = Field(default=None, alias="ZIP")
    lat: float | None = Field(default=None, alias="LAT")
    lon: float | None = Field(default=None, alias="LON")
    utilization: int | None = Field(default=None, alias="UTILIZATION")


class PayerRow(CsvRow):
    id: str = Field(alias="Id")
    name: str = Field(alias="NAME")
    address: str | None = Field(default=None, alias="ADDRESS")
    city: str | None = Field(default=None, alias="CITY")
    state_headquartered: str | None = Field(default=None, alias="STATE_HEADQUARTERED")
    zip: str | None = Field(default=None, alias="ZIP")
    phone: str | None = Field(default=None, alias="PHONE")
    amount_covered: Decimal | None = Field(default=None, alias="AMOUNT_COVERED")
    amount_uncovered: Decimal | None = Field(default=None, alias="AMOUNT_UNCOVERED")
    revenue: Decimal | None = Field(default=None, alias="REVENUE")
    covered_encounters: int | None = Field(default=None, alias="COVERED_ENCOUNTERS")
    uncovered_encounters: int | None = Field(default=None, alias="UNCOVERED_ENCOUNTERS")
    covered_medications: int | None = Field(default=None, alias="COVERED_MEDICATIONS")
    uncovered_medications: int | None = Field(default=None, alias="UNCOVERED_MEDICATIONS")
    covered_procedures: int | None = Field(default=None, alias="COVERED_PROCEDURES")
    uncovered_procedures: int | None = Field(default=None, alias="UNCOVERED_PROCEDURES")
    covered_immunizations: int | None = Field(default=None, alias="COVERED_IMMUNIZATIONS")
    uncovered_immunizations: int | None = Field(default=None, alias="UNCOVERED_IMMUNIZATIONS")
    unique_customers: int | None = Field(default=None, alias="UNIQUE_CUSTOMERS")
    qols_avg: float | None = Field(default=None, alias="QOLS_AVG")
    member_months: int | None = Field(default=None, alias="MEMBER_MONTHS")


class ClaimRow(CsvRow):
    id: str = Field(alias="Id")
    patient: str = Field(alias="PATIENTID")
    provider: str | None = Field(default=None, alias="PROVIDERID")
    primary_patient_insurance_id: str | None = Field(default=None, alias="PRIMARYPATIENTINSURANCEID")
    secondary_patient_insurance_id: str | None = Field(default=None, alias="SECONDARYPATIENTINSURANCEID")
    department_id: str | None = Field(default=None, alias="DEPARTMENTID")
    patient_department_id: str | None = Field(default=None, alias="PATIENTDEPARTMENTID")
    diagnosis_1: str | None = Field(default=None, alias="DIAGNOSIS1")
    diagnosis_2: str | None = Field(default=None, alias="DIAGNOSIS2")
    diagnosis_3: str | None = Field(default=None, alias="DIAGNOSIS3")
    diagnosis_4: str | None = Field(default=None, alias="DIAGNOSIS4")
    diagnosis_5: str | None = Field(default=None, alias="DIAGNOSIS5")
    diagnosis_6: str | None = Field(default=None, alias="DIAGNOSIS6")
    diagnosis_7: str | None = Field(default=None, alias="DIAGNOSIS7")
    diagnosis_8: str | None = Field(default=None, alias="DIAGNOSIS8")
    referring_provider_id: str | None = Field(default=None, alias="REFERRINGPROVIDERID")
    appointment_id: str | None = Field(default=None, alias="APPOINTMENTID")
    current_illness_date: Date | None = Field(default=None, alias="CURRENTILLNESSDATE")
    service_date: Date | None = Field(default=None, alias="SERVICEDATE")
    supervising_provider_id: str | None = Field(default=None, alias="SUPERVISINGPROVIDERID")
    status_1: str | None = Field(default=None, alias="STATUS1")
    status_2: str | None = Field(default=None, alias="STATUS2")
    status_p: str | None = Field(default=None, alias="STATUSP")
    outstanding_1: Decimal | None = Field(default=None, alias="OUTSTANDING1")
    outstanding_2: Decimal | None = Field(default=None, alias="OUTSTANDING2")
    outstanding_p: Decimal | None = Field(default=None, alias="OUTSTANDINGP")
    last_billed_date_1: Date | None = Field(default=None, alias="LASTBILLEDDATE1")
    last_billed_date_2: Date | None = Field(default=None, alias="LASTBILLEDDATE2")
    last_billed_date_p: Date | None = Field(default=None, alias="LASTBILLEDDATEP")
    healthcare_claim_type_id_1: str | None = Field(default=None, alias="HEALTHCARECLAIMTYPEID1")
    healthcare_claim_type_id_2: str | None = Field(default=None, alias="HEALTHCARECLAIMTYPEID2")


class ClaimTransactionRow(CsvRow):
    id: str = Field(alias="ID")
    claim_id: str = Field(alias="CLAIMID")
    charge_id: str | None = Field(default=None, alias="CHARGEID")
    patient_id: str = Field(alias="PATIENTID")
    type: str | None = Field(default=None, alias="TYPE")
    amount: Decimal | None = Field(default=None, alias="AMOUNT")
    method: str | None = Field(default=None, alias="METHOD")
    from_date: Date | None = Field(default=None, alias="FROMDATE")
    to_date: Date | None = Field(default=None, alias="TODATE")
    place_of_service: str | None = Field(default=None, alias="PLACEOFSERVICE")
    procedure_code: str | None = Field(default=None, alias="PROCEDURECODE")
    modifier_1: str | None = Field(default=None, alias="MODIFIER1")
    modifier_2: str | None = Field(default=None, alias="MODIFIER2")
    diagnosis_ref_1: str | None = Field(default=None, alias="DIAGNOSISREF1")
    diagnosis_ref_2: str | None = Field(default=None, alias="DIAGNOSISREF2")
    units: Decimal | None = Field(default=None, alias="UNITS")
    department_id: str | None = Field(default=None, alias="DEPARTMENTID")
    transfer_type: str | None = Field(default=None, alias="TRANSFERTYPE")
    payments: Decimal | None = Field(default=None, alias="PAYMENTS")
    adjustments: Decimal | None = Field(default=None, alias="ADJUSTMENTS")
    transfers: Decimal | None = Field(default=None, alias="TRANSFERS")
    outstanding: Decimal | None = Field(default=None, alias="OUTSTANDING")
    appointment_id: str | None = Field(default=None, alias="APPOINTMENTID")
    line_note: str | None = Field(default=None, alias="LINENOTE")
    patient_insurance_id: str | None = Field(default=None, alias="PATIENTINSURANCEID")
    fee_schedule_id: str | None = Field(default=None, alias="FEESCHEDULEID")
    provider_id: str | None = Field(default=None, alias="PROVIDERID")
    supervising_provider_id: str | None = Field(default=None, alias="SUPERVISINGPROVIDERID")


class PayerTransitionRow(CsvRow):
    patient: str = Field(alias="PATIENT")
    member_id: str | None = Field(default=None, alias="MEMBERID")
    start_year: int | None = Field(default=None, alias="START_YEAR")
    end_year: int | None = Field(default=None, alias="END_YEAR")
    payer: str | None = Field(default=None, alias="PAYER")
    ownership: str | None = Field(default=None, alias="OWNERSHIP")


class SupplyRow(CsvRow):
    date: Date | None = Field(default=None, alias="DATE")
    patient: str = Field(alias="PATIENT")
    encounter: str | None = Field(default=None, alias="ENCOUNTER")
    code: str | None = Field(default=None, alias="CODE")
    description: str = Field(alias="DESCRIPTION")
    quantity: int | None = Field(default=None, alias="QUANTITY")


class ClinicalEvent(BaseModel):
    source_file: str
    source_id: str | None = None
    encounter_id: str | None = None
    code: str | None = None
    description: str
    start: Date | DateTime | None = None
    stop: Date | DateTime | None = None
    status: str | None = None
    value: str | None = None
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatientDemographics(BaseModel):
    patient_id: str
    first_name: str | None = None
    last_name: str | None = None
    birthdate: Date | None = None
    deathdate: Date | None = None
    gender: Gender = "UNKNOWN"
    race: str | None = None
    ethnicity: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None


class ClinicalPatientRecord(BaseModel):
    patient: PatientDemographics
    source: Literal["synthea_csv"] = "synthea_csv"
    risk_level: RiskLevel = "UNKNOWN"
    primary_diagnosis: str | None = None
    clinical_note: str | None = None
    allergies: list[ClinicalEvent] = Field(default_factory=list)
    careplans: list[ClinicalEvent] = Field(default_factory=list)
    claims: list[ClaimRow] = Field(default_factory=list)
    claim_transactions: list[ClaimTransactionRow] = Field(default_factory=list)
    conditions: list[ClinicalEvent] = Field(default_factory=list)
    devices: list[ClinicalEvent] = Field(default_factory=list)
    encounters: list[ClinicalEvent] = Field(default_factory=list)
    imaging_studies: list[ClinicalEvent] = Field(default_factory=list)
    immunizations: list[ClinicalEvent] = Field(default_factory=list)
    medications: list[ClinicalEvent] = Field(default_factory=list)
    observations: list[ClinicalEvent] = Field(default_factory=list)
    payer_transitions: list[PayerTransitionRow] = Field(default_factory=list)
    procedures: list[ClinicalEvent] = Field(default_factory=list)
    supplies: list[ClinicalEvent] = Field(default_factory=list)
    events: list[ClinicalEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)