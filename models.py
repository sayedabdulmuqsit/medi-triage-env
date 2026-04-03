from pydantic import BaseModel
from typing import Optional, List
from enum import IntEnum

class UrgencyLevel(IntEnum):
    SELF_CARE = 0
    NON_URGENT = 1
    URGENT = 2
    EMERGENCY = 3

class PatientObservation(BaseModel):
    patient_id: str
    age: int
    symptoms: List[str]
    duration_hours: int
    has_chronic_condition: bool
    past_visits: int
    language: str = "english"

class TriageAction(BaseModel):
    urgency_level: UrgencyLevel
    predicted_disease: str
    first_aid_suggestion: str
    alert_doctor: bool

class StepResult(BaseModel):
    observation: PatientObservation
    reward: float
    done: bool
    info: dict