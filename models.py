from pydantic import BaseModel, Field,  validator
from typing import Optional, List
from enum import IntEnum


class UrgencyLevel(IntEnum):
    SELF_CARE = 0
    NON_URGENT = 1
    URGENT = 2
    EMERGENCY = 3


class VitalSigns(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in BPM (normal: 60-100)")
    systolic_bp: int = Field(..., description="Systolic blood pressure in mmHg (normal: 90-120)")
    diastolic_bp: int = Field(..., description="Diastolic blood pressure in mmHg (normal: 60-80)")
    oxygen_saturation: float = Field(..., description="SpO2 percentage (normal: 95-100)")
    temperature: float = Field(..., description="Body temperature in Celsius (normal: 36.1-37.2)")
    respiratory_rate: int = Field(..., description="Breaths per minute (normal: 12-20)")


class PatientObservation(BaseModel):
    patient_id: str
    age: int
    symptoms: List[str]
    symptom_duration_hours: int
    chronic_conditions: List[str]
    past_visits_30_days: int
    pain_scale: int = Field(..., ge=0, le=10)
    vitals: VitalSigns
    visit_history: Optional[List[dict]] = Field(default=[], description="Past visits in this session")
    time_of_day: Optional[str] = Field(default="morning", description="morning/afternoon/evening/night")
    season: Optional[str] = Field(default="summer", description="summer/winter/monsoon/spring")


class TriageAction(BaseModel):
    urgency_level: UrgencyLevel
    reasoning: str
    recommended_action: str
    estimated_wait_minutes: int
    predicted_diagnosis: Optional[str] = None


class StepResult(BaseModel):
    observation: PatientObservation
    reward: float = Field(..., gt=0.0, lt=1.0)  # strictly between 0 and 1
    done: bool
    info: dict
    next_patient: Optional[PatientObservation] = None

    @validator("reward")
    def reward_open_interval(cls, v):
        return max(0.001, min(0.999, v))