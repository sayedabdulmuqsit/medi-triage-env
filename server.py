import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from models import PatientObservation, TriageAction, StepResult, UrgencyLevel
from env import MediGuideEnv
from grader import grade

app = FastAPI(
    title="MediGuide RL Environment",
    description="Healthcare Triage RL Environment for OpenEnv Hackathon",
    version="2.0.0",
)

# Single shared environment instance
env = MediGuideEnv()

# ── Hindi symptom keyword mapping ─────────────────────────────────────────────
HINDI_SYMPTOM_MAP = {
    "सिरदर्द": "headache",
    "बुखार": "fever",
    "खांसी": "cough",
    "सांस लेने में तकलीफ": "shortness of breath",
    "सीने में दर्द": "chest pain",
    "पेट दर्द": "stomach pain",
    "उल्टी": "vomiting",
    "जी मिचलाना": "nausea",
    "चक्कर आना": "dizziness",
    "थकान": "fatigue",
    "दस्त": "diarrhea",
    "पीठ दर्द": "back pain",
    "गले में खराश": "sore throat",
    "बेहोशी": "loss of consciousness",
    "धड़कन तेज": "rapid heartbeat",
    "हाथ में दर्द": "arm pain",
    "बाएं हाथ में दर्द": "left arm pain",
    "पसीना": "sweating",
    "भ्रम": "confusion",
    "बोलने में तकलीफ": "slurred speech",
    "मुंह टेढ़ा": "facial drooping",
    "खून की उल्टी": "coughing blood",
    "रात को पसीना": "night sweats",
    "जोड़ों का दर्द": "joint pain",
    "ठंड लगना": "chills",
}


def translate_hindi_symptoms(text: str) -> str:
    """Translate Hindi symptom keywords to English."""
    result = text
    for hindi, english in HINDI_SYMPTOM_MAP.items():
        result = result.replace(hindi, english)
    return result


# ── Request/Response models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    urgency_level: int
    reasoning: str
    recommended_action: str
    estimated_wait_minutes: int
    predicted_diagnosis: Optional[str] = None
    hindi_input: Optional[str] = None


class GradeRequest(BaseModel):
    urgency_level: int
    reasoning: str
    recommended_action: str
    estimated_wait_minutes: int
    predicted_diagnosis: Optional[str] = None
    task: str = "easy"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "MediGuide RL Environment v2.0 — visit /docs"}


@app.post("/reset", response_model=PatientObservation)
async def reset(request: ResetRequest):
    valid_tasks = ["easy", "medium", "hard", "expert", "adversarial"]
    if request.task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Choose from: {valid_tasks}",
        )
    obs = env.reset(task=request.task)
    return obs


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    if env.current_scenario is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    # Translate Hindi input if provided
    reasoning = request.reasoning
    if request.hindi_input:
        translated = translate_hindi_symptoms(request.hindi_input)
        reasoning = f"{reasoning} [Translated: {translated}]"

    try:
        urgency = UrgencyLevel(request.urgency_level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="urgency_level must be 0=Self-Care, 1=Non-Urgent, 2=Urgent, 3=Emergency",
        )

    action = TriageAction(
        urgency_level=urgency,
        reasoning=reasoning,
        recommended_action=request.recommended_action,
        estimated_wait_minutes=request.estimated_wait_minutes,
        predicted_diagnosis=request.predicted_diagnosis,
    )

    result = env.step(action)
    return result


@app.get("/state")
async def state():
    return env.state()


@app.get("/grade")
async def grade_endpoint(
    urgency_level: int,
    reasoning: str,
    recommended_action: str,
    estimated_wait_minutes: int,
    task: str = "easy",
    predicted_diagnosis: Optional[str] = None,
):
    if env.current_scenario is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    try:
        urgency = UrgencyLevel(urgency_level)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid urgency_level (0-3)")

    action = TriageAction(
        urgency_level=urgency,
        reasoning=reasoning,
        recommended_action=recommended_action,
        estimated_wait_minutes=estimated_wait_minutes,
        predicted_diagnosis=predicted_diagnosis,
    )

    correct = UrgencyLevel(env.current_scenario["correct_urgency"])

    if env.current_scenario is None:
        raise HTTPException(status_code=400, detail="No active scenario")

    obs = PatientObservation(
        patient_id=env.current_scenario["patient_id"],
        age=env.current_scenario["age"],
        symptoms=env.current_scenario["symptoms"],
        symptom_duration_hours=env.current_scenario["symptom_duration_hours"],
        chronic_conditions=env.current_scenario["chronic_conditions"],
        past_visits_30_days=env.current_scenario["past_visits_30_days"],
        pain_scale=env.current_scenario["pain_scale"],
        vitals=env.reset(task).__dict__["vitals"] if False else _dummy_vitals(),
    )

    score = grade(obs, action, correct, task=task)
    return {
        "score": score,
        "correct_urgency": int(correct),
        "predicted_urgency": urgency_level,
        "task": task,
    }


@app.get("/analytics")
async def analytics():
    return env.analytics()


@app.get("/translate")
async def translate(text: str):
    """Translate Hindi symptom descriptions to English."""
    translated = translate_hindi_symptoms(text)
    return {"original": text, "translated": translated}


@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Easy",
                "description": "Clear-cut cases: obvious self-care or emergency",
                "typical_score": 0.90,
            },
            {
                "id": "medium",
                "name": "Medium",
                "description": "Ambiguous cases requiring reasoning about chronic conditions",
                "typical_score": 0.70,
            },
            {
                "id": "hard",
                "name": "Hard",
                "description": "Complex multi-comorbidity cases requiring expert judgment",
                "typical_score": 0.95,
            },
            {
                "id": "expert",
                "name": "Expert",
                "description": "Mass casualty event: triage 5 simultaneous patients",
                "typical_score": 0.85,
            },
            {
                "id": "adversarial",
                "name": "Adversarial",
                "description": "Contradicting vitals vs reported symptoms — trust the data",
                "typical_score": 0.80,
            },
        ]
    }


def _dummy_vitals():
    from models import VitalSigns
    return VitalSigns(
        heart_rate=72,
        systolic_bp=120,
        diastolic_bp=80,
        oxygen_saturation=98.0,
        temperature=36.8,
        respiratory_rate=16,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)