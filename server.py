import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from models import PatientObservation, TriageAction, StepResult, UrgencyLevel
from env import MediGuideEnv, _clamp
from grader import grade

app = FastAPI(
    title="MediGuide RL Environment",
    description="Healthcare Triage RL Environment for OpenEnv Hackathon",
    version="2.0.0",
)

env = MediGuideEnv()

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
    result = text
    for hindi, english in HINDI_SYMPTOM_MAP.items():
        result = result.replace(hindi, english)
    return result


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    urgency_level: int
    reasoning: str
    recommended_action: str
    estimated_wait_minutes: int
    predicted_diagnosis: Optional[str] = None
    hindi_input: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "MediGuide RL Environment v2.0 — visit /docs"}


@app.post("/reset", response_model=PatientObservation)
async def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
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

    # Step the environment (reward already clamped inside env.step via _clamp)
    result = env.step(action)

    # Also run the grader for the current scenario (used by validator)
    current_obs  = env.current_scenario
    grader_score = grade(
        observation=current_obs,
        action={
            "urgency_level":      request.urgency_level,
            "reasoning":          reasoning,
            "recommended_action": request.recommended_action,
            "predicted_diagnosis": request.predicted_diagnosis or "",
        },
        correct_urgency=current_obs.get("correct_urgency", 0),
        task=env.current_task,
    )

    # Use grader score as the final reward (grader already clamps to (0.001, 0.999))
    final_reward = _clamp(grader_score)

    return StepResult(
        observation=result.observation,
        reward=final_reward,
        done=result.done,
        info={
            **result.info,
            "grader_score": final_reward,
        },
        next_patient=result.next_patient,
    )


@app.get("/state")
async def state():
    return env.state()


@app.get("/analytics")
async def analytics():
    return env.analytics()


@app.get("/translate")
async def translate(text: str):
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
                "typical_score": 0.88,
            },
            {
                "id": "medium",
                "name": "Medium",
                "description": "Ambiguous cases requiring reasoning about chronic conditions",
                "typical_score": 0.69,
            },
            {
                "id": "hard",
                "name": "Hard",
                "description": "Complex multi-comorbidity cases requiring expert judgment",
                "typical_score": 0.92,
            },
            {
                "id": "expert",
                "name": "Expert",
                "description": "Mass casualty event: triage 5 simultaneous patients",
                "typical_score": 0.84,
            },
            {
                "id": "adversarial",
                "name": "Adversarial",
                "description": "Contradicting vitals vs reported symptoms — trust the data",
                "typical_score": 0.79,
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)