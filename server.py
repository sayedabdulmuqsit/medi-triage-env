from fastapi import FastAPI
from models import TriageAction, PatientObservation, StepResult
from env import MediTriageEnv

app = FastAPI(title="MediGuide RL - Healthcare Triage Environment")

envs = {
    "easy": MediTriageEnv("easy"),
    "medium": MediTriageEnv("medium"),
    "hard": MediTriageEnv("hard")
}

@app.get("/")
def root():
    return {"status": "ok", "env": "MediGuide RL"}

@app.post("/reset/{task_level}", response_model=PatientObservation)
def reset(task_level: str = "easy"):
    return envs[task_level].reset()

@app.post("/step/{task_level}", response_model=StepResult)
def step(action: TriageAction, task_level: str = "easy"):
    return envs[task_level].step(action)

@app.get("/state/{task_level}")
def state(task_level: str = "easy"):
    return envs[task_level].state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)