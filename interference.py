import os
import json
from openai import OpenAI
from env import MediTriageEnv
from models import TriageAction, UrgencyLevel
from grader import run_grader

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

def llm_triage_agent(obs) -> TriageAction:
    prompt = f"""You are a medical triage AI. Given patient info, decide urgency.
Patient: Age {obs.age}, Symptoms: {obs.symptoms}, Duration: {obs.duration_hours}hrs, Chronic condition: {obs.has_chronic_condition}

Respond ONLY with valid JSON:
{{"urgency_level": 0-3, "predicted_disease": "string", "first_aid_suggestion": "string", "alert_doctor": true/false}}
(0=Self-care, 1=Non-urgent, 2=Urgent, 3=Emergency)"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(response.choices[0].message.content)
        return TriageAction(**data)
    except:
        return TriageAction(
            urgency_level=UrgencyLevel.NON_URGENT,
            predicted_disease="Unknown",
            first_aid_suggestion="Consult a doctor",
            alert_doctor=False
        )

def run_inference():
    for task_level in ["easy", "medium", "hard"]:
        env = MediTriageEnv(task_level)
        total_reward = 0.0
        episodes = 5

        print(json.dumps({"type": "[START]", "task": task_level, "episodes": episodes}))

        for ep in range(episodes):
            obs = env.reset()
            action = llm_triage_agent(obs)
            result = env.step(action)
            total_reward += result.reward

            print(json.dumps({
                "type": "[STEP]",
                "task": task_level,
                "episode": ep + 1,
                "reward": result.reward,
                "predicted": int(action.urgency_level),
                "true": result.info["true_urgency"]
            }))

        score = run_grader(task_level)
        print(json.dumps({
            "type": "[END]",
            "task": task_level,
            "total_reward": round(total_reward, 3),
            "score": score
        }))

if __name__ == "__main__":
    run_inference()