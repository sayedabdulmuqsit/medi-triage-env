"""
MediGuide RL Inference Agent
Logs in strict [START][STEP][END] format required by Scaler/Meta OpenEnv validator.
"""

import os
import json
import time
import requests
from openai import OpenAI

ENV_BASE_URL = "https://sayedabdulmuqsit11-medi-triage-env.hf.space"

# Exactly as Scaler requires — use injected env vars
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", "dummy")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASKS             = ["easy", "medium", "hard", "expert", "adversarial"]
EPISODES_PER_TASK = 3

# Initialize at module level exactly like Scaler's sample
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _clamp(v: float) -> float:
    return max(0.001, min(0.999, float(v)))


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical triage AI. Classify urgency:\n"
                    "0=Self-Care, 1=Non-Urgent, 2=Urgent, 3=Emergency\n"
                    "CRITICAL: SpO2<92, HR>120, BP systolic<90, RR>28 = Emergency.\n"
                    "Respond ONLY with valid JSON:\n"
                    '{"urgency_level": <0-3>, "reasoning": "<text>", '
                    '"recommended_action": "<text>", "estimated_wait_minutes": <int>, '
                    '"predicted_diagnosis": "<text>"}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def build_prompt(obs: dict) -> str:
    vitals = obs.get("vitals", {})
    return (
        f"Patient: age={obs.get('age')}, symptoms={obs.get('symptoms')}, "
        f"duration={obs.get('symptom_duration_hours')}h\n"
        f"Chronic: {obs.get('chronic_conditions')}, pain={obs.get('pain_scale')}/10\n"
        f"Vitals: HR={vitals.get('heart_rate')}, BP={vitals.get('systolic_bp')}/"
        f"{vitals.get('diastolic_bp')}, SpO2={vitals.get('oxygen_saturation')}%, "
        f"Temp={vitals.get('temperature')}C, RR={vitals.get('respiratory_rate')}\n"
        "Respond with JSON only."
    )


def run_task(task: str) -> float:
    print(f"[START] task={task} env=medi-triage model={MODEL_NAME}", flush=True)

    all_rewards = []
    step_num = 0

    for ep in range(1, EPISODES_PER_TASK + 1):
        # Reset environment
        try:
            r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
            r.raise_for_status()
            obs = r.json()
        except Exception as e:
            step_num += 1
            print(f"[STEP] step={step_num} action=null reward=0.00 done=true error={str(e)[:80]}", flush=True)
            all_rewards.append(0.001)
            continue

        # LLM call through Scaler's proxy — no silent fallback
        prompt = build_prompt(obs)
        raw = call_llm(prompt).replace("```json", "").replace("```", "").strip()
        decision = json.loads(raw)

        # Step environment
        try:
            r2 = requests.post(
                f"{ENV_BASE_URL}/step",
                json={
                    "urgency_level":          decision.get("urgency_level", 2),
                    "reasoning":              decision.get("reasoning", ""),
                    "recommended_action":     decision.get("recommended_action", ""),
                    "estimated_wait_minutes": decision.get("estimated_wait_minutes", 60),
                    "predicted_diagnosis":    decision.get("predicted_diagnosis", ""),
                },
                timeout=30,
            )
            r2.raise_for_status()
            result = r2.json()
        except Exception as e:
            step_num += 1
            print(f"[STEP] step={step_num} action={json.dumps(decision)} reward=0.00 done=true error={str(e)[:80]}", flush=True)
            all_rewards.append(0.001)
            continue

        reward = _clamp(result.get("reward", 0.001))
        done   = result.get("done", True)
        step_num += 1

        action_json = json.dumps({"urgency_level": decision.get("urgency_level")})
        print(f"[STEP] step={step_num} action={action_json} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        all_rewards.append(reward)

    avg_score   = _clamp(sum(all_rewards) / max(len(all_rewards), 1))
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    success     = avg_score > 0.5
    print(f"[END] success={str(success).lower()} steps={step_num} score={avg_score:.2f} rewards={rewards_str}", flush=True)

    return avg_score


def main():
    for task in TASKS:
        run_task(task)
        time.sleep(1)


if __name__ == "__main__":
    main()