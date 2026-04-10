"""
MediGuide RL Inference Agent
Logs in strict [START][STEP][END] format required by Scaler/Meta OpenEnv validator.
"""

import os
import json
import time
from openai import OpenAI

ENV_BASE_URL = "https://sayedabdulmuqsit11-medi-triage-env.hf.space"
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", "dummy"))

TASKS            = ["easy", "medium", "hard", "expert", "adversarial"]
EPISODES_PER_TASK = 3

# ── OpenAI client (required by submission rules) ──────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL.rstrip("/"),
)


def _clamp(v: float) -> float:
    """Strictly (0, 1) — never 0.0 or 1.0."""
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
        "Decide triage urgency as JSON."
    )


def run_task(task: str) -> float:
    """
    Run EPISODES_PER_TASK episodes for one task.
    Emits exactly ONE [START] and ONE [END] block per task (required format).
    Returns average clamped score.
    """
    import requests  # only for env API, not LLM

    # ── [START] ───────────────────────────────────────────────────────────────
    print(f"[START] task={task} env=medi-triage model={MODEL_NAME}")

    all_rewards: list[float] = []
    step_num = 0

    for ep in range(1, EPISODES_PER_TASK + 1):
        # reset
        try:
            r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
            r.raise_for_status()
            obs = r.json()
        except Exception as e:
            step_num += 1
            print(f"[STEP] step={step_num} action=null reward=0.001 done=true error={str(e)[:80]}")
            all_rewards.append(0.001)
            continue


        # LLM decision
        prompt = build_prompt(obs)
        try:
            raw   = call_llm(prompt).replace("```json", "").replace("```", "")
            decision = json.loads(raw)
        except Exception as e:
            decision = {
                "urgency_level": 2,
                "reasoning": "fallback",
                "recommended_action": "See doctor",
                "estimated_wait_minutes": 60,
                "predicted_diagnosis": "unknown",
            }

        # step
        try:
            r2 = requests.post(
                f"{ENV_BASE_URL}/step",
                json={
                    "urgency_level":       decision.get("urgency_level", 2),
                    "reasoning":           decision.get("reasoning", ""),
                    "recommended_action":  decision.get("recommended_action", ""),
                    "estimated_wait_minutes": decision.get("estimated_wait_minutes", 60),
                    "predicted_diagnosis": decision.get("predicted_diagnosis", ""),
                },
                timeout=30,
            )
            r2.raise_for_status()
            result = r2.json()
        except Exception as e:
            step_num += 1
            print(f"[STEP] step={step_num} action={json.dumps(decision)} reward=0.001 done=true error={str(e)[:80]}")
            all_rewards.append(0.001)
            continue

        raw_reward = result.get("reward", 0.001)
        reward     = _clamp(raw_reward)
        done       = result.get("done", True)

        step_num += 1
        action_json = json.dumps({"urgency_level": decision.get("urgency_level")})
        print(f"[STEP] step={step_num} action={action_json} reward={reward:.4f} done={str(done).lower()} error=null")
        all_rewards.append(reward)

    # aggregate
    avg_score = _clamp(sum(all_rewards) / max(len(all_rewards), 1))

    # ── [END] ─────────────────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.4f}" for r in all_rewards)
    success     = avg_score > 0.5
    print(f"[END] success={str(success).lower()} steps={step_num} score={avg_score:.4f} rewards={rewards_str}")

    return avg_score


def main():
    all_scores = []
    for task in TASKS:
        score = run_task(task)
        all_scores.append(score)
        time.sleep(1)  # brief pause between tasks

   

if __name__ == "__main__":
    main()