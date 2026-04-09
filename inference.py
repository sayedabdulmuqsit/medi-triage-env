"""
MediGuide RL Inference Agent
Runs against the live API and logs in [START][STEP][END] format.
"""

import os
import json
import time
import requests

ENV_BASE_URL = "https://sayedabdulmuqsit11-medi-triage-env.hf.space"
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASKS = ["easy", "medium", "hard", "expert", "adversarial"]
EPISODES_PER_TASK = 3


def call_llm(prompt):
    base = os.environ.get("API_BASE_URL", "").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    api_key = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy"))
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
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
        "max_tokens": 300,
        "temperature": 0.1,
    }
    r = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def build_prompt(obs):
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


def run_episode(task, episode_num):
    print(f"[START] task={task} episode={episode_num}")
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        print(f"[STEP] reset_error | {str(e)[:100]}")
        print(f"[END] task={task} episode={episode_num} reward=0")
        return {"task": task, "episode": episode_num, "reward": 0.01, "score": 0.01}

    print(f"[STEP] reset | patient={obs.get('patient_id')} | task={task}")
    prompt = build_prompt(obs)
    start = time.time()

    try:
        llm_response = call_llm(prompt)
        raw = llm_response.strip().replace("```json", "").replace("```", "")
        decision = json.loads(raw)
    except Exception as e:
        print(f"[STEP] llm_error | {str(e)[:100]}")
        decision = {"urgency_level": 2, "reasoning": "fallback", "recommended_action": "See doctor", "estimated_wait_minutes": 60, "predicted_diagnosis": "unknown"}

    elapsed = round(time.time() - start, 2)
    print(f"[STEP] decision | urgency={decision.get('urgency_level')} | time={elapsed}s")

    try:
        r2 = requests.post(f"{ENV_BASE_URL}/step", json={
            "urgency_level": decision.get("urgency_level", 2),
            "reasoning": decision.get("reasoning", ""),
            "recommended_action": decision.get("recommended_action", ""),
            "estimated_wait_minutes": decision.get("estimated_wait_minutes", 60),
            "predicted_diagnosis": decision.get("predicted_diagnosis", ""),
        }, timeout=30)
        r2.raise_for_status()
        result = r2.json()
    except Exception as e:
        print(f"[STEP] step_error | {str(e)[:100]}")
        print(f"[END] task={task} episode={episode_num} reward=0")
        return {"task": task, "episode": episode_num, "reward": 0.01, "score": 0.01}

    reward = result.get("reward", 0)
    correct = result.get("info", {}).get("correct_urgency")
    score = max(0.01, min(0.99, float(reward)))
    print(f"[STEP] result | reward={reward} | correct_urgency={correct} | predicted={decision.get('urgency_level')}")
    print(f"[END] task={task} episode={episode_num} reward={reward}")
    return {"task": task, "episode": episode_num, "reward": reward, "score": score}


def main():
    print("[START] MediGuide RL Inference Agent v2.0")
    print(f"[STEP] ENV_BASE_URL={ENV_BASE_URL} | MODEL={MODEL_NAME}")
    results = []
    for task in TASKS:
        task_scores = []
        for ep in range(1, EPISODES_PER_TASK + 1):
            res = run_episode(task, ep)
            results.append(res)
            task_scores.append(res["score"])
        avg = round(sum(task_scores) / max(len(task_scores), 1), 3)
        print(f"[STEP] task_summary | task={task} | avg_score={avg}")
    try:
        r = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
        if r.ok:
            print(f"[STEP] final_state | {json.dumps(r.json())}")
    except Exception:
        pass
    overall = round(sum(x["score"] for x in results) / max(len(results), 1), 3)
    print(f"[END] all_tasks_complete | overall_avg_score={overall} | episodes={len(results)}")


if __name__ == "__main__":
    main()