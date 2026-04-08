"""
MediGuide RL Inference Agent
Runs against the live API and logs in [START][STEP][END] format.
"""

import os
import json
import time
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://sayedabdulmuqsit11-medi-triage-env.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TASKS = ["easy", "medium", "hard", "expert", "adversarial"]
EPISODES_PER_TASK = 3


def call_llm(prompt: str) -> str:
    """Call the LLM to make a triage decision."""
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical triage AI. Given patient data including vitals, "
                    "symptoms, age, and history, classify urgency:\n"
                    "0=Self-Care (handle at home)\n"
                    "1=Non-Urgent (see doctor within 48h)\n"
                    "2=Urgent (see doctor today)\n"
                    "3=Emergency (immediate ER)\n\n"
                    "CRITICAL: Always check vitals first. "
                    "SpO2 < 92, HR > 120, BP systolic < 90, or RR > 28 = Emergency regardless of symptoms.\n\n"
                    "Respond ONLY with valid JSON:\n"
                    '{"urgency_level": <0-3>, "reasoning": "<text>", '
                    '"recommended_action": "<text>", "estimated_wait_minutes": <int>, '
                    '"predicted_diagnosis": "<optional text>"}'
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
    visit_history = obs.get("visit_history", [])
    history_str = ""
    if visit_history:
        history_str = f"\nVisit History (last {len(visit_history)} visits):\n"
        for v in visit_history:
            history_str += (
                f"  - Symptoms: {v.get('symptoms', [])}, "
                f"Urgency assigned: {v.get('urgency_assigned')}, "
                f"Reward: {v.get('reward', 0):.2f}\n"
            )

    return f"""Patient Data:
- Patient ID: {obs.get('patient_id')}
- Age: {obs.get('age')}
- Symptoms: {', '.join(obs.get('symptoms', []))}
- Duration: {obs.get('symptom_duration_hours')} hours
- Chronic Conditions: {', '.join(obs.get('chronic_conditions', [])) or 'None'}
- Past Visits (30 days): {obs.get('past_visits_30_days')}
- Pain Scale: {obs.get('pain_scale')}/10
- Time of Day: {obs.get('time_of_day', 'unknown')}
- Season: {obs.get('season', 'unknown')}

Vital Signs:
- Heart Rate: {vitals.get('heart_rate')} BPM (normal: 60-100)
- Blood Pressure: {vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg (normal: 120/80)
- Oxygen Saturation (SpO2): {vitals.get('oxygen_saturation')}% (normal: 95-100%)
- Temperature: {vitals.get('temperature')}°C (normal: 36.1-37.2°C)
- Respiratory Rate: {vitals.get('respiratory_rate')} breaths/min (normal: 12-20){history_str}

Provide your triage decision as JSON."""


def run_episode(task: str, episode_num: int) -> dict:
    print(f"[START] task={task} episode={episode_num}")

    # Reset
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    obs = r.json()
    print(f"[STEP] reset | patient={obs.get('patient_id')} | task={task}")

    # Build prompt
    prompt = build_prompt(obs)

    # LLM decision
    start = time.time()
    try:
        llm_response = call_llm(prompt)
        raw = llm_response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
    except Exception as e:
        print(f"[STEP] llm_error={e} | using fallback decision")
        decision = {
            "urgency_level": 2,
            "reasoning": "Fallback: LLM unavailable, defaulting to Urgent for safety",
            "recommended_action": "Refer to doctor",
            "estimated_wait_minutes": 60,
            "predicted_diagnosis": None,
        }

    elapsed = round(time.time() - start, 2)
    print(
        f"[STEP] decision | urgency={decision.get('urgency_level')} | "
        f"time={elapsed}s | reasoning={decision.get('reasoning', '')[:80]}"
    )

    # Step
    step_payload = {
        "urgency_level": decision.get("urgency_level", 2),
        "reasoning": decision.get("reasoning", ""),
        "recommended_action": decision.get("recommended_action", ""),
        "estimated_wait_minutes": decision.get("estimated_wait_minutes", 60),
        "predicted_diagnosis": decision.get("predicted_diagnosis"),
    }
    r2 = requests.post(f"{API_BASE_URL}/step", json=step_payload, timeout=30)
    r2.raise_for_status()
    result = r2.json()

    reward = result.get("reward", 0)
    correct = result.get("info", {}).get("correct_urgency")
    print(
        f"[STEP] result | reward={reward} | correct_urgency={correct} | "
        f"predicted={decision.get('urgency_level')}"
    )

    # Grade
    grade_params = {
        "urgency_level": decision.get("urgency_level", 2),
        "reasoning": decision.get("reasoning", ""),
        "recommended_action": decision.get("recommended_action", ""),
        "estimated_wait_minutes": decision.get("estimated_wait_minutes", 60),
        "task": task,
        "predicted_diagnosis": decision.get("predicted_diagnosis", ""),
    }
    r3 = requests.get(f"{API_BASE_URL}/grade", params=grade_params, timeout=30)
    score = r3.json().get("score", 0) if r3.ok else 0
    print(f"[STEP] grade | score={score} | task={task}")

    print(f"[END] task={task} episode={episode_num} reward={reward} score={score}")
    return {"task": task, "episode": episode_num, "reward": reward, "score": score}


def main():
    print("[START] MediGuide RL Inference Agent v2.0")
    print(f"[STEP] API_BASE_URL={API_BASE_URL} | MODEL={MODEL_NAME}")

    results = []
    for task in TASKS:
        task_scores = []
        for ep in range(1, EPISODES_PER_TASK + 1):
            try:
                res = run_episode(task, ep)
                results.append(res)
                task_scores.append(res["score"])
            except Exception as e:
                print(f"[STEP] error | task={task} ep={ep} | {e}")

        avg = round(sum(task_scores) / max(len(task_scores), 1), 3)
        print(f"[STEP] task_summary | task={task} | avg_score={avg}")

    # Final state
    r = requests.get(f"{API_BASE_URL}/state", timeout=10)
    if r.ok:
        state = r.json()
        print(f"[STEP] final_state | {json.dumps(state)}")

    overall = round(sum(r["score"] for r in results) / max(len(results), 1), 3)
    print(f"[END] all_tasks_complete | overall_avg_score={overall} | episodes={len(results)}")


if __name__ == "__main__":
    main()