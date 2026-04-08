import requests

BASE = "http://localhost:7860"

def test(name, passed, detail=""):
    print(f"{'✅' if passed else '❌'}  {name}")
    if detail and not passed:
        print(f"   → {detail}")

# API online
r = requests.get(f"{BASE}/state")
test("API is online", r.status_code == 200)

# Feature 1 - Vital Signs
r = requests.post(f"{BASE}/reset", json={"task": "easy"})
obs = r.json()
vitals_ok = all(k in obs.get("vitals", {}) for k in ["heart_rate","oxygen_saturation","temperature","systolic_bp","diastolic_bp","respiratory_rate"])
test("Feature 1 - Vital Signs", vitals_ok, str(obs.get("vitals")))

# Feature 3 - Stochasticity
test("Feature 3 - Season & Time of Day", "season" in obs and "time_of_day" in obs)

# Feature 2 - Shaped Reward
requests.post(f"{BASE}/reset", json={"task": "hard"})
r = requests.post(f"{BASE}/step", json={
    "urgency_level": 3,
    "reasoning": "Emergency - high HR, low SpO2, chest pain, hypertension",
    "recommended_action": "Immediate ER",
    "estimated_wait_minutes": 0,
    "predicted_diagnosis": "Acute Myocardial Infarction"
})
result = r.json()
test("Feature 2 - Shaped Reward returns value", "reward" in result, str(result))
test("Feature 2 - Correct triage gives positive reward", result.get("reward", -99) > 0)
# Feature 6 - Analytics
r = requests.get(f"{BASE}/analytics")
ana = r.json()
test("Feature 6 - Analytics endpoint", all(k in ana for k in ["total_episodes","avg_reward","missed_emergencies_rate","overtriage_rate"]))

# Feature 7 - Hindi
r = requests.get(f"{BASE}/translate", params={"text": "सीने में दर्द"})
test("Feature 7 - Hindi Translation", r.json().get("translated") == "chest pain", str(r.json()))

# Feature 4&5 - All 5 tasks
for task in ["easy","medium","hard","expert","adversarial"]:
    r = requests.post(f"{BASE}/reset", json={"task": task})
    test(f"Task level - {task}", r.status_code == 200 and "vitals" in r.json())

# Feature 8 - SQLite Visit History
requests.post(f"{BASE}/reset", json={"task": "easy"})
requests.post(f"{BASE}/step", json={
    "urgency_level": 2,
    "reasoning": "Urgent care needed",
    "recommended_action": "See doctor today",
    "estimated_wait_minutes": 60
})
r = requests.post(f"{BASE}/reset", json={"task": "easy"})
test("Feature 8 - SQLite Visit History", isinstance(r.json().get("visit_history"), list))

# Dashboard
r = requests.get(f"{BASE}/")
test("Feature 9 - Dashboard loads", r.status_code == 200 and "MediGuide" in r.text)

print("\nDone. Fix any ❌ before submitting.")