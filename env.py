import random
import time
import sqlite3
import json
import os
from typing import Optional
from models import (
    PatientObservation, TriageAction, StepResult,
    UrgencyLevel, VitalSigns
)

DB_PATH = "/tmp/mediguide_sessions.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            session_id TEXT,
            patient_id TEXT,
            timestamp REAL,
            symptoms TEXT,
            urgency_assigned INTEGER,
            reward REAL
        )
    """)
    conn.commit()
    conn.close()


_init_db()


# ── Vital sign generators per urgency level ──────────────────────────────────

def _vitals_for_urgency(urgency: int) -> VitalSigns:
    if urgency == UrgencyLevel.EMERGENCY:
        return VitalSigns(
            heart_rate=random.randint(120, 160),
            systolic_bp=random.choice([random.randint(70, 89), random.randint(180, 210)]),
            diastolic_bp=random.choice([random.randint(40, 59), random.randint(110, 130)]),
            oxygen_saturation=round(random.uniform(82.0, 91.0), 1),
            temperature=round(random.uniform(39.5, 41.0), 1),
            respiratory_rate=random.randint(28, 40)
        )
    elif urgency == UrgencyLevel.URGENT:
        return VitalSigns(
            heart_rate=random.randint(100, 119),
            systolic_bp=random.randint(130, 160),
            diastolic_bp=random.randint(85, 100),
            oxygen_saturation=round(random.uniform(92.0, 94.9), 1),
            temperature=round(random.uniform(38.5, 39.4), 1),
            respiratory_rate=random.randint(21, 27)
        )
    elif urgency == UrgencyLevel.NON_URGENT:
        return VitalSigns(
            heart_rate=random.randint(72, 99),
            systolic_bp=random.randint(110, 129),
            diastolic_bp=random.randint(70, 84),
            oxygen_saturation=round(random.uniform(95.0, 97.9), 1),
            temperature=round(random.uniform(37.3, 38.4), 1),
            respiratory_rate=random.randint(16, 20)
        )
    else:  # SELF_CARE
        return VitalSigns(
            heart_rate=random.randint(60, 75),
            systolic_bp=random.randint(100, 120),
            diastolic_bp=random.randint(60, 80),
            oxygen_saturation=round(random.uniform(97.0, 100.0), 1),
            temperature=round(random.uniform(36.1, 37.2), 1),
            respiratory_rate=random.randint(12, 16)
        )


# ── Scenario bank ─────────────────────────────────────────────────────────────

def _get_time_and_season():
    hour = random.randint(0, 23)
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 17:
        tod = "afternoon"
    elif 17 <= hour < 21:
        tod = "evening"
    else:
        tod = "night"

    month = random.randint(1, 12)
    if month in [6, 7, 8, 9]:
        season = "monsoon"
    elif month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "summer"
    else:
        season = "spring"

    return tod, season


def _season_symptoms(season: str, tod: str):
    """Return extra contextual symptoms based on season/time."""
    extras = []
    if season == "monsoon":
        extras = random.choice([
            ["high fever", "joint pain"],        # dengue
            ["chills", "sweating"],               # malaria
            ["watery diarrhea", "nausea"],        # cholera risk
            []
        ])
    elif season == "summer":
        extras = random.choice([
            ["dizziness", "excessive thirst"],    # heat stroke risk
            ["sunburn", "headache"],
            []
        ])
    elif season == "winter":
        extras = random.choice([
            ["runny nose", "sore throat"],
            ["chest tightness", "wheezing"],      # asthma flare
            []
        ])
    if tod == "night":
        extras += random.choice([["severe chest pain"], ["difficulty breathing"], []])
    return extras


SCENARIOS = {
    "easy": [
        {
            "patient_id": "E001",
            "age": 28,
            "symptoms": ["mild headache", "runny nose"],
            "symptom_duration_hours": 24,
            "chronic_conditions": [],
            "past_visits_30_days": 0,
            "pain_scale": 3,
            "correct_urgency": UrgencyLevel.SELF_CARE,
        },
        {
            "patient_id": "E002",
            "age": 45,
            "symptoms": ["severe chest pain", "left arm pain", "sweating"],
            "symptom_duration_hours": 1,
            "chronic_conditions": ["hypertension"],
            "past_visits_30_days": 1,
            "pain_scale": 9,
            "correct_urgency": UrgencyLevel.EMERGENCY,
        },
        {
            "patient_id": "E003",
            "age": 32,
            "symptoms": ["mild sore throat", "low-grade fever"],
            "symptom_duration_hours": 48,
            "chronic_conditions": [],
            "past_visits_30_days": 0,
            "pain_scale": 2,
            "correct_urgency": UrgencyLevel.SELF_CARE,
        },
    ],
    "medium": [
        {
            "patient_id": "M001",
            "age": 60,
            "symptoms": ["persistent cough", "mild fever", "fatigue"],
            "symptom_duration_hours": 72,
            "chronic_conditions": ["diabetes"],
            "past_visits_30_days": 2,
            "pain_scale": 5,
            "correct_urgency": UrgencyLevel.URGENT,
        },
        {
            "patient_id": "M002",
            "age": 35,
            "symptoms": ["stomach pain", "nausea", "vomiting"],
            "symptom_duration_hours": 12,
            "chronic_conditions": [],
            "past_visits_30_days": 0,
            "pain_scale": 6,
            "correct_urgency": UrgencyLevel.URGENT,
        },
        {
            "patient_id": "M003",
            "age": 50,
            "symptoms": ["back pain", "frequent urination", "burning sensation"],
            "symptom_duration_hours": 36,
            "chronic_conditions": ["hypertension"],
            "past_visits_30_days": 1,
            "pain_scale": 6,
            "correct_urgency": UrgencyLevel.URGENT,
        },
    ],
    "hard": [
        {
            "patient_id": "H001",
            "age": 70,
            "symptoms": ["sudden confusion", "slurred speech", "facial drooping"],
            "symptom_duration_hours": 2,
            "chronic_conditions": ["hypertension", "diabetes", "heart disease"],
            "past_visits_30_days": 3,
            "pain_scale": 7,
            "correct_urgency": UrgencyLevel.EMERGENCY,
        },
        {
            "patient_id": "H002",
            "age": 55,
            "symptoms": ["severe abdominal pain", "rigid abdomen", "fever"],
            "symptom_duration_hours": 6,
            "chronic_conditions": ["diabetes"],
            "past_visits_30_days": 1,
            "pain_scale": 9,
            "correct_urgency": UrgencyLevel.EMERGENCY,
        },
        {
            "patient_id": "H003",
            "age": 40,
            "symptoms": ["shortness of breath", "coughing blood", "night sweats"],
            "symptom_duration_hours": 120,
            "chronic_conditions": [],
            "past_visits_30_days": 0,
            "pain_scale": 8,
            "correct_urgency": UrgencyLevel.EMERGENCY,
        },
    ],
    "expert": [
        {
            "patient_id": "X001",
            "age": 55,
            "symptoms": ["chest pain", "shortness of breath"],
            "symptom_duration_hours": 2,
            "chronic_conditions": ["diabetes", "hypertension"],
            "past_visits_30_days": 0,
            "pain_scale": 8,
            "correct_urgency": UrgencyLevel.EMERGENCY,
            "mass_casualty": True,
            "queue_size": 5,
        },
    ],
    "adversarial": [
        {
            "patient_id": "A001",
            "age": 30,
            "symptoms": ["mild headache"],
            "symptom_duration_hours": 1,
            "chronic_conditions": [],
            "past_visits_30_days": 0,
            "pain_scale": 2,
            "correct_urgency": UrgencyLevel.EMERGENCY,
            "adversarial_note": "Vitals show critical hypoxia despite mild reported symptoms",
        },
    ],
}


class MediGuideEnv:
    def __init__(self):
        self.current_scenario = None
        self.current_task = "easy"
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.missed_emergencies = 0
        self.overtriage_count = 0
        self.decision_times = []
        self.session_id = f"session_{int(time.time())}"
        self._step_start_time = None

    def reset(self, task: str = "easy") -> PatientObservation:
        self.current_task = task
        self.step_count = 0
        self.episode_count += 1
        self._step_start_time = time.time()

        scenarios = SCENARIOS.get(task, SCENARIOS["easy"])
        base = random.choice(scenarios).copy()

        tod, season = _get_time_and_season()
        extra = _season_symptoms(season, tod)
        base["symptoms"] = list(set(base["symptoms"] + extra))

        vitals = _vitals_for_urgency(base["correct_urgency"])

        # Adversarial: override vitals to contradict symptoms
        if base.get("adversarial_note"):
            vitals = VitalSigns(
                heart_rate=random.randint(120, 145),
                systolic_bp=random.randint(80, 88),
                diastolic_bp=random.randint(50, 58),
                oxygen_saturation=round(random.uniform(83.0, 89.0), 1),
                temperature=round(random.uniform(36.0, 36.8), 1),
                respiratory_rate=random.randint(28, 36)
            )

        # Fetch visit history from DB
        visit_history = self._get_visit_history(base["patient_id"])

        self.current_scenario = base
        obs = PatientObservation(
            patient_id=base["patient_id"],
            age=base["age"],
            symptoms=base["symptoms"],
            symptom_duration_hours=base["symptom_duration_hours"],
            chronic_conditions=base["chronic_conditions"],
            past_visits_30_days=base["past_visits_30_days"],
            pain_scale=base["pain_scale"],
            vitals=vitals,
            visit_history=visit_history,
            time_of_day=tod,
            season=season,
        )
        return obs

    def step(self, action: TriageAction) -> StepResult:
        if self.current_scenario is None:
            raise ValueError("Call reset() before step()")

        decision_time = time.time() - (self._step_start_time or time.time())
        self.decision_times.append(decision_time)
        self._step_start_time = time.time()

        correct = self.current_scenario["correct_urgency"]
        predicted = action.urgency_level
        reward = self._shaped_reward(correct, predicted, decision_time, action)

        self.total_reward += reward
        self.step_count += 1

        # Track errors
        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.EMERGENCY:
            self.missed_emergencies += 1
        if predicted > correct + 1:
            self.overtriage_count += 1

        # Save to DB
        self._save_visit(
            patient_id=self.current_scenario["patient_id"],
            urgency=int(predicted),
            reward=reward
        )

        done = True
        next_obs = None
        if not done:
            next_obs = self.reset(self.current_task)

        return StepResult(
            observation=PatientObservation(
                patient_id=self.current_scenario["patient_id"],
                age=self.current_scenario["age"],
                symptoms=self.current_scenario["symptoms"],
                symptom_duration_hours=self.current_scenario["symptom_duration_hours"],
                chronic_conditions=self.current_scenario["chronic_conditions"],
                past_visits_30_days=self.current_scenario["past_visits_30_days"],
                pain_scale=self.current_scenario["pain_scale"],
                vitals=_vitals_for_urgency(correct),
                visit_history=self._get_visit_history(self.current_scenario["patient_id"]),
                time_of_day=self.current_scenario.get("time_of_day", "morning"),
                season=self.current_scenario.get("season", "summer"),
            ),
            reward=round(reward, 4),
            done=done,
            info={
                "correct_urgency": int(correct),
                "predicted_urgency": int(predicted),
                "decision_time_seconds": round(decision_time, 2),
                "task": self.current_task,
                "episode": self.episode_count,
                "step": self.step_count,
            },
            next_patient=next_obs,
        )

    def state(self) -> dict:
        return {
            "episode": self.episode_count,
            "step": self.step_count,
            "task": self.current_task,
            "total_reward": round(self.total_reward, 4),
            "avg_reward": round(
                self.total_reward / max(self.episode_count, 1), 4
            ),
            "missed_emergencies": self.missed_emergencies,
            "overtriage_count": self.overtriage_count,
            "avg_decision_time": round(
                sum(self.decision_times) / max(len(self.decision_times), 1), 2
            ),
            "session_id": self.session_id,
        }

    # ── Shaped reward ─────────────────────────────────────────────────────────

    def _shaped_reward(
        self,
        correct: UrgencyLevel,
        predicted: UrgencyLevel,
        decision_time: float,
        action: TriageAction,
    ) -> float:
        diff = abs(int(correct) - int(predicted))

        # Base accuracy reward
        if diff == 0:
            reward = 1.0
        elif diff == 1:
            reward = 0.4
        else:
            reward = -0.3 * diff

        # Speed bonus for emergencies decided fast (under 30s)
        if correct == UrgencyLevel.EMERGENCY and predicted == UrgencyLevel.EMERGENCY:
            if decision_time < 30:
                reward += 0.2

        # Chronic condition awareness bonus
        if self.current_scenario.get("chronic_conditions") and diff == 0:
            reward += 0.1

        # Diagnosis prediction bonus
        if action.predicted_diagnosis and diff == 0:
            reward += 0.3

        # Resource efficiency: penalize massive overtriage
        if int(predicted) - int(correct) >= 2:
            reward -= 0.2

        # Missed emergency penalty (heavy)
        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.URGENT:
            reward -= 0.5

        return max(-1.0, min(1.6, reward))

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _save_visit(self, patient_id: str, urgency: int, reward: float):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT INTO visits VALUES (?,?,?,?,?,?)",
                (
                    self.session_id,
                    patient_id,
                    time.time(),
                    json.dumps(self.current_scenario.get("symptoms", [])),
                    urgency,
                    reward,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _get_visit_history(self, patient_id: str) -> list:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT timestamp, symptoms, urgency_assigned, reward FROM visits "
                "WHERE session_id=? AND patient_id=? ORDER BY timestamp DESC LIMIT 3",
                (self.session_id, patient_id),
            )
            rows = c.fetchall()
            conn.close()
            return [
                {
                    "timestamp": r[0],
                    "symptoms": json.loads(r[1]),
                    "urgency_assigned": r[2],
                    "reward": r[3],
                }
                for r in rows
            ]
        except Exception:
            return []

    def analytics(self) -> dict:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM visits WHERE session_id=?", (self.session_id,))
            total = c.fetchone()[0]
            c.execute("SELECT AVG(reward) FROM visits WHERE session_id=?", (self.session_id,))
            avg_r = c.fetchone()[0] or 0.0
            c.execute(
                "SELECT COUNT(*) FROM visits WHERE session_id=? AND urgency_assigned < 3",
                (self.session_id,),
            )
            missed_e = self.missed_emergencies
            c.execute(
                "SELECT symptoms, COUNT(*) as cnt FROM visits WHERE session_id=? "
                "GROUP BY symptoms ORDER BY cnt DESC LIMIT 5",
                (self.session_id,),
            )
            dist = {r[0][:30]: r[1] for r in c.fetchall()}
            conn.close()
            return {
                "total_episodes": total,
                "avg_reward": round(avg_r, 4),
                "missed_emergencies_rate": round(
                    missed_e / max(total, 1), 4
                ),
                "overtriage_rate": round(
                    self.overtriage_count / max(total, 1), 4
                ),
                "disease_distribution": dist,
                "avg_decision_time_seconds": round(
                    sum(self.decision_times) / max(len(self.decision_times), 1), 2
                ),
            }
        except Exception as e:
            return {"error": str(e)}