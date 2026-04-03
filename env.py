import random
from models import PatientObservation, TriageAction, StepResult, UrgencyLevel

PATIENT_SCENARIOS = [
    # Easy
    {"age": 25, "symptoms": ["mild fever"], "duration_hours": 12, "chronic": False, "true_urgency": 0, "disease": "Common Cold", "first_aid": "Rest, fluids, paracetamol 500mg"},
    {"age": 30, "symptoms": ["runny nose", "sneezing"], "duration_hours": 24, "chronic": False, "true_urgency": 0, "disease": "Allergic Rhinitis", "first_aid": "Antihistamine, avoid allergens"},
    {"age": 22, "symptoms": ["mild headache"], "duration_hours": 6, "chronic": False, "true_urgency": 1, "disease": "Tension Headache", "first_aid": "Rest, hydration, OTC painkiller"},
    # Medium
    {"age": 45, "symptoms": ["high fever", "body ache", "fatigue"], "duration_hours": 48, "chronic": False, "true_urgency": 2, "disease": "Dengue Fever", "first_aid": "Immediate clinic visit, stay hydrated"},
    {"age": 60, "symptoms": ["fever", "cough", "breathlessness"], "duration_hours": 36, "chronic": True, "true_urgency": 2, "disease": "Pneumonia", "first_aid": "Urgent doctor visit, no self-medication"},
    {"age": 50, "symptoms": ["chest tightness", "sweating"], "duration_hours": 2, "chronic": True, "true_urgency": 3, "disease": "Cardiac Event", "first_aid": "Call 108 immediately, do NOT move patient"},
    # Hard
    {"age": 35, "symptoms": ["severe chest pain", "left arm pain", "nausea"], "duration_hours": 1, "chronic": False, "true_urgency": 3, "disease": "Heart Attack", "first_aid": "Call 108 immediately, chew aspirin if available"},
    {"age": 70, "symptoms": ["sudden confusion", "facial drooping", "slurred speech"], "duration_hours": 1, "chronic": True, "true_urgency": 3, "disease": "Stroke", "first_aid": "Call 108 immediately, note time of symptom onset"},
    {"age": 28, "symptoms": ["severe abdominal pain", "vomiting blood"], "duration_hours": 3, "chronic": False, "true_urgency": 3, "disease": "GI Bleed", "first_aid": "Emergency room immediately"},
]

class MediTriageEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.current_scenario = None
        self.step_count = 0
        self.max_steps = 10
        self._set_task_pool()

    def _set_task_pool(self):
        if self.task_level == "easy":
            self.pool = PATIENT_SCENARIOS[:3]
        elif self.task_level == "medium":
            self.pool = PATIENT_SCENARIOS[3:6]
        else:
            self.pool = PATIENT_SCENARIOS[6:]

    def reset(self) -> PatientObservation:
        self.current_scenario = random.choice(self.pool)
        self.step_count = 0
        return PatientObservation(
            patient_id=f"P{random.randint(1000,9999)}",
            age=self.current_scenario["age"],
            symptoms=self.current_scenario["symptoms"],
            duration_hours=self.current_scenario["duration_hours"],
            has_chronic_condition=self.current_scenario["chronic"],
            past_visits=random.randint(0, 5),
            language="english"
        )

    def step(self, action: TriageAction) -> StepResult:
        self.step_count += 1
        true_urgency = self.current_scenario["true_urgency"]
        predicted = int(action.urgency_level)
        reward = self._compute_reward(predicted, true_urgency)
        done = self.step_count >= self.max_steps
        obs = self.reset() if not done else self.reset()
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={"true_urgency": true_urgency, "predicted": predicted}
        )

    def _compute_reward(self, predicted: int, true: int) -> float:
        if predicted == true:
            return 1.0
        diff = abs(predicted - true)
        if diff == 1:
            return 0.5
        if predicted < true:  # under-triaged — dangerous
            return -1.0
        return -0.3  # over-triaged — wasteful but safer

    def state(self) -> dict:
        return {
            "task_level": self.task_level,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_scenario": self.current_scenario
        }