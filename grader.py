from env import MediTriageEnv
from models import TriageAction, UrgencyLevel

def run_grader(task_level: str, episodes: int = 10) -> float:
    env = MediTriageEnv(task_level=task_level)
    total_reward = 0.0

    for _ in range(episodes):
        obs = env.reset()
        # Simple rule-based baseline agent
        urgency = _rule_based_agent(obs)
        action = TriageAction(
            urgency_level=urgency,
            predicted_disease="Unknown",
            first_aid_suggestion="Consult a doctor",
            alert_doctor=(urgency == UrgencyLevel.EMERGENCY)
        )
        result = env.step(action)
        total_reward += result.reward

    score = max(0.0, min(1.0, (total_reward / episodes + 1) / 2))
    return round(score, 3)

def _rule_based_agent(obs) -> UrgencyLevel:
    symptoms = [s.lower() for s in obs.symptoms]
    emergency_keywords = ["chest pain", "stroke", "heart", "unconscious", "bleeding", "facial drooping", "slurred"]
    urgent_keywords = ["high fever", "breathlessness", "vomiting", "severe"]

    for kw in emergency_keywords:
        if any(kw in s for s in symptoms):
            return UrgencyLevel.EMERGENCY
    for kw in urgent_keywords:
        if any(kw in s for s in symptoms):
            return UrgencyLevel.URGENT
    if obs.has_chronic_condition and obs.duration_hours > 24:
        return UrgencyLevel.URGENT
    if obs.duration_hours > 48:
        return UrgencyLevel.NON_URGENT
    return UrgencyLevel.SELF_CARE

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        score = run_grader(level)
        print(f"Task [{level}] Score: {score}")