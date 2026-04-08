from models import UrgencyLevel, TriageAction, PatientObservation


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, round(score, 4)))


def grade(observation, action, correct_urgency, task="easy") -> float:
    predicted = action.urgency_level
    correct = correct_urgency
    diff = abs(int(correct) - int(predicted))

    if task == "easy":
        return _clamp(0.88) if diff == 0 else _clamp(0.45) if diff == 1 else _clamp(0.05)

    elif task == "medium":
        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.URGENT:
            return _clamp(0.05)
        base = 0.75 if diff == 0 else 0.40 if diff == 1 else 0.05
        reasoning_lower = action.reasoning.lower()
        bonus = min(0.10, sum(0.03 for s in observation.symptoms if s.lower() in reasoning_lower))
        return _clamp(base + bonus)

    elif task == "hard":
        return _clamp(0.92) if diff == 0 else _clamp(0.40) if diff == 1 else _clamp(0.05)

    elif task == "expert":
        if diff == 0:
            s = 0.85
            if action.predicted_diagnosis:
                s += 0.05
            if len(action.reasoning) > 50:
                s += 0.04
            return _clamp(s)
        return _clamp(0.45) if diff == 1 else _clamp(0.05)

    elif task == "adversarial":
        v = observation.vitals
        critical = (v.oxygen_saturation < 92 or v.heart_rate > 120
                    or v.systolic_bp < 90 or v.respiratory_rate > 28)
        if critical and predicted == UrgencyLevel.EMERGENCY:
            return _clamp(0.95)
        return _clamp(0.70) if diff == 0 else _clamp(0.05)

    return _clamp(0.50)