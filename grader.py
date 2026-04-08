from models import UrgencyLevel, TriageAction, PatientObservation


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, round(score, 4)))


def grade(
    observation: PatientObservation,
    action: TriageAction,
    correct_urgency: UrgencyLevel,
    task: str = "easy",
) -> float:
    predicted = action.urgency_level
    correct = correct_urgency
    diff = abs(int(correct) - int(predicted))

    if diff == 0:
        base = 1.0
    elif diff == 1:
        base = 0.5
    elif diff == 2:
        base = 0.2
    else:
        base = 0.0

    if task == "easy":
        if diff == 0:
            return _clamp(0.90)
        elif diff == 1:
            return _clamp(0.50)
        return _clamp(0.05)

    elif task == "medium":
        score = base
        reasoning_lower = action.reasoning.lower()
        symptom_keywords = [s.lower() for s in observation.symptoms]
        matched = sum(1 for kw in symptom_keywords if kw in reasoning_lower)
        reasoning_bonus = min(0.15, matched * 0.05)

        vitals = observation.vitals
        vitals_match = _vitals_match_urgency(vitals, correct)
        vital_bonus = 0.10 if vitals_match else 0.0

        chronic_bonus = 0.0
        if observation.chronic_conditions and diff <= 1:
            if any(c.lower() in reasoning_lower for c in observation.chronic_conditions):
                chronic_bonus = 0.10

        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.URGENT:
            return _clamp(0.05)

        score = min(0.99, score + reasoning_bonus + vital_bonus + chronic_bonus)
        result = 0.60 + (score - 0.60) * 0.85 if score >= 0.60 else score
        return _clamp(result)

    elif task == "hard":
        if diff == 0:
            return _clamp(0.95)
        elif diff == 1:
            return _clamp(0.40)
        return _clamp(0.05)

    elif task == "expert":
        if diff == 0:
            score = 0.88
            if action.predicted_diagnosis:
                score += 0.05
            if len(action.reasoning) > 50:
                score += 0.04
            return _clamp(score)
        elif diff == 1:
            return _clamp(0.45)
        return _clamp(0.05)

    elif task == "adversarial":
        vitals = observation.vitals
        reasoning_lower = action.reasoning.lower()

        critical_vitals = (
            vitals.oxygen_saturation < 92
            or vitals.heart_rate > 120
            or vitals.systolic_bp < 90
            or vitals.respiratory_rate > 28
        )

        if critical_vitals and predicted == UrgencyLevel.EMERGENCY:
            base_score = 0.88
            if any(word in reasoning_lower for word in ["vital", "spo2", "oxygen", "heart rate", "bp", "pressure"]):
                base_score += 0.09
            return _clamp(base_score)
        elif critical_vitals and diff == 0:
            return _clamp(0.70)
        elif diff == 0:
            return _clamp(0.80)
        return _clamp(0.05)

    return _clamp(base)


def _vitals_match_urgency(vitals, urgency: UrgencyLevel) -> bool:
    hr = vitals.heart_rate
    spo2 = vitals.oxygen_saturation
    rr = vitals.respiratory_rate
    temp = vitals.temperature
    sbp = vitals.systolic_bp

    if urgency == UrgencyLevel.EMERGENCY:
        return hr > 110 or spo2 < 92 or rr > 25 or temp > 39.5 or sbp < 90 or sbp > 180
    elif urgency == UrgencyLevel.URGENT:
        return (90 <= hr <= 120) or (92 <= spo2 <= 95) or (20 <= rr <= 27)
    elif urgency == UrgencyLevel.NON_URGENT:
        return 72 <= hr <= 100 and spo2 >= 95
    else:
        return 60 <= hr <= 80 and spo2 >= 97