from models import UrgencyLevel, TriageAction, PatientObservation


def grade(
    observation: PatientObservation,
    action: TriageAction,
    correct_urgency: UrgencyLevel,
    task: str = "easy",
) -> float:
    """
    Returns a score in [0.0, 1.0].
    """
    predicted = action.urgency_level
    correct = correct_urgency
    diff = abs(int(correct) - int(predicted))

    # ── Base score ────────────────────────────────────────────────────────────
    if diff == 0:
        base = 1.0
    elif diff == 1:
        base = 0.5
    elif diff == 2:
        base = 0.2
    else:
        base = 0.0

    # ── Task-specific adjustments ─────────────────────────────────────────────

    if task == "easy":
        # Simple: exact match = 0.90, one-off = 0.50
        if diff == 0:
            return 0.90
        elif diff == 1:
            return 0.50
        return 0.0

    elif task == "medium":
        # Improved medium grader — rewards correct reasoning & vital alignment
        score = base

        # Bonus: reasoning mentions key symptom words
        reasoning_lower = action.reasoning.lower()
        symptom_keywords = [s.lower() for s in observation.symptoms]
        matched = sum(1 for kw in symptom_keywords if kw in reasoning_lower)
        reasoning_bonus = min(0.15, matched * 0.05)

        # Bonus: vital signs alignment
        vitals = observation.vitals
        vitals_match = _vitals_match_urgency(vitals, correct)
        vital_bonus = 0.10 if vitals_match else 0.0

        # Bonus: chronic condition awareness
        chronic_bonus = 0.0
        if observation.chronic_conditions and diff <= 1:
            if any(
                c.lower() in reasoning_lower
                for c in observation.chronic_conditions
            ):
                chronic_bonus = 0.10

        # Penalty: completely ignoring emergency vitals
        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.URGENT:
            return 0.0

        score = min(1.0, score + reasoning_bonus + vital_bonus + chronic_bonus)
        # Normalize to 0.60-0.85 range for medium
        return round(0.60 + (score - 0.60) * 0.85, 3) if score >= 0.60 else round(score, 3)

    elif task == "hard":
        if diff == 0:
            return 0.95
        elif diff == 1:
            return 0.40
        return 0.0

    elif task == "expert":
        # Mass casualty: full credit only for perfect triage + fast reasoning
        if diff == 0:
            score = 0.90
            if action.predicted_diagnosis:
                score += 0.05
            if len(action.reasoning) > 50:
                score += 0.05
            return min(1.0, score)
        elif diff == 1:
            return 0.45
        return 0.0

    elif task == "adversarial":
        # Must catch contradicting vitals vs symptoms
        vitals = observation.vitals
        reasoning_lower = action.reasoning.lower()

        # Critical vitals: SpO2 < 92, HR > 120, BP systolic < 90
        critical_vitals = (
            vitals.oxygen_saturation < 92
            or vitals.heart_rate > 120
            or vitals.systolic_bp < 90
            or vitals.respiratory_rate > 28
        )

        if critical_vitals and predicted == UrgencyLevel.EMERGENCY:
            base_score = 0.90
            # Bonus if reasoning mentions vitals
            if any(
                word in reasoning_lower
                for word in ["vital", "spo2", "oxygen", "heart rate", "bp", "pressure"]
            ):
                base_score += 0.10
            return min(1.0, base_score)
        elif critical_vitals and diff == 0:
            return 0.70
        elif diff == 0:
            return 0.80
        return 0.0

    # Fallback
    return round(base, 3)


def _vitals_match_urgency(vitals, urgency: UrgencyLevel) -> bool:
    """Check if vitals are consistent with expected urgency level."""
    hr = vitals.heart_rate
    spo2 = vitals.oxygen_saturation
    rr = vitals.respiratory_rate
    temp = vitals.temperature
    sbp = vitals.systolic_bp

    if urgency == UrgencyLevel.EMERGENCY:
        return (
            hr > 110 or spo2 < 92 or rr > 25 or temp > 39.5 or sbp < 90 or sbp > 180
        )
    elif urgency == UrgencyLevel.URGENT:
        return (
            (90 <= hr <= 120) or (92 <= spo2 <= 95) or (20 <= rr <= 27)
        )
    elif urgency == UrgencyLevel.NON_URGENT:
        return 72 <= hr <= 100 and spo2 >= 95
    else:
        return 60 <= hr <= 80 and spo2 >= 97