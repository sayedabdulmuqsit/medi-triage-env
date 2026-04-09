from models import UrgencyLevel


def _clamp(score):
    return max(0.01, min(0.99, round(float(score), 4)))


def grade(observation, action, correct_urgency, task="easy"):
    is_action_dict = isinstance(action, dict)
    is_obs_dict = isinstance(observation, dict)

    predicted = action.get("urgency_level", 0) if is_action_dict else getattr(action, "urgency_level", 0)
    reasoning = action.get("reasoning", "") if is_action_dict else getattr(action, "reasoning", "")
    reasoning = reasoning or ""
    predicted_diagnosis = action.get("predicted_diagnosis", "") if is_action_dict else getattr(action, "predicted_diagnosis", "")

    correct = correct_urgency
    try:
        diff = abs(int(correct) - int(predicted))
    except (TypeError, ValueError):
        diff = 3

    if task == "easy":
        return _clamp(0.88) if diff == 0 else _clamp(0.45) if diff == 1 else _clamp(0.05)

    elif task == "medium":
        if correct == UrgencyLevel.EMERGENCY and predicted < UrgencyLevel.URGENT:
            return _clamp(0.05)
        base = 0.75 if diff == 0 else _clamp(0.40) if diff == 1 else _clamp(0.05)
        symptoms = observation.get("symptoms", []) if is_obs_dict else getattr(observation, "symptoms", [])
        symptoms = symptoms or []
        bonus = min(0.10, sum(0.03 for s in symptoms if isinstance(s, str) and s.lower() in reasoning.lower()))
        return _clamp(base + bonus)

    elif task == "hard":
        return _clamp(0.92) if diff == 0 else _clamp(0.40) if diff == 1 else _clamp(0.05)

    elif task == "expert":
        if diff == 0:
            s = 0.85
            if predicted_diagnosis:
                s += 0.05
            if len(reasoning) > 50:
                s += 0.04
            return _clamp(s)
        return _clamp(0.45) if diff == 1 else _clamp(0.05)

    elif task == "adversarial":
        vitals = observation.get("vitals", {}) if is_obs_dict else getattr(observation, "vitals", None)
        v_is_dict = isinstance(vitals, dict)
        if vitals:
            o2 = vitals.get("oxygen_saturation", 100) if v_is_dict else getattr(vitals, "oxygen_saturation", 100)
            hr = vitals.get("heart_rate", 80) if v_is_dict else getattr(vitals, "heart_rate", 80)
            sbp = vitals.get("systolic_bp", 120) if v_is_dict else getattr(vitals, "systolic_bp", 120)
            rr = vitals.get("respiratory_rate", 16) if v_is_dict else getattr(vitals, "respiratory_rate", 16)
            critical = (o2 < 92 or hr > 120 or sbp < 90 or rr > 28)
        else:
            critical = False
            
        if critical and predicted == UrgencyLevel.EMERGENCY:
            return _clamp(0.95)
        return _clamp(0.70) if diff == 0 else _clamp(0.05)

    return _clamp(0.50)