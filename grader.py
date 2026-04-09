def grade(observation, action, correct_urgency, task="easy"):
    # Inline all dependencies to survive strict AST sandboxing
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

    # Inline clamp equivalent
    def safe_score(val):
        return max(0.01, min(0.99, round(float(val), 4)))

    if task == "easy":
        return safe_score(0.88) if diff == 0 else safe_score(0.45) if diff == 1 else safe_score(0.05)

    elif task == "medium":
        # 3 = EMERGENCY, 2 = URGENT
        if int(correct) == 3 and int(predicted) < 2:
            return safe_score(0.05)
        base = 0.75 if diff == 0 else 0.40 if diff == 1 else 0.05
        symptoms = observation.get("symptoms", []) if is_obs_dict else getattr(observation, "symptoms", [])
        symptoms = symptoms or []
        bonus = min(0.10, sum(0.03 for s in symptoms if isinstance(s, str) and s.lower() in reasoning.lower()))
        return safe_score(base + bonus)

    elif task == "hard":
        return safe_score(0.92) if diff == 0 else safe_score(0.40) if diff == 1 else safe_score(0.05)

    elif task == "expert":
        if diff == 0:
            s_val = 0.85
            if predicted_diagnosis:
                s_val += 0.05
            if len(reasoning) > 50:
                s_val += 0.04
            return safe_score(s_val)
        return safe_score(0.45) if diff == 1 else safe_score(0.05)

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
            
        if critical and int(predicted) == 3:
            return safe_score(0.95)
        return safe_score(0.70) if diff == 0 else safe_score(0.05)

    return safe_score(0.50)