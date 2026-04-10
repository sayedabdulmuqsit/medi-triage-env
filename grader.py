def grade(observation, action, correct_urgency, task="easy"):
    """
    Grade a triage action.
    ALWAYS returns a float strictly in (0, 1) — never 0.0 or 1.0.
    """

    def safe_score(val: float) -> float:
        """Clamp to open interval (0.001, 0.999). Never 0.0 or 1.0."""
        try:
            v = float(val)
            if v != v:  # NaN guard
                return 0.5
            # Use 0.001 / 0.999 floor/ceil — validator rejects 0.0 and 1.0
            return max(0.001, min(0.999, v))
        except (TypeError, ValueError):
            return 0.5

    def safe_int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    try:
        is_action_dict = isinstance(action, dict)
        is_obs_dict    = isinstance(observation, dict)

        predicted = safe_int(
            action.get("urgency_level", 0) if is_action_dict else getattr(action, "urgency_level", 0)
        )
        reasoning = (
            action.get("reasoning", "") if is_action_dict else getattr(action, "reasoning", "")
        )
        reasoning = reasoning if isinstance(reasoning, str) else ""

        predicted_diagnosis = (
            action.get("predicted_diagnosis", "") if is_action_dict
            else getattr(action, "predicted_diagnosis", "")
        )

        correct = safe_int(correct_urgency, 0)
        diff    = abs(correct - predicted)

        # ── easy ──────────────────────────────────────────────────────────────
        if task == "easy":
            return safe_score(0.88 if diff == 0 else 0.45 if diff == 1 else 0.05)

        # ── medium ────────────────────────────────────────────────────────────
        elif task == "medium":
            if correct == 3 and predicted < 2:
                return safe_score(0.05)
            base     = 0.75 if diff == 0 else 0.40 if diff == 1 else 0.05
            symptoms = (
                observation.get("symptoms", []) if is_obs_dict
                else getattr(observation, "symptoms", [])
            )
            symptoms = symptoms if isinstance(symptoms, list) else []
            bonus    = min(0.10, sum(
                0.03 for s in symptoms
                if isinstance(s, str) and s.lower() in reasoning.lower()
            ))
            return safe_score(base + bonus)

        # ── hard ──────────────────────────────────────────────────────────────
        elif task == "hard":
            return safe_score(0.92 if diff == 0 else 0.40 if diff == 1 else 0.05)

        # ── expert ────────────────────────────────────────────────────────────
        elif task == "expert":
            if diff == 0:
                s_val = 0.85
                if predicted_diagnosis:
                    s_val += 0.05
                if len(reasoning) > 50:
                    s_val += 0.04
                return safe_score(s_val)
            return safe_score(0.45 if diff == 1 else 0.05)

        # ── adversarial ───────────────────────────────────────────────────────
        elif task == "adversarial":
            vitals = (
                observation.get("vitals", {}) if is_obs_dict
                else getattr(observation, "vitals", None)
            )
            v_is_dict = isinstance(vitals, dict)
            critical  = False
            if vitals:
                o2  = safe_int(vitals.get("oxygen_saturation",  100) if v_is_dict else getattr(vitals, "oxygen_saturation",  100), 100)
                hr  = safe_int(vitals.get("heart_rate",          80) if v_is_dict else getattr(vitals, "heart_rate",          80),  80)
                sbp = safe_int(vitals.get("systolic_bp",        120) if v_is_dict else getattr(vitals, "systolic_bp",        120), 120)
                rr  = safe_int(vitals.get("respiratory_rate",    16) if v_is_dict else getattr(vitals, "respiratory_rate",    16),  16)
                critical = (o2 < 92 or hr > 120 or sbp < 90 or rr > 28)

            if critical and predicted == 3:
                return safe_score(0.95)
            return safe_score(0.70 if diff == 0 else 0.05)

        # ── unknown task fallback ─────────────────────────────────────────────
        return safe_score(0.50)

    except Exception:
        return 0.5  # safe fallback — never 0.0 or 1.0