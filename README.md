---
title: MediGuide RL
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MediGuide RL — Healthcare Triage RL Environment

An OpenEnv RL environment where an AI agent learns to triage patients by urgency level, addressing India's healthcare resource allocation crisis.

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode `{"task": "easy"}` |
| POST | `/step` | Submit triage action, get reward |
| GET | `/state` | Current environment stats |
| GET | `/analytics` | Training analytics |
| GET | `/translate?text=...` | Hindi → English translation |
| GET | `/tasks` | List all task levels |
| GET | `/docs` | Interactive API docs |

## Tasks & Scoring

| Task | Correct | Off-by-one | Wrong |
|------|---------|------------|-------|
| easy | 0.88 | 0.45 | 0.05 |
| medium | 0.75+ | 0.40 | 0.05 |
| hard | 0.92 | 0.40 | 0.05 |
| expert | 0.85+ | 0.45 | 0.05 |
| adversarial | 0.95 (critical+correct) | — | 0.05 |

Medium bonuses: +0.03 per symptom keyword in reasoning (max +0.10)
Expert bonuses: +0.05 for diagnosis, +0.04 for reasoning > 50 chars

## Action Space

| Level | Label |
|-------|-------|
| 0 | Self-Care |
| 1 | Non-Urgent |
| 2 | Urgent |
| 3 | Emergency |

## Observation Space

- `patient_id`, `age`, `symptoms`, `symptom_duration_hours`
- `chronic_conditions`, `past_visits_30_days`, `pain_scale`
- `vitals`: HR, BP, SpO2, temperature, respiratory rate
- `visit_history` (SQLite-backed session memory)
- `time_of_day`, `season` (stochastic context)

## Reward Range

All rewards strictly in **(0.001, 0.999)**. Never 0.0 or 1.0.

## Features

- 🇮🇳 Hindi symptom translation (25+ keywords)
- 🧠 SQLite session memory (visit history per patient)
- ⚡ Shaped rewards with symptom/diagnosis/reasoning bonuses
- 🎲 Stochastic patients by season and time-of-day
- ⚔️ Adversarial task (trust vitals over reported symptoms)
- 🏥 5 difficulty levels from easy to adversarial