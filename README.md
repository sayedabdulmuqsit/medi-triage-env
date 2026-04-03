---
title: MediGuide RL
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MediGuide RL — Healthcare Triage RL Environment

A real-world OpenEnv environment where an AI agent learns to triage patients by urgency level.

## Action Space
- 0: Self-Care
- 1: Non-Urgent
- 2: Urgent
- 3: Emergency

## Observation Space
Patient age, symptoms, duration, chronic condition, past visits

## Tasks
- **Easy:** Single symptom, healthy patient
- **Medium:** Multi-symptom with chronic conditions
- **Hard:** Critical/ambiguous emergency scenarios

## API Endpoints
- `POST /reset/{task_level}` — Start new episode
- `POST /step/{task_level}` — Take action, get reward
- `GET /state/{task_level}` — Current environment state
- `GET /docs` — Interactive API docs

## Reward
- Correct triage: +1.0
- Off by one level: +0.5
- Over-triaged: -0.3
- Under-triaged (dangerous): -1.0

## Setup
```bash
pip install -r requirements.txt
python server.py
```