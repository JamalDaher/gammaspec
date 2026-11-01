"""Pure helpers for session-scoped Learning Lab progress and numeric challenges."""
import csv
import io
import math

CHALLENGES = {
    "Poisson relative uncertainty": {
        "prompt": "A peak contains 400 counts. What is its approximate relative Poisson uncertainty (%)?",
        "answer": 5.0, "tolerance": 0.15, "unit": "%",
        "explanation": "Relative Poisson uncertainty is 100/sqrt(N) percent, or 5% for 400 counts.",
    },
    "Two half-value layers": {
        "prompt": "What percent of an uncollided narrow beam remains after 2 HVLs?",
        "answer": 25.0, "tolerance": 0.15, "unit": "%",
        "explanation": "Transmission is 100*(1/2)^2 = 25%. This neglects scattered photons and buildup.",
    },
}

def grade_numeric(name, response):
    if name not in CHALLENGES:
        raise ValueError("Unknown challenge")
    value = float(response)
    if not math.isfinite(value):
        raise ValueError("Answer must be finite")
    question = CHALLENGES[name]
    return abs(value - question["answer"]) <= question["tolerance"]

def record_attempt(progress, topic, correct):
    """Return a new progress mapping; do not mutate Streamlit session data."""
    if not topic or not isinstance(correct, bool):
        raise ValueError("Invalid attempt")
    updated = {key: dict(value) for key, value in progress.items()}
    item = updated.setdefault(topic, {"attempts": 0, "correct": 0, "completed": False})
    item["attempts"] += 1
    item["correct"] += int(correct)
    item["completed"] = item["completed"] or correct
    return updated

def progress_summary(progress, topics):
    """Completed means answered correctly at least once; accuracy is per attempt."""
    topics = list(topics)
    completed = sum(bool(progress.get(t, {}).get("completed", False)) for t in topics)
    attempts = sum(int(progress.get(t, {}).get("attempts", 0)) for t in topics)
    correct = sum(int(progress.get(t, {}).get("correct", 0)) for t in topics)
    return {"completed": completed, "total": len(topics), "attempts": attempts,
            "correct": correct, "accuracy_percent": 100 * correct / attempts if attempts else 0.0}

def progress_csv(progress, topics):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Topic", "Completed", "Attempts", "Correct", "Accuracy (%)"])
    for topic in topics:
        item = progress.get(topic, {})
        attempts = int(item.get("attempts", 0))
        correct = int(item.get("correct", 0))
        writer.writerow([topic, bool(item.get("completed", False)), attempts, correct,
                         round(100 * correct / attempts, 2) if attempts else 0.0])
    return output.getvalue()
