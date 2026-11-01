import csv
import io
import pytest
from gammaspec.learning_progress import (
    CHALLENGES, grade_numeric, record_attempt, progress_summary, progress_csv,
)

def test_attempts_and_completion():
    original = {}
    first = record_attempt(original, "Photopeak identification", False)
    second = record_attempt(first, "Photopeak identification", True)
    assert original == {}
    assert first["Photopeak identification"]["completed"] is False
    assert second["Photopeak identification"] == {"attempts": 2, "correct": 1, "completed": True}
    summary = progress_summary(second, ["Photopeak identification", "Counting statistics"])
    assert summary == {"completed": 1, "total": 2, "attempts": 2, "correct": 1, "accuracy_percent": 50.0}

def test_zero_attempts_and_csv():
    assert progress_summary({}, ["A"])["accuracy_percent"] == 0
    rows = list(csv.DictReader(io.StringIO(progress_csv({}, ["A"]))))
    assert rows[0]["Topic"] == "A"
    assert rows[0]["Attempts"] == "0"

@pytest.mark.parametrize("topic", list(CHALLENGES))
def test_numeric_challenges(topic):
    answer = CHALLENGES[topic]["answer"]
    assert grade_numeric(topic, answer)
    assert not grade_numeric(topic, answer + 5)

def test_invalid_attempt_and_numeric():
    with pytest.raises(ValueError):
        record_attempt({}, "", True)
    with pytest.raises(ValueError):
        grade_numeric("No such challenge", 1)
    with pytest.raises(ValueError):
        grade_numeric("Poisson relative uncertainty", float("nan"))
