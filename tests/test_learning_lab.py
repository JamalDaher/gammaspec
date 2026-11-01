import pytest
from gammaspec.learning_lab import EXERCISES, check_answer, lab_example

@pytest.mark.parametrize("name", list(EXERCISES))
def test_answer_check_and_example(name):
    exercise = EXERCISES[name]
    assert check_answer(name, exercise["answer"])
    assert len(lab_example(name)) > 10
    assert exercise["explanation"]
    for wrong in exercise["choices"]:
        if wrong != exercise["answer"]:
            assert not check_answer(name, wrong)

def test_unknown_exercise_and_answer():
    with pytest.raises(ValueError):
        check_answer("Unknown", "A")
    with pytest.raises(ValueError):
        check_answer("Compton edge", "Unlisted choice")
    with pytest.raises(ValueError):
        lab_example("Unknown")
