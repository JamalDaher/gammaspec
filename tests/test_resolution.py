import pytest
from gammaspec.resolution import resolution_study, summarize_resolution

def test_resolution_study_reproducible_and_grouped():
    a = resolution_study([15, 25], trials=2, seed=9)
    b = resolution_study([15, 25], trials=2, seed=9)
    assert a.equals(b)
    summary = summarize_resolution(a)
    assert summary.trials.tolist() == [2, 2]
    assert summary.recovery_rate.between(0, 1).all()

def test_reject_invalid_parameters():
    with pytest.raises(ValueError):
        resolution_study([0], trials=2)
    with pytest.raises(ValueError):
        resolution_study([20], trials=1)
