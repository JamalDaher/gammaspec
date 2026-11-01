import numpy as np
import pytest
from gammaspec.monte_carlo import monte_carlo_recovery, recovery_summary

def test_monte_carlo_is_reproducible():
    a = monte_carlo_recovery(trials=3, live_time_s=30, seed=11)
    b = monte_carlo_recovery(trials=3, live_time_s=30, seed=11)
    assert a.equals(b)
    assert len(a) == 3
    assert set(a.status).issubset({"ok", "failed"})
    summary = recovery_summary(a)
    assert summary["trials"] == 3
    assert 0 <= summary["success_rate"] <= 1

def test_rejects_invalid_trials():
    with pytest.raises(ValueError):
        monte_carlo_recovery(trials=1)
