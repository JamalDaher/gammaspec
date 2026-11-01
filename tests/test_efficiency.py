import numpy as np
import pytest
from gammaspec.efficiency import peak_efficiency, counting_statistics

def test_reference_efficiency_and_bounds():
    assert peak_efficiency(662, "HPGe (illustrative)") == pytest.approx(0.08)
    curve = peak_efficiency(np.array([100, 662, 1600]), "NaI(Tl) (illustrative)")
    assert np.all((curve > 0) & (curve <= 1))
    assert np.all(np.diff(curve) < 0)

def test_poisson_statistics_scale_with_time():
    a = counting_statistics(662, "HPGe (illustrative)", 100, 10, 2)
    b = counting_statistics(662, "HPGe (illustrative)", 100, 40, 2)
    assert a["signal_counts"] == pytest.approx(80)
    assert a["background_counts"] == pytest.approx(20)
    assert a["poisson_sigma_counts"] == pytest.approx(10)
    assert b["approx_snr"] == pytest.approx(2 * a["approx_snr"])

def test_zero_counts_and_invalid_inputs():
    assert counting_statistics(662, "HPGe (illustrative)", 0, 10)["approx_snr"] == 0
    with pytest.raises(ValueError):
        peak_efficiency(0, "HPGe (illustrative)")
    with pytest.raises(ValueError):
        counting_statistics(662, "HPGe (illustrative)", -1, 10)
