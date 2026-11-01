import numpy as np
import pytest
from gammaspec.attenuation import MATERIALS, linear_attenuation, transmission, attenuation_lengths

@pytest.mark.parametrize("material", list(MATERIALS))
def test_transmission_and_hvl(material):
    mu = linear_attenuation(material, 662)
    assert mu > 0
    assert transmission(material, 662, 0) == pytest.approx(1)
    hvl = attenuation_lengths(material, 662)["hvl_cm"]
    tvl = attenuation_lengths(material, 662)["tvl_cm"]
    assert transmission(material, 662, hvl) == pytest.approx(0.5)
    assert transmission(material, 662, tvl) == pytest.approx(0.1)
    assert np.all(np.diff(transmission(material, 662, np.linspace(0, 10, 20))) < 0)

def test_invalid_values():
    with pytest.raises(ValueError):
        transmission("Lead", 662, -1)
    with pytest.raises(ValueError):
        linear_attenuation("Lead", 0)
    with pytest.raises(ValueError):
        linear_attenuation("Unknown", 662)
