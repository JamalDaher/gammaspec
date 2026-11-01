import pytest
from gammaspec.importers import import_spectrum_text

def test_csv_energy_axis():
    s = import_spectrum_text("channel,energy_kev,counts\n0,1,3\n1,2,4\n")
    assert s.counts.tolist() == [3, 4]

def test_csv_derived_energy():
    s = import_spectrum_text("channel,counts\n0,3\n1,4\n", energy_per_channel_kev=0.5)
    assert s.energy_kev.tolist() == [0, 0.5]

def test_plain_counts():
    s = import_spectrum_text("3\n4\n5\n", energy_per_channel_kev=2)
    assert s.channel.tolist() == [0, 1, 2]

def test_missing_energy_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text("3\n4\n")

def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text("3\n-1\n", energy_per_channel_kev=1)

def test_fractional_counts_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text("3.5\n4\n", energy_per_channel_kev=1)

def test_nonmonotonic_channels_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text("channel,energy_kev,counts\n1,2,3\n0,3,4\n")

def test_binary_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text(b"\xff\xfe")

def test_invalid_axis_rejected():
    with pytest.raises(ValueError):
        import_spectrum_text("1\n2\n", energy_per_channel_kev=-1)
