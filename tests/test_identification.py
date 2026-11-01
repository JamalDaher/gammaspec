import pytest
from gammaspec.identification import match_candidates

def test_cs137_candidate():
    result = match_candidates([662.1])
    assert result.iloc[0]["candidate"] == "Cs-137"

def test_co60_both_lines():
    result = match_candidates([1172.8, 1333.1])
    co = result.set_index("candidate").loc["Co-60"]
    assert co.matched_lines == 2
    assert co.coverage == 1

def test_no_matches():
    assert match_candidates([222.0], tolerance_kev=2).empty

def test_tolerance():
    assert match_candidates([670.0], tolerance_kev=5).empty
    assert "Cs-137" in match_candidates([670.0], tolerance_kev=10).candidate.tolist()

def test_one_to_one_matching():
    result = match_candidates([661.1, 662.2])
    cs = result.set_index("candidate").loc["Cs-137"]
    assert cs.matched_lines == 1

def test_invalid_tolerance():
    with pytest.raises(ValueError):
        match_candidates([661.7], tolerance_kev=0)

def test_invalid_energy():
    with pytest.raises(ValueError):
        match_candidates([float("nan")])

def test_empty_peaks():
    assert match_candidates([]).empty
