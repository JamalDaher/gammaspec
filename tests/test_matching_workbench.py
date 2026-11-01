import pytest
from gammaspec.matching_workbench import rank_reference_candidates

def test_two_line_candidate_and_residuals():
    ranks, detail = rank_reference_candidates([1173.0, 1333.0])
    assert ranks.iloc[0].candidate == "Co-60"
    co = ranks.set_index("candidate").loc["Co-60"]
    assert co.matched_lines == 2
    assert co.coverage == 1
    assert len(detail[detail.candidate == "Co-60"]) == 2

def test_unmatched_candidates_not_claimed():
    ranks, detail = rank_reference_candidates([250.0], tolerance_kev=2)
    assert (ranks.matched_lines == 0).all()
    assert (ranks.ranking_score == 0).all()
    assert detail.empty

def test_one_to_one_and_optimal_cardinality():
    ranks, detail = rank_reference_candidates([100, 108], tolerance_kev=9,
        library={"test": [104, 100]})
    assert ranks.iloc[0].matched_lines == 2
    assert len(detail) == 2
    assert detail.observed_kev.nunique() == 2

def test_duplicate_peaks_do_not_double_match():
    ranks, detail = rank_reference_candidates([661.0, 662.0])
    assert ranks.set_index("candidate").loc["Cs-137"].matched_lines == 1
    assert len(detail[detail.candidate == "Cs-137"]) == 1

def test_invalid_inputs():
    with pytest.raises(ValueError):
        rank_reference_candidates([float("nan")])
    with pytest.raises(ValueError):
        rank_reference_candidates([1], tolerance_kev=0)
