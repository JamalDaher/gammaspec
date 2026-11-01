from gammaspec.selection import selected_peak_energies


def test_select_two_markers():
    points = [{"curve_number": 1, "point_index": 2}, {"curve_number": 1, "point_index": 0}]
    assert selected_peak_energies(points, [485, 500, 505]) == [485.0, 505.0]


def test_ignore_other_traces_and_invalid_indices():
    points = [{"curve_number": 0, "point_index": 0}, {"curve_number": 1, "point_index": -1}, {"curve_number": 1, "point_index": 1}]
    assert selected_peak_energies(points, [485, 505]) == [505.0]
