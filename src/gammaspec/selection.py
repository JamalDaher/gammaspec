"""Interpret selected candidate marker indices."""

def selected_peak_energies(points, energies):
    result = []
    for point in points:
        index = point.get("point_index")
        if point.get("curve_number") != 1 or type(index) is not int or not 0 <= index < len(energies):
            continue
        energy = float(energies[index])
        if energy not in result:
            result.append(energy)
        if len(result) == 2:
            break
    return sorted(result)
