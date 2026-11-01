from io import BytesIO
import json
import zipfile
import pandas as pd
import pytest
from gammaspec.notebook import create_experiment_record
from gammaspec.project_archive import export_experiment_archive, import_experiment_archive

def fixture():
    record = create_experiment_record(title="Demo", notes="Observed 662 keV",
        dataset="Cs-137", origin="Synthetic", acquisition={"seed": 42},
        calibration={"applied": False}, analysis={"total_counts": 30},
        recorded_at="2026-11-01T12:00:00+00:00")
    spectrum = pd.DataFrame({"channel": [0, 1, 2],
        "energy_kev": [0., 1., 2.], "counts": [5, 10, 15]})
    return record, spectrum

def test_round_trip_with_integrity():
    record, spectrum = fixture()
    payload = export_experiment_archive(record, spectrum)
    loaded, loaded_spectrum, manifest = import_experiment_archive(payload)
    assert loaded == record
    pd.testing.assert_frame_equal(loaded_spectrum, spectrum, check_dtype=False)
    assert manifest["files"]["spectrum.csv"]["sha256"]

def test_detect_modified_member():
    record, spectrum = fixture()
    archive_bytes = export_experiment_archive(record, spectrum)
    with zipfile.ZipFile(BytesIO(archive_bytes)) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    entries["spectrum.csv"] = entries["spectrum.csv"].replace(b"15", b"16")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for name, value in entries.items():
            z.writestr(name, value)
    with pytest.raises(ValueError, match="Integrity"):
        import_experiment_archive(stream.getvalue())

def test_reject_unknown_files_and_invalid_archive():
    with pytest.raises(ValueError):
        import_experiment_archive(b"not a zip")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("../unexpected", "hello")
    with pytest.raises(ValueError):
        import_experiment_archive(stream.getvalue())
