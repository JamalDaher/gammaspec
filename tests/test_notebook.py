import json
import pytest
from gammaspec.notebook import create_experiment_record, notebook_json, load_notebook_json

def example():
    return create_experiment_record(title="  Cs-137 demo ", notes="Check peak",
        dataset="Synthetic Cs-137", origin="Simulated",
        acquisition={"seed": 42, "live_time_s": 60},
        calibration={"applied": False}, analysis={"candidate_maxima": 1},
        recorded_at="2026-11-01T12:00:00+00:00")

def test_record_round_trip():
    record = example()
    assert record["title"] == "Cs-137 demo"
    assert load_notebook_json(notebook_json(record)) == record
    assert json.loads(notebook_json(record))["schema_version"] == 1

def test_missing_title():
    with pytest.raises(ValueError):
        create_experiment_record(title="", notes="", dataset="x", origin="x",
            acquisition={}, calibration={}, analysis={})

def test_reject_unknown_schema():
    record = example()
    record["schema_version"] = 99
    with pytest.raises(ValueError):
        load_notebook_json(json.dumps(record))

def test_invalid_json():
    with pytest.raises(ValueError):
        load_notebook_json("not json")
