"""Portable, versioned educational experiment notebook records."""
import json
from datetime import datetime, timezone
import numpy as np

SCHEMA_VERSION = 1

def create_experiment_record(*, title, notes, dataset, origin, acquisition,
                             calibration, analysis, recorded_at=None):
    if not isinstance(title, str) or not title.strip() or len(title) > 160:
        raise ValueError("Experiment title must be 1–160 characters")
    if not isinstance(notes, str) or len(notes) > 20000:
        raise ValueError("Notes must be text no longer than 20000 characters")
    if recorded_at is None:
        recorded_at = datetime.now(timezone.utc).isoformat()
    record = {"schema_version": SCHEMA_VERSION, "title": title.strip(),
              "recorded_at_utc": recorded_at, "notes": notes,
              "dataset": dataset, "origin": origin, "acquisition": acquisition,
              "calibration": calibration, "analysis": analysis,
              "disclaimer": "Educational analysis; not a certified measurement or radiation safety record."}
    validate_experiment_record(record)
    return record

def validate_experiment_record(record):
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported experiment notebook schema")
    required = ("title", "recorded_at_utc", "notes", "dataset", "origin",
                "acquisition", "calibration", "analysis")
    if any(key not in record for key in required):
        raise ValueError("Incomplete experiment notebook record")
    if not isinstance(record["title"], str) or not record["title"].strip():
        raise ValueError("Missing experiment title")
    if not isinstance(record["notes"], str):
        raise ValueError("Invalid experiment notes")
    if not isinstance(record["acquisition"], dict) or not isinstance(record["calibration"], dict) or not isinstance(record["analysis"], dict):
        raise ValueError("Invalid notebook sections")
    try:
        datetime.fromisoformat(record["recorded_at_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid record timestamp") from exc
    return record

def notebook_json(record):
    validate_experiment_record(record)
    return json.dumps(record, indent=2, allow_nan=False)

def load_notebook_json(content):
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    record = json.loads(content)
    return validate_experiment_record(record)
