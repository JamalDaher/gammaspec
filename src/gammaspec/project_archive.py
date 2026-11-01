"""Self-contained GammaSpec experiment archives with SHA-256 integrity checks.

Archives are untrusted input: no extraction to disk, fixed member names and size caps.
"""
from io import BytesIO
from hashlib import sha256
import json
import zipfile
import pandas as pd
from .core import _validate
from .notebook import notebook_json, load_notebook_json

ARCHIVE_SCHEMA = 1
MAX_ARCHIVE_BYTES = 15_000_000
MAX_MEMBER_BYTES = 10_000_000
FILES = ("notebook.json", "spectrum.csv")

def export_experiment_archive(record, spectrum, software_version="1.0.0"):
    _validate(spectrum)
    if not isinstance(software_version, str) or not software_version:
        raise ValueError("Invalid software version")
    payloads = {
        "notebook.json": notebook_json(record).encode("utf-8"),
        "spectrum.csv": spectrum.to_csv(index=False, float_format="%.17g").encode("utf-8"),
    }
    manifest = {"schema_version": ARCHIVE_SCHEMA, "software": "GammaSpec",
                "software_version": software_version,
                "files": {name: {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in payloads.items()}}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    return stream.getvalue()

def import_experiment_archive(content):
    if not isinstance(content, bytes) or len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid or oversized archive")
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            names = archive.namelist()
            if len(names) != 3 or set(names) != {*FILES, "manifest.json"}:
                raise ValueError("Archive must contain only the three expected files")
            for info in archive.infolist():
                if info.file_size > MAX_MEMBER_BYTES or info.is_dir():
                    raise ValueError("Oversized or invalid archive member")
            manifest = json.loads(archive.read("manifest.json"))
            if (manifest.get("schema_version") != ARCHIVE_SCHEMA
                    or manifest.get("software") != "GammaSpec"
                    or set(manifest.get("files", {})) != set(FILES)):
                raise ValueError("Unsupported archive manifest")
            data = {}
            for name in FILES:
                payload = archive.read(name)
                expected = manifest["files"][name]
                if len(payload) != expected["bytes"] or sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Integrity check failed for {name}")
                data[name] = payload
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid experiment archive") from exc
    record = load_notebook_json(data["notebook.json"])
    try:
        spectrum = pd.read_csv(BytesIO(data["spectrum.csv"]))
        _validate(spectrum)
    except (pd.errors.ParserError, ValueError, KeyError) as exc:
        raise ValueError("Invalid archived spectrum") from exc
    return record, spectrum, manifest
