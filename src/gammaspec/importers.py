"""Import common text-based multichannel analyzer exports.

Supported: CSV with channel/counts (optionally energy_kev), or a single
counts-per-line text file. Binary/vendor-specific formats are not supported.
"""
from io import StringIO
import numpy as np
import pandas as pd
from .core import _validate

def import_spectrum_text(content, format="auto", energy_per_channel_kev=None,
                         energy_offset_kev=0.0):
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Binary or non-UTF-8 data is not supported") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty spectrum file")
    if format not in ("auto", "csv", "counts"):
        raise ValueError("Unknown text format")
    if format == "auto":
        first = next((line.strip() for line in content.splitlines() if line.strip()), "")
        format = "csv" if "," in first or first.lower().startswith("channel") else "counts"
    if format == "csv":
        frame = pd.read_csv(StringIO(content))
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if not {"channel", "counts"}.issubset(frame.columns):
            raise ValueError("CSV requires channel and counts columns")
        frame = frame[[c for c in ("channel", "energy_kev", "counts") if c in frame]]
    else:
        try:
            values = [float(line.strip()) for line in content.splitlines() if line.strip()]
        except ValueError as exc:
            raise ValueError("Counts text requires one numeric value per line") from exc
        frame = pd.DataFrame({"channel": np.arange(len(values)), "counts": values})
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Spectrum contains nonnumeric values") from exc
    if len(frame) < 2:
        raise ValueError("At least two channels are required")
    ch = frame.channel.to_numpy(dtype=float)
    counts = frame.counts.to_numpy(dtype=float)
    if (not np.isfinite(ch).all() or not np.isfinite(counts).all()
            or (np.diff(ch) <= 0).any() or (counts < 0).any()
            or not np.allclose(ch, np.round(ch)) or not np.allclose(counts, np.round(counts))):
        raise ValueError("Channels must increase and counts must be nonnegative integers")
    if "energy_kev" not in frame:
        if energy_per_channel_kev is None:
            raise ValueError("No energy axis; supply a positive keV/channel value or calibrate separately")
        if not np.isfinite(energy_per_channel_kev) or energy_per_channel_kev <= 0:
            raise ValueError("keV/channel must be positive and finite")
        if not np.isfinite(energy_offset_kev):
            raise ValueError("Energy offset must be finite")
        frame["energy_kev"] = energy_offset_kev + ch * energy_per_channel_kev
    frame = frame[["channel", "energy_kev", "counts"]].reset_index(drop=True)
    _validate(frame)
    return frame
