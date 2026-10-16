"""
crop_names.py
=============
Parses the facility, unit and capture date out of a crop filename.

prep.py names every crop `{facility}_{unit_label}_{original_stem}_r{i}.jpg`.
The date format inside original_stem differs per facility:

    Atlantis         "Atlantis 02_2022"        MM_YYYY
    CapeFlats        "capeflats3_202101"       YYYYMM
    Waterval         "Waterval 06_2014"        MM_YYYY
    Fisantekraal     "Fisantekraal 2015_06"    YYYY_MM
    NoordelikeWerke  "Northern Works 2012 06"  YYYY MM
"""

import re

# Some source images have a "(2)"-style duplicate marker before the region
# suffix, e.g. "capeflats2_201408(2)_r3.jpg".
_DUP_SUFFIX = r"(?:\s?\(\d+\))?_r\d+"

# (pattern, order); the lookarounds stop a match inside a longer run of digits.
_PATTERNS = {
    "Atlantis": (re.compile(r"(?<!\d)(\d{2})_(\d{4})(?!\d)"), "my"),
    "Waterval": (re.compile(r"(?<!\d)(\d{2})_(\d{4})(?!\d)"), "my"),
    "CapeFlats": (re.compile(r"(?<!\d)(\d{4})(\d{2})" + _DUP_SUFFIX), "ym"),
    "Fisantekraal": (re.compile(r"(?<!\d)(\d{4})_(\d{2})" + _DUP_SUFFIX), "ym"),
    "NoordelikeWerke": (re.compile(r"(?<!\d)(\d{4}) (\d{2})" + _DUP_SUFFIX), "ym"),
}


def get_facility(filename: str) -> str:
    return filename.split("_")[0]


def get_facility_and_unit(filename: str) -> str:
    """Grouping key for the stratified splits, e.g. "CapeFlats_Images-CF1"."""
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "Unknown_Unit"


def parse_crop_date(filename: str) -> tuple[int, int]:
    """Returns (year, month). Raises ValueError rather than skipping a file,
    so no image silently drops out of the chronological split."""
    facility = get_facility(filename)
    if facility not in _PATTERNS:
        raise ValueError(f"Unknown facility '{facility}' in filename {filename!r}")

    pattern, order = _PATTERNS[facility]
    m = pattern.search(filename)
    if not m:
        raise ValueError(f"Could not parse a date out of {filename!r} for facility '{facility}'")

    a, b = m.group(1), m.group(2)
    year, month = (a, b) if order == "ym" else (b, a)
    year, month = int(year), int(month)
    if not (1 <= month <= 12):
        raise ValueError(f"Parsed an invalid month {month} out of {filename!r}")
    return year, month
