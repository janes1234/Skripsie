"""
date_utils.py
=============
Parses the capture date (year, month) out of a WWTW crop filename.

prep.py names every crop `{facility}_{safe_unit_label}_{original_stem}_r{i}.jpg`
(see prep.py's `process_json_file`), so the facility is always the first
underscore-separated token, same convention `split_dataset_facility.py`'s
`get_facility()` already relies on. The *original_stem* part -- and with it,
where the date sits and how it's formatted -- differs per facility, because
each facility's source images were named differently to begin with:

    Atlantis         "Atlantis 02_2022"        MM_YYYY
    CapeFlats        "capeflats3_202101"       YYYYMM (contiguous)
    Waterval         "Waterval 06_2014"        MM_YYYY
    Fisantekraal     "Fisantekraal 2015_06"    YYYY_MM
    NoordelikeWerke  "Northern Works 2012 06"  YYYY MM (space-separated)

Used by split_dataset_chronological.py (to build the split) and by its
own --list date-distribution report (to pick a sensible cutoff).
"""

import re

# Each pattern is matched against the *whole* crop filename (not just the
# original stem) -- the facility/unit-label prefix prep.py adds never
# contains a run of digits that could be confused for one of these, since
# unit labels are things like "Images-CF3" or "Images-Waterval_(Top)".
# (?<!\d)/(?!\d) guards stop a pattern from matching a *sub*-string of a
# longer digit run that isn't actually a date.
# Some source images have a "(2)"-style disambiguator (a duplicate upload)
# between the date and the region suffix, e.g. "capeflats2_201408(2)_r3.jpg"
# or "Fisantekraal 2020_10 (1)_r0.jpg" -- optional, so it's allowed but not
# required right before "_r\d+".
_DUP_SUFFIX = r"(?:\s?\(\d+\))?_r\d+"

_PATTERNS = {
    # MM_YYYY, e.g. "Atlantis 02_2022_r0.jpg" / "Waterval 06_2014_r1.jpg"
    "Atlantis": (re.compile(r"(?<!\d)(\d{2})_(\d{4})(?!\d)"), "my"),
    "Waterval": (re.compile(r"(?<!\d)(\d{2})_(\d{4})(?!\d)"), "my"),
    # YYYYMM immediately before the region suffix, e.g. "capeflats3_202101_r0.jpg"
    "CapeFlats": (re.compile(r"(?<!\d)(\d{4})(\d{2})" + _DUP_SUFFIX), "ym"),
    # YYYY_MM, e.g. "Fisantekraal 2015_06_r0.jpg"
    "Fisantekraal": (re.compile(r"(?<!\d)(\d{4})_(\d{2})" + _DUP_SUFFIX), "ym"),
    # YYYY MM (space), e.g. "Northern Works 2012 06_r3.jpg"
    "NoordelikeWerke": (re.compile(r"(?<!\d)(\d{4}) (\d{2})" + _DUP_SUFFIX), "ym"),
}


def get_facility(filename: str) -> str:
    """Same convention as split_dataset_facility.py's get_facility()."""
    return filename.split("_")[0]


def parse_crop_date(filename: str) -> tuple[int, int]:
    """Returns (year, month) for a crop filename. Raises ValueError if the
    facility is unknown or the filename doesn't match that facility's date
    pattern -- silently skipping an unparseable file would mean it quietly
    vanishes from one side of the chronological split, which is worse than
    crashing loudly here."""
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
