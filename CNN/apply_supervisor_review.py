"""
apply_supervisor_review.py
==========================
Applies the supervisor's verdicts from "wastewater/Clarifier changes made.xlsx"
to the relabel/ working copies of the clarifier label jsons. The original
Labels/ jsons are never touched.

The spreadsheet has one row per region that was relabelled (old_label is what
Labels/ says, new_label is what relabel/ currently says), and the supervisor's
verdict in the last column ("change label?"):

    1                 -> keep the proposed change        (new_label)
    0                 -> revert to the original label    (old_label)
    <label>           -> use that label instead          (e.g. "scum")
    ..., bl           -> borderline; her call is still used as-is
                         ("1, bl" -> new_label, "0, bl" -> old_label,
                          "scum, bl" -> Scum)

Before changing anything, every row is checked against the current relabel/
json (the region must exist and currently hold new_label), so a spreadsheet
that's out of sync with the jsons aborts instead of writing wrong labels. The
relabel/ jsons are backed up to wastewater/_relabel_backup_<timestamp>/ first,
and every decision is written to relabel_review/supervisor_decisions_applied.csv.

Usage (from CNN/):
    python apply_supervisor_review.py --dry-run   # check + print, write nothing
    python apply_supervisor_review.py
"""

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import openpyxl

RAW_DATA_ROOT = Path("../wastewater")
XLSX_PATH = RAW_DATA_ROOT / "Clarifier changes made.xlsx"
AUDIT_CSV = Path("../relabel_review/supervisor_decisions_applied.csv")
ATTR_KEY = "clarifier"
VALID_LABELS = {"Functional", "Dysfunctional", "Scum", "Empty"}


def resolve_verdict(verdict, old_label, new_label):
    """Returns (final_label, rule) for one spreadsheet verdict cell."""
    tokens = [t.strip().lower() for t in str(verdict).split(",")]
    borderline = "bl" in tokens
    tokens = [t for t in tokens if t and t != "bl"]
    if len(tokens) != 1:
        raise ValueError(f"can't parse verdict {verdict!r}")
    tok = tokens[0]
    if tok in ("1", "1.0"):
        final, rule = new_label, "1: change to proposed"
    elif tok in ("0", "0.0"):
        final, rule = old_label, "0: keep original"
    elif tok.capitalize() in VALID_LABELS:
        final, rule = tok.capitalize(), "supervisor's own label"
    else:
        raise ValueError(f"can't parse verdict {verdict!r}")
    if borderline:
        rule += " (borderline)"
    return final, rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = list(openpyxl.load_workbook(XLSX_PATH).worksheets[0].iter_rows(values_only=True))
    header, rows = rows[0], rows[1:]
    assert header[:8] == ("facility", "unit", "component", "json_file", "filename",
                          "region", "old_label", "new_label"), header

    # Load every relabel json referenced by the spreadsheet once.
    jsons = {}
    for r in rows:
        key = (r[0], r[3])
        if key not in jsons:
            path = RAW_DATA_ROOT / r[0] / "relabel" / r[3]
            jsons[key] = (path, json.loads(path.read_text(encoding="utf-8")))

    decisions = []
    for r in rows:
        facility, unit, component, json_file, filename, region, old_label, new_label = r[:8]
        verdict = r[9]
        assert component == ATTR_KEY, r
        final, rule = resolve_verdict(verdict, old_label, new_label)
        assert final in VALID_LABELS, (r, final)

        path, data = jsons[(facility, json_file)]
        entries = [e for e in data["_via_img_metadata"].values() if e["filename"] == filename]
        assert len(entries) == 1, f"{path}: {len(entries)} entries for {filename}"
        regions = entries[0]["regions"]
        assert region < len(regions), f"{path}: {filename} has no region {region}"
        current = regions[region]["region_attributes"].get(ATTR_KEY)
        assert current == new_label, (
            f"{path}: {filename} r{region} is {current!r}, spreadsheet expects {new_label!r}")

        regions[region]["region_attributes"][ATTR_KEY] = final
        decisions.append({
            "facility": facility, "unit": unit, "json_file": json_file, "filename": filename,
            "region": region, "original_label": old_label, "proposed_label": new_label,
            "supervisor_verdict": verdict, "rule": rule, "final_label": final,
        })

    print(f"{len(decisions)} regions resolved.")
    print("By rule:", dict(Counter(d["rule"] for d in decisions)))
    print("Final == original (reverted):", sum(d["final_label"] == d["original_label"] for d in decisions))
    print("Final == proposed (kept):     ", sum(d["final_label"] == d["proposed_label"] for d in decisions))
    print("Final is a third label:       ", sum(d["final_label"] not in (d["original_label"], d["proposed_label"]) for d in decisions))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup_root = RAW_DATA_ROOT / f"_relabel_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    for (facility, json_file), (path, data) in jsons.items():
        dest = backup_root / facility / "relabel" / json_file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f"Backed up {len(jsons)} relabel json(s) to {backup_root}")

    for path, data in jsons.values():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(jsons)} relabel json(s).")

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0]))
        w.writeheader()
        w.writerows(decisions)
    print(f"Audit trail written to {AUDIT_CSV}")


if __name__ == "__main__":
    main()
