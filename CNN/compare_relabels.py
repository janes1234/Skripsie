"""
compare_relabels.py
====================
Compares each facility's Labels/ (original VIA2 annotations) against its
relabel/ (working copy edited via relabel.ipynb / view_labels.ipynb) and
reports every relabel that's actually been made: which image, which region,
what it used to be, and what it is now.

Only looks at relabel/ files that exist -- if a unit/component hasn't been
touched yet (no relabel/ copy for it), it's silently skipped, not reported
as "no changes".

Usage:
    python compare_relabels.py                        # everything
    python compare_relabels.py --facility CapeFlats
    python compare_relabels.py --component clarifier
    python compare_relabels.py --csv relabels.csv      # also write a CSV

Run this from the same directory as prep.py (it imports prep for the
facility/unit/class config, same as view_labels.ipynb does).
"""

import argparse
import csv
from pathlib import Path

import prep


def region_label_map(entry: dict, attr_key: str) -> dict:
    """Returns {region_index: label_or_None} for one image entry."""
    return {
        i: r.get("region_attributes", {}).get(attr_key)
        for i, r in enumerate(entry.get("regions", []))
    }


def compare_json_pair(original_path: Path, working_path: Path, attr_key: str) -> list[dict]:
    """Returns a list of dicts describing every changed region between the
    original json and its relabel/ working copy. Assumes both files exist."""
    original_meta = prep.load_via2_json(original_path)
    working_meta = prep.load_via2_json(working_path)

    diffs = []
    for img_key, working_entry in working_meta.items():
        filename = working_entry.get("filename", img_key)
        original_entry = original_meta.get(img_key)

        if original_entry is None:
            # This exact key (filename+size) isn't in the original at all --
            # either a brand new entry, or the file was resaved with a
            # different size, so the key itself changed. Report any labelled
            # regions so it doesn't get missed.
            for i, label in region_label_map(working_entry, attr_key).items():
                if label:
                    diffs.append({
                        "filename": filename, "region": i,
                        "old_label": "(no matching entry in Labels/)", "new_label": label,
                    })
            continue

        old_labels = region_label_map(original_entry, attr_key)
        new_labels = region_label_map(working_entry, attr_key)

        for i in sorted(set(old_labels) | set(new_labels)):
            old = old_labels.get(i, "(region did not exist)")
            new = new_labels.get(i, "(region removed)")
            if old != new:
                diffs.append({"filename": filename, "region": i, "old_label": old, "new_label": new})

    # Entries present in Labels/ but missing entirely from relabel/ -- shouldn't
    # normally happen since a working copy starts as a full copy of the
    # original, but worth surfacing if it ever does.
    for img_key in set(original_meta) - set(working_meta):
        filename = original_meta[img_key].get("filename", img_key)
        diffs.append({
            "filename": filename, "region": "-",
            "old_label": "(entry existed)", "new_label": "(entry missing from relabel/ copy)",
        })

    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facility", default=None,
                         help=f"Restrict to one facility. Choices: {sorted(prep.FACILITIES)}")
    parser.add_argument("--component", default=None, choices=["clarifier", "aerobic_zone"],
                         help="Restrict to one component (default: both)")
    parser.add_argument("--csv", default=None, help="Also write results to this CSV path")
    args = parser.parse_args()

    if args.facility and args.facility not in prep.FACILITIES:
        parser.error(f"Unknown facility {args.facility!r}. Choices: {sorted(prep.FACILITIES)}")

    facilities = [args.facility] if args.facility else list(prep.FACILITIES)
    components = [args.component] if args.component else ["clarifier", "aerobic_zone"]

    all_rows = []
    for facility in facilities:
        facility_root = prep.RAW_DATA_ROOT / facility
        facility_cfg = prep.FACILITIES[facility]
        relabel_dir = facility_root / prep.RELABEL_SUBDIR

        for component in components:
            attr_key = prep.CLARIFIER_ATTR_KEY if component == "clarifier" else prep.AEROBIC_ATTR_KEY
            json_key = "clarifier_json" if component == "clarifier" else "aerobic_json"

            for unit in facility_cfg["units"]:
                json_name = unit.get(json_key)
                if not json_name:
                    continue  # this unit has no json for this component

                original_path = facility_root / prep.LABELS_SUBDIR / json_name
                working_path = relabel_dir / json_name

                if not working_path.exists():
                    continue  # nothing relabelled yet for this unit/component
                if not original_path.exists():
                    print(f"[warn] {original_path} is missing, can't diff against it -- skipping")
                    continue

                unit_label = unit.get("images_dir") or facility

                for d in compare_json_pair(original_path, working_path, attr_key):
                    all_rows.append({
                        "facility": facility, "unit": unit_label, "component": component,
                        "json_file": json_name, **d,
                    })

    if not all_rows:
        print("No relabels found -- every relabel/ working copy (if any exist) still matches its Labels/ original.")
        return

    print(f"Found {len(all_rows)} relabelled region(s):\n")
    for row in all_rows:
        print(f"  [{row['facility']}/{row['unit']}/{row['component']}] {row['filename']}  "
              f"region {row['region']}:  {row['old_label']}  ->  {row['new_label']}")

    if args.csv:
        fieldnames = ["facility", "unit", "component", "json_file", "filename", "region", "old_label", "new_label"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
