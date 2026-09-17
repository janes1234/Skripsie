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

    # Also crop out every relabelled region (captioned "OLD -> NEW") into
    # export-dir/<facility>/<component>/, plus an index.csv there, so the
    # actual before/after images can be sent off for review:
    python compare_relabels.py --export-dir ../relabel_review

Run this from the same directory as prep.py (it imports prep for the
facility/unit/class config, same as view_labels.ipynb does).
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

import prep

CAPTION_BAR_HEIGHT = 22
CAPTION_BG = (0, 0, 0)
CAPTION_FG = (255, 255, 0)


def region_label_map(entry: dict, attr_key: str) -> dict:
    """Returns {region_index: label_or_None} for one image entry."""
    return {
        i: r.get("region_attributes", {}).get(attr_key)
        for i, r in enumerate(entry.get("regions", []))
    }


def compare_json_pair(original_path: Path, working_path: Path, attr_key: str) -> list[dict]:
    """Returns a list of dicts describing every changed region between the
    original json and its relabel/ working copy. Assumes both files exist.

    Each dict also carries "shape_attributes" (the VIA2 shape dict for that
    region, taken from whichever copy still has that region, or None if
    neither does) so a changed region can be cropped back out of the source
    image later -- see export_relabel_crop() below.
    """
    original_meta = prep.load_via2_json(original_path)
    working_meta = prep.load_via2_json(working_path)

    diffs = []
    for img_key, working_entry in working_meta.items():
        filename = working_entry.get("filename", img_key)
        original_entry = original_meta.get(img_key)
        working_regions = working_entry.get("regions", [])

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
                        "shape_attributes": working_regions[i].get("shape_attributes"),
                    })
            continue

        original_regions = original_entry.get("regions", [])
        old_labels = region_label_map(original_entry, attr_key)
        new_labels = region_label_map(working_entry, attr_key)

        for i in sorted(set(old_labels) | set(new_labels)):
            old = old_labels.get(i, "(region did not exist)")
            new = new_labels.get(i, "(region removed)")
            if old != new:
                # Prefer the working copy's shape (the region as it exists
                # now) and fall back to the original's, so a region that was
                # deleted in relabel/ can still be cropped as it used to be.
                if i < len(working_regions):
                    shape_attributes = working_regions[i].get("shape_attributes")
                elif i < len(original_regions):
                    shape_attributes = original_regions[i].get("shape_attributes")
                else:
                    shape_attributes = None
                diffs.append({
                    "filename": filename, "region": i,
                    "old_label": old, "new_label": new,
                    "shape_attributes": shape_attributes,
                })

    # Entries present in Labels/ but missing entirely from relabel/ -- shouldn't
    # normally happen since a working copy starts as a full copy of the
    # original, but worth surfacing if it ever does.
    for img_key in set(original_meta) - set(working_meta):
        filename = original_meta[img_key].get("filename", img_key)
        diffs.append({
            "filename": filename, "region": "-",
            "old_label": "(entry existed)", "new_label": "(entry missing from relabel/ copy)",
            "shape_attributes": None,
        })

    return diffs


def export_relabel_crop(image_path: Path, shape_attributes: dict, old_label, new_label, out_path: Path):
    """Crops `shape_attributes`'s region out of the image at image_path (the
    same masked crop prep.py feeds to the CNN), stamps a caption bar reading
    "OLD -> NEW" underneath it, and saves it to out_path. Raises ValueError
    via prep.region_bbox() if the shape type is unsupported."""
    img = Image.open(image_path).convert("RGB")
    box = prep.clamp_box(prep.region_bbox(shape_attributes), img.width, img.height)
    masked = prep.apply_shape_mask(img, shape_attributes)
    crop = masked.crop(box)

    captioned = Image.new("RGB", (crop.width, crop.height + CAPTION_BAR_HEIGHT), CAPTION_BG)
    captioned.paste(crop, (0, 0))
    draw = ImageDraw.Draw(captioned)
    draw.text((4, crop.height + 4), f"{old_label} -> {new_label}", fill=CAPTION_FG)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    captioned.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facility", default=None,
                         help=f"Restrict to one facility. Choices: {sorted(prep.FACILITIES)}")
    parser.add_argument("--component", default=None, choices=["clarifier", "aerobic_zone"],
                         help="Restrict to one component (default: both)")
    parser.add_argument("--csv", default=None, help="Also write results to this CSV path")
    parser.add_argument("--export-dir", default=None,
                         help="Also crop every relabelled region (captioned OLD -> NEW) into "
                              "this directory, organised as <export-dir>/<facility>/<component>/, "
                              "plus an index.csv there mapping each crop back to its source image.")
    args = parser.parse_args()

    if args.facility and args.facility not in prep.FACILITIES:
        parser.error(f"Unknown facility {args.facility!r}. Choices: {sorted(prep.FACILITIES)}")

    facilities = [args.facility] if args.facility else list(prep.FACILITIES)
    components = [args.component] if args.component else ["clarifier", "aerobic_zone"]
    export_root = Path(args.export_dir) if args.export_dir else None

    all_rows = []
    exported_rows = []
    export_failures = []

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

                unit_img_dir = unit.get("images_dir")
                unit_label = unit_img_dir or facility
                images_dir = facility_root / unit_img_dir if unit_img_dir else facility_root
                safe_unit_label = unit_label.replace("/", "-").replace(" ", "_")

                for d in compare_json_pair(original_path, working_path, attr_key):
                    row = {
                        "facility": facility, "unit": unit_label, "component": component,
                        "json_file": json_name, **d,
                    }
                    all_rows.append(row)

                    if export_root is None or d["region"] == "-":
                        continue  # nothing to crop for a whole-entry diff

                    image_path = prep.find_image_case_insensitive(images_dir, row["filename"])
                    if image_path is None:
                        export_failures.append(f"{facility}/{unit_label}/{component}: "
                                                f"source image not found for {row['filename']}")
                        continue
                    if d["shape_attributes"] is None:
                        export_failures.append(f"{facility}/{unit_label}/{component}: "
                                                f"no shape data for {row['filename']} region {row['region']}")
                        continue

                    safe_old = str(d["old_label"]).replace("/", "-")
                    safe_new = str(d["new_label"]).replace("/", "-")
                    out_name = (f"{facility}_{safe_unit_label}_{image_path.stem}_r{d['region']}_"
                                f"{safe_old}_to_{safe_new}.jpg")
                    out_path = export_root / facility / component / out_name

                    try:
                        export_relabel_crop(image_path, d["shape_attributes"],
                                             d["old_label"], d["new_label"], out_path)
                    except Exception as e:
                        export_failures.append(f"{facility}/{unit_label}/{component}: "
                                                f"could not crop {row['filename']} region {row['region']}: {e}")
                        continue

                    exported_rows.append({**row, "exported_file": str(out_path.relative_to(export_root))})

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
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")

    if export_root is not None:
        print(f"\nExported {len(exported_rows)} relabelled crop(s) to {export_root.resolve()}")
        if export_failures:
            print(f"[!] {len(export_failures)} region(s) could not be exported:")
            for msg in export_failures:
                print(f"  - {msg}")

        index_path = export_root / "index.csv"
        fieldnames = ["facility", "unit", "component", "json_file", "filename", "region",
                      "old_label", "new_label", "exported_file"]
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(exported_rows)
        print(f"Wrote export index to {index_path}")


if __name__ == "__main__":
    main()
