"""
Parses VIA2 (VGG Image Annotator) JSON label files for the Atlantis, Cape Flats,
and Waterval WWTW image sets, crops each labelled region (aerobic zone or
clarifier) out of the source image, and saves the crops into a class-folder
structure suitable for CNN training:

    output_root/
        aerobic_zone/
            Functional/
            Suboptimal/
            Dysfunctional/
        clarifier/
            Functional/
            Dysfunctional/
            Scum/
            Empty/

"""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

# ------------------------- CONFIG ------------------------

RAW_DATA_ROOT = Path("../wastewater")
OUTPUT_ROOT = Path("../cnn_dataset")

AEROBIC_ATTR_KEY = "aerobic zone"
CLARIFIER_ATTR_KEY = "clarifier"

FACILITIES = {
    "Atlantis": {
        "units": [
            {
                "images_dir": "Images",
                "aerobic_json": "Atlantis Aerobic Zone.json",
                "clarifier_json": "Atlantis Clarifiers.json",
            }
        ]
    },
    "CapeFlats": {
        "units": [
            {
                "images_dir": "Images/CF1",
                "aerobic_json": "Cape Flats 1 Aerobic Zones.json",
                "clarifier_json": "Cape Flats 1 Clarifiers.json",
            },
            {
                "images_dir": "Images/CF2",
                "aerobic_json": "Cape Flats 2 Aerobic Zones.json",
                "clarifier_json": "Cape Flats 2 Clarifiers.json",
            },
            {
                "images_dir": "Images/CF3",
                "aerobic_json": "Cape Flats 3 Aerobic Zones.json",
                "clarifier_json": "Cape Flats 3 Clarifiers.json",
            },
            {
                "images_dir": "Images/CF4",
                "aerobic_json": "Cape Flats 4 Aerobic Zones.json",
                "clarifier_json": "Cape Flats 4 Clarifiers.json",
            },
        ]
    },
    "Waterval": {
        "units": [
            {
                "images_dir": "Images/Waterval (Top)",
                "aerobic_json": "Waterval Aerobic Zone Top.json",
                "clarifier_json": "Waterval Clarifiers Top.json",
            },
            {
                "images_dir": "Images/Waterval (Bottom)",
                "aerobic_json": "Waterval Aerobic Zone Bottom.json",
                "clarifier_json": "Waterval Clarifiers Bottom.json",
            },
        ]
    },
    "Fisantekraal": {
            "units": [
                {
                    "images_dir": "Images",
                    "clarifier_json": "Fisantekraal Clarifiers.json",
                }
            ]
        },
        "NoordelikeWerke": {
                "units": [
                    {
                        "images_dir": "Images",
                        "clarifier_json": "Noordelike Werke Clarifiers.json",
                    }
                ]
            },
}

LABELS_SUBDIR = "Labels"
RELABEL_SUBDIR = "relabel"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ------------------------------------------------------------------------------

AEROBIC_CLASSES = {"Functional", "Suboptimal", "Dysfunctional", "Empty"}
CLARIFIER_CLASSES = {"Functional", "Dysfunctional", "Scum", "Empty", "Stagnant"}

EXCLUDED_AEROBIC_CLASSES = {"Empty"}
EXCLUDED_CLARIFIER_CLASSES = {"Stagnant"}


def normalize_label(raw_label: str) -> str:
    return raw_label.strip().capitalize()


def load_via2_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("_via_img_metadata", data)


def get_images_on_disk(images_dir: Path) -> set:
    """Return a set of all valid image paths in the directory."""
    if not images_dir.exists():
        return set()
    return {p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


def resolve_label_path(facility_root: Path, filename: str) -> Path:
    """The single place that decides which copy of a label json to read.

    Prefers facility_root/relabel/{filename} (the working copy the relabel
    notebook edits) if it exists, otherwise falls back to
    facility_root/Labels/{filename} (the pristine original). Every caller
    should go through this instead of hardcoding either subfolder, so a
    relabel edit is picked up everywhere the json is read from without
    needing every facility/unit to already have a relabel/ folder."""
    relabel_path = facility_root / RELABEL_SUBDIR / filename
    if relabel_path.exists():
        return relabel_path
    return facility_root / LABELS_SUBDIR / filename


def find_image_case_insensitive(images_dir: Path, filename: str):
    """Attempt to find a file exactly, then fall back to case-insensitive match."""
    exact_path = images_dir / filename
    if exact_path.exists():
        return exact_path
    
    # Fallback for case sensitivity issues (e.g. .JPG vs .jpg)
    if images_dir.exists():
        target_lower = filename.lower()
        for p in images_dir.iterdir():
            if p.name.lower() == target_lower:
                return p
    return None


def region_bbox(shape_attributes: dict) -> tuple[int, int, int, int]:
    shape = shape_attributes.get("name")

    if shape == "rect":
        x = shape_attributes["x"]
        y = shape_attributes["y"]
        w = shape_attributes["width"]
        h = shape_attributes["height"]
        return x, y, x + w, y + h

    if shape == "circle":
        cx = shape_attributes["cx"]
        cy = shape_attributes["cy"]
        r = shape_attributes["r"]
        return cx - r, cy - r, cx + r, cy + r

    if shape == "ellipse":
        cx = shape_attributes["cx"]
        cy = shape_attributes["cy"]
        rx = shape_attributes["rx"]
        ry = shape_attributes["ry"]
        return cx - rx, cy - ry, cx + rx, cy + ry

    raise ValueError(f"Unsupported VIA2 shape type: {shape}")


def clamp_box(box, img_width, img_height):
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)
    return left, top, right, bottom


def apply_shape_mask(img, shape_attrs):
    """
    Draws a black mask outside the annotated VIA2 region to remove 
    corners, background concrete, and facility-specific shortcut features.
    """
    # Create a blank mask (0 = black)
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    shape = shape_attrs.get("name")
    
    # Draw the annotated shape in white (255) onto the mask
    if shape == "circle":
        cx, cy, r = shape_attrs["cx"], shape_attrs["cy"], shape_attrs["r"]
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    elif shape == "ellipse":
        cx, cy, rx, ry = shape_attrs["cx"], shape_attrs["cy"], shape_attrs["rx"], shape_attrs["ry"]
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    elif shape == "rect":
        x, y, w, h = shape_attrs["x"], shape_attrs["y"], shape_attrs["width"], shape_attrs["height"]
        draw.rectangle((x, y, x + w, y + h), fill=255)
    else:
        return img  # Fallback if shape is unknown
        
    # Create a solid black background image
    black_bg = Image.new("RGB", img.size, (0, 0, 0))
    
    # Composite the original image over the black background using the white mask
    return Image.composite(img, black_bg, mask)


def process_json_file(
    json_path: Path,
    images_dir: Path,
    attr_key: str,
    output_component_root: Path,
    facility_name: str,
    unit_label: str,
    counters: dict,
    excluded_counters: dict,
    facility_class_counts: dict,
    valid_classes: set,
    excluded_classes: set,
    manifest: dict,
):
    images_processed = 0
    missing_filenames = []
    excluded_filenames = []
    used_paths = set()

    if not json_path.exists():
        print(f"  [skip] label file not found: {json_path}")
        return images_processed, missing_filenames, used_paths, excluded_filenames

    img_metadata = load_via2_json(json_path)

    for img_key, entry in img_metadata.items():
        filename = entry.get("filename")
        if not filename:
            continue

        image_path = find_image_case_insensitive(images_dir, filename)
        if not image_path:
            missing_filenames.append(filename)
            continue

        used_paths.add(image_path)

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not open {image_path}: {e}")
            continue

        images_processed += 1

        for i, region in enumerate(entry.get("regions", [])):
            shape_attrs = region.get("shape_attributes", {})
            region_attrs = region.get("region_attributes", {})
            label = region_attrs.get(attr_key)

            if not label:
                continue

            label = normalize_label(label)
            if label not in valid_classes:
                print(
                    f"  [ALERT] unexpected label '{label}' on region {i} in {filename} "
                    f"(not in {sorted(valid_classes)}) — check this annotation manually"
                )

            if label in excluded_classes:
                excluded_counters[label] = excluded_counters.get(label, 0) + 1
                excluded_filenames.append(f"{filename} (Label: {label})")
                continue

            try:
                box = region_bbox(shape_attrs)
            except ValueError as e:
                print(f"  [warn] {e} in {filename}, skipping region")
                continue

            box = clamp_box(box, img.width, img.height)
            if box[2] <= box[0] or box[3] <= box[1]:
                print(f"  [warn] degenerate crop box in {filename}, skipping region")
                continue

            masked_img = apply_shape_mask(img, shape_attrs)
            crop = masked_img.crop(box)

            class_dir = output_component_root / label
            class_dir.mkdir(parents=True, exist_ok=True)

            safe_unit_label = unit_label.replace("/", "-").replace(" ", "_")
            
            # Always save using the actual file name on disk, not the JSON name, 
            # to avoid extension mismatches downstream
            out_name = f"{facility_name}_{safe_unit_label}_{image_path.stem}_r{i}.jpg"
            crop.save(class_dir / out_name, quality=95)

            counters[label] = counters.get(label, 0) + 1
            facility_class_counts[facility_name][label] += 1

            # Record everything needed to find this exact region again later
            # (e.g. from the relabel notebook) without having to reverse-parse
            # the crop filename, which is unreliable since facility/unit
            # names can themselves contain underscores or spaces.
            manifest[out_name] = {
                "facility": facility_name,
                "unit_label": unit_label,
                "attr_key": attr_key,
                "json_path": str(json_path.resolve()),
                "img_key": img_key,
                "region_index": i,
                "label_at_export": label,
            }

    return images_processed, missing_filenames, used_paths, excluded_filenames


def print_facility_breakdown(title: str, facility_class_counts: dict, classes: set):
    print(f"\n{title} — per facility, per class:")
    class_list = sorted(classes)
    header = f"  {'Facility':<12}" + "".join(f"{c:>15}" for c in class_list)
    print(header)
    for facility_name in FACILITIES:
        counts = facility_class_counts.get(facility_name, {})
        row = f"  {facility_name:<12}" + "".join(f"{counts.get(c, 0):>15}" for c in class_list)
        print(row)


def main():
    # Always rebuild OUTPUT_ROOT from scratch. Without this, a relabel that
    # moves an image from one class to another (e.g. Functional -> Empty)
    # would save a fresh copy under the new class folder but leave the
    # stale old copy sitting under the old class folder forever, since the
    # per-class crop.save() calls below only ever create files, never
    # remove them. That stale duplicate then gets picked up by
    # split_dataset_facility.py too, so the same image ends up in the
    # training/test data under both its old AND new label at once -- which
    # is exactly the kind of thing that makes a relabel "not stick" even
    # after retraining. A clean wipe-and-rebuild guarantees cnn_dataset
    # only ever reflects whatever the (relabel-aware) json files currently
    # say, with nothing left over from a previous run.
    if OUTPUT_ROOT.exists():
        print(f"Cleaning up existing output directory: {OUTPUT_ROOT.resolve()}")
        shutil.rmtree(OUTPUT_ROOT)

    aerobic_counts = {}
    clarifier_counts = {}
    aerobic_excluded = {}
    clarifier_excluded = {}
    aerobic_by_facility = defaultdict(lambda: defaultdict(int))
    clarifier_by_facility = defaultdict(lambda: defaultdict(int))

    # Keyed by crop filename (unique within a component, but aerobic_zone
    # and clarifier crops can share a filename since region indices restart
    # at 0 for each json file), so these stay separate.
    aerobic_manifest = {}
    clarifier_manifest = {}

    total_images_on_disk = 0
    total_images_processed = 0
    
    global_missing_json = []
    global_ignored_disk = []
    global_excluded_files = []

    for facility_name, facility_cfg in FACILITIES.items():
        facility_root = RAW_DATA_ROOT / facility_name

        print(f"\n=== {facility_name} ===")

        for unit in facility_cfg["units"]:
            # Safely fetch the images directory 
            unit_img_dir = unit.get("images_dir")
            unit_label = unit_img_dir if unit_img_dir else facility_name
            images_dir = facility_root / unit_img_dir if unit_img_dir else facility_root

            disk_paths = get_images_on_disk(images_dir)
            on_disk = len(disk_paths)
            print(f" -- unit: {unit_label} ({images_dir}) — {on_disk} image(s) on disk")

            # Safely get the filenames (returns None if the key doesn't exist)
            aerobic_json_name = unit.get("aerobic_json")
            clarifier_json_name = unit.get("clarifier_json")

            # Process Aerobic JSON if it exists in the config
            if aerobic_json_name:
                aerobic_json = resolve_label_path(facility_root, aerobic_json_name)
                print(f"    [aerobic]   using: {aerobic_json}")
                aerobic_processed, a_missing, a_used, a_excluded = process_json_file(
                    json_path=aerobic_json,
                    images_dir=images_dir,
                    attr_key=AEROBIC_ATTR_KEY,
                    output_component_root=OUTPUT_ROOT / "aerobic_zone",
                    facility_name=facility_name,
                    unit_label=unit_label,
                    counters=aerobic_counts,
                    excluded_counters=aerobic_excluded,
                    facility_class_counts=aerobic_by_facility,
                    valid_classes=AEROBIC_CLASSES,
                    excluded_classes=EXCLUDED_AEROBIC_CLASSES,
                    manifest=aerobic_manifest,
                )
            else:
                aerobic_processed, a_missing, a_used, a_excluded = 0, [], set(), []

            # Process Clarifier JSON if it exists in the config
            if clarifier_json_name:
                clarifier_json = resolve_label_path(facility_root, clarifier_json_name)
                print(f"    [clarifier] using: {clarifier_json}")
                clarifier_processed, c_missing, c_used, c_excluded = process_json_file(
                    json_path=clarifier_json,
                    images_dir=images_dir,
                    attr_key=CLARIFIER_ATTR_KEY,
                    output_component_root=OUTPUT_ROOT / "clarifier",
                    facility_name=facility_name,
                    unit_label=unit_label,
                    counters=clarifier_counts,
                    excluded_counters=clarifier_excluded,
                    facility_class_counts=clarifier_by_facility,
                    valid_classes=CLARIFIER_CLASSES,
                    excluded_classes=EXCLUDED_CLARIFIER_CLASSES,
                    manifest=clarifier_manifest,
                )
            else:
                clarifier_processed, c_missing, c_used, c_excluded = 0, [], set(), []

            total_images_on_disk += on_disk
            total_images_processed += aerobic_processed + clarifier_processed

            # Track exactly which files are missing from disk
            for f in a_missing:
                global_missing_json.append(f"[{facility_name}/{unit_label} - Aerobic] {f}")
            for f in c_missing:
                global_missing_json.append(f"[{facility_name}/{unit_label} - Clarifier] {f}")
                
            for f in a_excluded:
                global_excluded_files.append(f"[{facility_name}/{unit_label} - Aerobic] {f}")
            for f in c_excluded:
                global_excluded_files.append(f"[{facility_name}/{unit_label} - Clarifier] {f}")
                
            # Track exactly which files on disk were NEVER referenced in either JSON
            ignored_paths = disk_paths - (a_used | c_used)
            for p in ignored_paths:
                global_ignored_disk.append(f"[{facility_name}/{unit_label}] {p.name}")

    print("\n=== Done ===")
    print("Aerobic zone crops per class:", aerobic_counts)
    print("Clarifier crops per class:   ", clarifier_counts)

    print_facility_breakdown("Aerobic zone", aerobic_by_facility, AEROBIC_CLASSES)
    print_facility_breakdown("Clarifier", clarifier_by_facility, CLARIFIER_CLASSES)

    print("\n=== Image Diagnostic Report ===")
    print(f"Images found on disk (all units):      {total_images_on_disk}")
    print(f"Images successfully processed:          {total_images_processed}")
    
    if global_missing_json:
        print("\n[!] MISSING FILES: Expected by JSON, but not found in folder:")
        for f in sorted(set(global_missing_json)):  # set to remove duplicates if missing in both JSONs
            print(f"  - {f}")
            
    if global_ignored_disk:
        print("\n[!] IGNORED FILES: Present in folder, but JSON doesn't ask for them:")
        for f in sorted(global_ignored_disk):
            print(f"  - {f}")
            
    if global_excluded_files:
        print("\n[!] EXCLUDED FILES: Skipped because they belong to an excluded class:")
        for f in sorted(set(global_excluded_files)):
            print(f"  - {f}")
            
    if not global_missing_json and not global_ignored_disk and not global_excluded_files:
        print("\nAll files perfectly matched between disk and JSON annotations, with no excluded classes!")

    print(f"\nOutput written to: {OUTPUT_ROOT.resolve()}")

    manifest_path = OUTPUT_ROOT / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {"aerobic_zone": aerobic_manifest, "clarifier": clarifier_manifest},
            f,
            indent=2,
        )
    print(f"Crop manifest written to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()