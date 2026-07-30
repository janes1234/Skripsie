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
            Empty/
        clarifier/
            Functional/
            Dysfunctional/
            Scum/
            Empty/
            Stagnant/

"""

import json
from pathlib import Path
from PIL import Image

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
}

LABELS_SUBDIR = "Labels"

# ------------------------------------------------------------------------------


AEROBIC_CLASSES = {"Functional", "Suboptimal", "Dysfunctional", "Empty"}
CLARIFIER_CLASSES = {"Functional", "Dysfunctional", "Scum", "Empty", "Stagnant"}


def normalize_label(raw_label: str) -> str:
    return raw_label.strip().capitalize()


def load_via2_json(path: Path) -> dict:
    """Load a VIA2 project JSON file and return its image metadata dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # VIA2 project files nest the per-image entries under this key
    return data.get("_via_img_metadata", data)


def region_bbox(shape_attributes: dict) -> tuple[int, int, int, int]:
    """
    Return a (left, top, right, bottom) crop box for either a VIA2 rectangle
    or circle region.
    """
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
    """Clamp a crop box to the image bounds so PIL doesn't error on edge cases."""
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)
    return left, top, right, bottom


def process_json_file(
    json_path: Path,
    images_dir: Path,
    attr_key: str,
    output_component_root: Path,
    facility_name: str,
    unit_label: str,
    counters: dict,
    valid_classes: set,
):
    """Crop every region in one VIA2 JSON file and save it under its class folder."""
    if not json_path.exists():
        print(f"  [skip] label file not found: {json_path}")
        return

    img_metadata = load_via2_json(json_path)

    # One-time diagnostic: if the very first expected image isn't found,
    # print what's actually in the folder so filename mismatches are easy to spot
    if img_metadata and not images_dir.exists():
        print(f"  [warn] images directory does not exist at all: {images_dir}")
    elif img_metadata:
        first_entry = next(iter(img_metadata.values()))
        first_filename = first_entry.get("filename", "")
        if first_filename and not (images_dir / first_filename).exists():
            actual_files = sorted(p.name for p in images_dir.glob("*"))[:10]
            print(f"  [diagnostic] expected filename not found: '{first_filename}'")
            print(f"  [diagnostic] first 10 actual files in {images_dir}:")
            for f in actual_files:
                print(f"      {f}")

    for entry in img_metadata.values():
        filename = entry.get("filename")
        if not filename:
            continue

        image_path = images_dir / filename
        if not image_path.exists():
            print(f"  [warn] image not found, skipping: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not open {image_path}: {e}")
            continue

        for i, region in enumerate(entry.get("regions", [])):
            shape_attrs = region.get("shape_attributes", {})
            region_attrs = region.get("region_attributes", {})
            label = region_attrs.get(attr_key)

            if not label:
                print(f"  [warn] no '{attr_key}' label on region {i} in {filename}, skipping")
                continue

            label = normalize_label(label)
            if label not in valid_classes:
                print(
                    f"  [ALERT] unexpected label '{label}' on region {i} in {filename} "
                    f"(not in {sorted(valid_classes)}) — check this annotation manually"
                )

            try:
                box = region_bbox(shape_attrs)
            except ValueError as e:
                print(f"  [warn] {e} in {filename}, skipping region")
                continue

            box = clamp_box(box, img.width, img.height)
            if box[2] <= box[0] or box[3] <= box[1]:
                print(f"  [warn] degenerate crop box in {filename}, skipping region")
                continue

            crop = img.crop(box)

            class_dir = output_component_root / label
            class_dir.mkdir(parents=True, exist_ok=True)

            # Sanitize unit_label so slashes/spaces in e.g. "Images/CF1" don't
            # get interpreted as subfolders or cause filesystem issues
            safe_unit_label = unit_label.replace("/", "-").replace(" ", "_")
            out_name = f"{facility_name}_{safe_unit_label}_{Path(filename).stem}_r{i}.jpg"
            crop.save(class_dir / out_name, quality=95)

            counters[label] = counters.get(label, 0) + 1


def main():
    aerobic_counts = {}
    clarifier_counts = {}

    for facility_name, facility_cfg in FACILITIES.items():
        facility_root = RAW_DATA_ROOT / facility_name
        labels_root = facility_root / LABELS_SUBDIR if LABELS_SUBDIR else facility_root

        print(f"\n=== {facility_name} ===")

        for unit in facility_cfg["units"]:
            unit_label = unit["images_dir"] if unit["images_dir"] else facility_name
            images_dir = facility_root / unit["images_dir"] if unit["images_dir"] else facility_root

            print(f" -- unit: {unit_label} ({images_dir})")

            aerobic_json = labels_root / unit["aerobic_json"]
            clarifier_json = labels_root / unit["clarifier_json"]

            process_json_file(
                json_path=aerobic_json,
                images_dir=images_dir,
                attr_key=AEROBIC_ATTR_KEY,
                output_component_root=OUTPUT_ROOT / "aerobic_zone",
                facility_name=facility_name,
                unit_label=unit_label,
                counters=aerobic_counts,
                valid_classes=AEROBIC_CLASSES,
            )

            process_json_file(
                json_path=clarifier_json,
                images_dir=images_dir,
                attr_key=CLARIFIER_ATTR_KEY,
                output_component_root=OUTPUT_ROOT / "clarifier",
                facility_name=facility_name,
                unit_label=unit_label,
                counters=clarifier_counts,
                valid_classes=CLARIFIER_CLASSES,
            )

    print("\n=== Done ===")
    print("Aerobic zone crops per class:", aerobic_counts)
    print("Clarifier crops per class:   ", clarifier_counts)
    print(f"\nOutput written to: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()