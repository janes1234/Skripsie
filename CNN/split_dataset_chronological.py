"""
Chronological split: images captured before a cutoff date go to train/val,
images captured from the cutoff onward go to test.

This matches how the system is used: every facility is already in the
training data, and new images are always newer than the training images. It
also keeps crops of the same satellite image (same date and conditions) out of
both train and test, which a random split of the crops would not.

Pre-cutoff images are grouped by (facility, unit, class), shuffled within each
group and split by val_ratio, as in split_dataset_facility.py.

Usage:
    python split_dataset_chronological.py --list            # images per facility, year and class
    python split_dataset_chronological.py --cutoff 2020-01

Output: ../cnn_dataset_split-chronological/{component}/{split}/{class}/, so
wwtw_utils.get_test_set_id() names its test set "chronological".
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from crop_names import parse_crop_date, get_facility, get_facility_and_unit
from split_dataset_facility import split_group_train_val, MIN_GROUP_SIZE_FOR_GUARANTEE

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT = Path("../cnn_dataset_split-chronological")
COMPONENTS = ["clarifier", "aerobic_zone"]

DEFAULT_VAL_RATIO = 0.20
RANDOM_SEED = 42
# ---------------------------------------------------------

ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]


def iter_component_images(component_dir: Path):
    """Yields (class_name, path, facility, year, month) for every image."""
    for class_dir in sorted(component_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.glob("*.*"):
            year, month = parse_crop_date(p.name)
            yield class_dir.name, p, get_facility(p.name), year, month


def print_date_distribution():
    for component in COMPONENTS:
        component_dir = INPUT_ROOT / component
        if not component_dir.exists():
            print(f"\n[skip] no data for component '{component}'")
            continue

        # facility -> year -> class -> count
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        classes = set()
        for class_name, _, facility, year, _ in iter_component_images(component_dir):
            counts[facility][year][class_name] += 1
            classes.add(class_name)
        classes = sorted(classes)

        print(f"\n=== {component}: images per facility, per year, per class ===")
        for facility in ALL_FACILITIES:
            if facility not in counts:
                continue
            years = sorted(counts[facility])
            print(f"\n  {facility} ({years[0]}-{years[-1]}):")
            header = f"    {'Year':<6}" + "".join(f"{c:>15}" for c in classes) + f"{'Total':>10}"
            print(header)
            for year in years:
                row_counts = counts[facility][year]
                total = sum(row_counts.values())
                row = f"    {year:<6}" + "".join(f"{row_counts.get(c, 0):>15}" for c in classes) + f"{total:>10}"
                print(row)


def build_split(cutoff, val_ratio, seed):
    random.seed(seed)
    if OUTPUT_ROOT.exists():
        print(f"Cleaning up existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    thin_val_coverage = []
    facility_empty_side = []  # (component, facility, side) with zero images

    for component in COMPONENTS:
        component_dir = INPUT_ROOT / component
        if not component_dir.exists():
            print(f"\n[skip] no data for component '{component}'")
            continue
        print(f"\nProcessing {component}...")

        pre_cutoff_by_group = defaultdict(list)   # (facility_unit, class) -> [paths]
        post_cutoff = defaultdict(list)           # class -> [paths]
        facility_train_count = defaultdict(int)
        facility_test_count = defaultdict(int)

        for class_name, path, facility, year, month in iter_component_images(component_dir):
            if (year, month) >= cutoff:
                post_cutoff[class_name].append(path)
                facility_test_count[facility] += 1
            else:
                fac_unit = get_facility_and_unit(path.name)
                pre_cutoff_by_group[(fac_unit, class_name)].append(path)
                facility_train_count[facility] += 1

        split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
        for (fac_unit, class_name), images in sorted(pre_cutoff_by_group.items()):
            images.sort(key=lambda p: p.name)
            random.shuffle(images)
            train_imgs, val_imgs = split_group_train_val(images, val_ratio)
            for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
                dest_dir = OUTPUT_ROOT / component / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in split_imgs:
                    shutil.copy2(img, dest_dir / img.name)
                split_counts[class_name][split_name] += len(split_imgs)
            if len(val_imgs) == 0 and len(images) >= MIN_GROUP_SIZE_FOR_GUARANTEE:
                thin_val_coverage.append(f"{component}/{class_name}/{fac_unit}: {len(images)} total, 0 val")

        for class_name, images in sorted(post_cutoff.items()):
            dest_dir = OUTPUT_ROOT / component / "test" / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, dest_dir / img.name)
            split_counts[class_name]["test"] += len(images)

        for class_name in sorted(split_counts):
            c = split_counts[class_name]
            print(f"  {class_name:<15} -> Train: {c['train']:4d}, Val: {c['val']:4d}, Test: {c['test']:4d}")

        for facility in ALL_FACILITIES:
            if facility_train_count.get(facility, 0) == 0 and facility_test_count.get(facility, 0) == 0:
                continue  # no data for this component at all
            if facility_train_count.get(facility, 0) == 0:
                facility_empty_side.append((component, facility, "train/val (pre-cutoff)"))
            if facility_test_count.get(facility, 0) == 0:
                facility_empty_side.append((component, facility, "test (post-cutoff)"))

    print(f"\nDataset successfully split into: {OUTPUT_ROOT.resolve()}")

    if facility_empty_side:
        print("\n=== WARNING: a facility has no images on one side of the cutoff ===")
        for component, facility, side in facility_empty_side:
            print(f"  - {component}/{facility}: no images in {side}")
    else:
        print("\nEvery facility has at least one image on both sides of the cutoff.")

    if thin_val_coverage:
        print("\n=== Thin val coverage warning ===")
        for line in thin_val_coverage:
            print(f"  - {line}")


def parse_cutoff(s: str) -> tuple:
    year, month = s.split("-")
    return int(year), int(month)


def main():
    global COMPONENTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="Print per-facility, per-year, per-class image counts and exit (no split performed)")
    parser.add_argument("--component", choices=COMPONENTS, default=None,
                        help="Only report/split this component (default: both)")
    parser.add_argument("--cutoff", type=parse_cutoff, default=None,
                        help="Cutoff as YYYY-MM. Images from this month onward go to test.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help=f"Fraction of the pre-cutoff images to put in val (default: {DEFAULT_VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.component:
        COMPONENTS = [args.component]

    if args.list:
        print_date_distribution()
        return

    if args.cutoff is None:
        parser.error("No cutoff given. Run with --list to see the date distribution, then pass --cutoff YYYY-MM.")

    build_split(args.cutoff, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
