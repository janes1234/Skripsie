"""
Leave-one-facility-out split: one facility is the test set, and the other
facilities are split into train and val.

Train/val images are grouped by (facility, unit, class), shuffled within each
group and split by val_ratio, so every remaining facility, unit and class is
represented in both train and val.

Usage:
    python split_dataset_facility.py --list            # images per facility and class
    python split_dataset_facility.py --test CapeFlats

Output: ../cnn_dataset_split_facility_test-{TEST}/{component}/{split}/{class}/
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

from crop_names import get_facility, get_facility_and_unit

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT_TEMPLATE = "../cnn_dataset_split_facility_test-{test}"
COMPONENTS = ["aerobic_zone", "clarifier"]

ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]

DEFAULT_VAL_RATIO = 0.20

# A group with at least this many images always gets at least one val image.
MIN_GROUP_SIZE_FOR_GUARANTEE = 3

RANDOM_SEED = 42
# ---------------------------------------------------------


def gather_facility_class_counts(component_dir: Path):
    """facility -> class -> count."""
    counts = defaultdict(lambda: defaultdict(int))
    if not component_dir.exists():
        return counts
    for class_dir in sorted(component_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.*"):
            fac = get_facility(img_path.name)
            counts[fac][class_dir.name] += 1
    return counts


def print_facility_summary():
    for component in COMPONENTS:
        component_dir = INPUT_ROOT / component
        counts = gather_facility_class_counts(component_dir)
        if not counts:
            continue
        classes = sorted({c for cls in counts.values() for c in cls})
        print(f"\n{component} — images per facility per class:")
        header = f"  {'Facility':<18}" + "".join(f"{c:>15}" for c in classes) + f"{'Total':>10}"
        print(header)
        for fac in ALL_FACILITIES:
            row_counts = counts.get(fac, {})
            total = sum(row_counts.values())
            row = f"  {fac:<18}" + "".join(f"{row_counts.get(c, 0):>15}" for c in classes) + f"{total:>10}"
            print(row)
        missing = [fac for fac in ALL_FACILITIES if fac not in counts]
        if missing:
            print(f"  (no {component} data at all for: {', '.join(missing)})")


def split_group_train_val(images, val_ratio, min_group_size_for_guarantee=MIN_GROUP_SIZE_FOR_GUARANTEE):
    """Splits one already shuffled group into (train_imgs, val_imgs)."""
    n_total = len(images)
    if n_total == 0:
        return [], []

    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    if (
        min_group_size_for_guarantee is not None
        and n_total >= min_group_size_for_guarantee
        and n_val == 0
        and n_train > 1
    ):
        n_val = 1
        n_train -= 1

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    return train_imgs, val_imgs


def split_facility(test_facility, component_filter=None, val_ratio=DEFAULT_VAL_RATIO, seed=RANDOM_SEED):
    if test_facility not in ALL_FACILITIES:
        raise ValueError(f"Unknown facility '{test_facility}'. Choose from {ALL_FACILITIES}.")

    random.seed(seed)

    train_val_facilities = [f for f in ALL_FACILITIES if f != test_facility]
    print(f"Test facility (fully held out):     {test_facility}")
    print(f"Train/val facilities (mixed, {int((1 - val_ratio) * 100)}/{int(val_ratio * 100)}): {train_val_facilities}")

    output_root = Path(OUTPUT_ROOT_TEMPLATE.format(test=test_facility))
    if output_root.exists():
        print(f"Cleaning up existing output directory: {output_root}")
        shutil.rmtree(output_root)

    components = [component_filter] if component_filter else COMPONENTS
    thin_val_coverage = []
    empty_test_component = []

    for component in components:
        component_dir = INPUT_ROOT / component
        if not component_dir.exists():
            print(f"\n[skip] no data found for component '{component}'")
            continue
        print(f"\nProcessing {component}...")

        counts = gather_facility_class_counts(component_dir)
        if sum(counts.get(test_facility, {}).values()) == 0:
            empty_test_component.append(f"{component}/{test_facility}")

        for class_dir in sorted(component_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            split_counts = {"train": 0, "val": 0, "test": 0}

            test_imgs = [
                p for p in class_dir.glob("*.*") if get_facility(p.name) == test_facility
            ]
            for img_path in test_imgs:
                dest_dir = output_root / component / "test" / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)
                split_counts["test"] += 1

            grouped = defaultdict(list)
            for img_path in class_dir.glob("*.*"):
                fac = get_facility(img_path.name)
                if fac == test_facility:
                    continue
                fac_unit = get_facility_and_unit(img_path.name)
                grouped[fac_unit].append(img_path)

            for fac_unit, images in sorted(grouped.items()):
                # Sort before shuffling so the result does not depend on the
                # order the file system lists the files in.
                images.sort(key=lambda p: p.name)
                random.shuffle(images)

                train_imgs, val_imgs = split_group_train_val(images, val_ratio)

                for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
                    dest_dir = output_root / component / split_name / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for img in split_imgs:
                        shutil.copy2(img, dest_dir / img.name)
                    split_counts[split_name] += len(split_imgs)

                if len(val_imgs) == 0:
                    thin_val_coverage.append(
                        f"{component}/{class_name}/{fac_unit}: "
                        f"{len(images)} total -> train {len(train_imgs)}, val {len(val_imgs)}"
                    )

            print(f"  {class_name:<15} -> Train: {split_counts['train']:4d}, "
                  f"Val: {split_counts['val']:4d}, Test: {split_counts['test']:4d}")

    print(f"\nDataset successfully split into: {output_root.resolve()}")

    if empty_test_component:
        print("\n=== Empty test facility warning ===\n"
              "The held-out facility has no data for this component, so its test set is empty:")
        for line in sorted(set(empty_test_component)):
            print(f"  - {line}")

    if thin_val_coverage:
        print("\n=== Thin val coverage warning ===\n"
              "These (component/class/facility_unit) groups have no val images:")
        for line in thin_val_coverage:
            print(f"  - {line}")
    else:
        print("\nNo train/val groups with zero val coverage.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test", help=f"Facility to hold out entirely for testing. One of {ALL_FACILITIES}")
    parser.add_argument("--component", choices=COMPONENTS, default=None,
                        help="Only split this component (default: both aerobic_zone and clarifier)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help=f"Fraction of the non-test facilities' images to put in val (default: {DEFAULT_VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for the train/val shuffle (default: {RANDOM_SEED})")
    parser.add_argument("--list", action="store_true",
                        help="Print per-facility, per-class image counts and exit (no split performed)")
    args = parser.parse_args()

    if args.list:
        print_facility_summary()
        return

    if not args.test:
        parser.error("--test is required unless --list is given.")

    split_facility(args.test, component_filter=args.component, val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
