"""
Leave-one-facility-out split for the WWTW CNN dataset, with a stratified
train/val mix across the remaining facilities.

The TEST set is a single facility, chosen by the user, held out entirely -
the model never sees a single image from it during training or model
selection. That measures generalization to a completely unseen WWTW
facility.

TRAIN and VAL are drawn from the other 4 facilities together, using the
same kind of stratification as prepare_dataset_splits.py: images are
grouped by (facility, unit, class), shuffled within each group (which mixes
capture dates across train/val so one split doesn't end up all-early or
all-late), and split by val_ratio. This keeps a healthy mix of every
non-test facility in both train and val, rather than skewing val toward
whichever facility happens to have the most images.

Usage:
    # See per-facility, per-class counts first, to pick a sensible held-out
    # test facility (some facilities/classes are much smaller than others):
    python split_data_facility.py --list

    # Hold out CapeFlats for test, split the rest 80/20 train/val:
    python split_data_facility.py --test CapeFlats

    # Only split one component, with a different val ratio:
    python split_data_facility.py --test CapeFlats --component clarifier --val-ratio 0.25

Output goes to:
    ../cnn_dataset_split_facility_test-{TEST}/{component}/{split}/{class}/
so that different held-out choices don't overwrite each other - handy if you
end up running several folds (e.g. rotating the test facility through all 5,
see the note at the bottom of this file).
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT_TEMPLATE = "../cnn_dataset_split_facility_test-{test}"
COMPONENTS = ["aerobic_zone", "clarifier"]

ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]

# Default train/val split among the non-test facilities.
DEFAULT_VAL_RATIO = 0.20

# If a (facility, unit, class) group has at least this many images, guarantee
# at least 1 ends up in val even if the ratio would round it down to 0. Set
# to None to disable and fall back to pure ratio-based rounding.
MIN_GROUP_SIZE_FOR_GUARANTEE = 3

# Seed for reproducibility so the random train/val shuffles are identical
# across runs.
RANDOM_SEED = 42
# ---------------------------------------------------------


def get_facility(filename: str) -> str:
    """The crop filename convention (see prep.py) is
    {facility}_{safe_unit_label}_{original_stem}_r{i}.jpg, so the facility
    name is always the first underscore-separated token."""
    return filename.split("_")[0]


def get_facility_and_unit(filename: str) -> str:
    """
    Extracts (facility, unit) as a single grouping key, e.g. "CapeFlats_Images-CF1".
    Same convention as prepare_dataset_splits.py, used here to stratify the
    train/val split across the non-test facilities.
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "Unknown_Unit"


def gather_facility_class_counts(component_dir: Path):
    """facility -> class -> count, used by --list to help you pick which
    facility to hold out before you commit to a split."""
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
    """
    Split a single (facility, unit, class) group of images (already sorted +
    shuffled) into train/val only. Mirrors split_group() in
    prepare_dataset_splits.py but two-way, since test is handled separately
    by facility.

    Returns (train_imgs, val_imgs).
    """
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

        # Sanity check: does the held-out test facility even have any data
        # for this component? (e.g. Fisantekraal/NoordelikeWerke have no
        # aerobic_zone data at all)
        counts = gather_facility_class_counts(component_dir)
        if sum(counts.get(test_facility, {}).values()) == 0:
            empty_test_component.append(f"{component}/{test_facility}")

        for class_dir in sorted(component_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            split_counts = {"train": 0, "val": 0, "test": 0}

            # Test images: straight copy, no stratification needed since the
            # whole facility is test.
            test_imgs = [
                p for p in class_dir.glob("*.*") if get_facility(p.name) == test_facility
            ]
            for img_path in test_imgs:
                dest_dir = output_root / component / "test" / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)
                split_counts["test"] += 1

            # Train/val images: group by (facility, unit) across the 4
            # remaining facilities, shuffle within each group to mix capture
            # dates, then split by val_ratio. This keeps every non-test
            # facility represented in both train and val.
            grouped = defaultdict(list)
            for img_path in class_dir.glob("*.*"):
                fac = get_facility(img_path.name)
                if fac == test_facility:
                    continue
                fac_unit = get_facility_and_unit(img_path.name)
                grouped[fac_unit].append(img_path)

            for fac_unit, images in sorted(grouped.items()):
                # Sort first for deterministic ordering before shuffling, so
                # OS-level file listing order doesn't affect reproducibility.
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
        print(
            "\n=== Empty test facility warning ===\n"
            "The held-out test facility has NO data at all for this component - "
            "that component's test set will be entirely empty:"
        )
        for line in sorted(set(empty_test_component)):
            print(f"  - {line}")

    if thin_val_coverage:
        print(
            "\n=== Thin val coverage warning ===\n"
            "The following (component/class/facility_unit) groups from the\n"
            "train/val facilities have zero val images, even after the\n"
            "small-group guarantee. That facility/unit's images for this class\n"
            "landed entirely in train:"
        )
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


# ------------------------------------------------------------------------
# Note on full leave-one-facility-out cross-validation
# ------------------------------------------------------------------------
# A single run of this script tells you how the model does on ONE held-out
# test facility. With only 5 facilities that's a fairly noisy estimate of
# "generalization to a new facility" in general - it's really a single
# sample from a population of 5.
#
# If your thesis timeline allows it, the more defensible version is to
# rotate the test facility through all 5 choices, train 5 separate models,
# and report mean +/- spread of test accuracy / mean AUC across folds.
# Something like:
#
#   for test_fac in ALL_FACILITIES:
#       split_facility(test_fac, val_ratio=0.20)
#       # then run your training notebooks against that fold's output dir
#
# Even if you don't have compute budget for all 5 folds, running 2-3 is far
# more convincing evidence of generalization than a single held-out
# facility, and it's easy to justify in the write-up as an acknowledged
# limitation if you only do one.