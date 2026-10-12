"""
Chronological split for the WWTW CNN dataset: train/val on images captured
before a cutoff date, test on images captured after it.

This estimates the "known facility, new images over time" scenario the
dashboard actually runs in -- a professional relabels flagged images and
they're added back to training, so by the time the model sees a new batch
of satellite images every facility is already in the training data, just
not those specific (newer) images. The stratified 70/20/10 split
(split_dataset.py) is optimistic for that scenario for two reasons: it
splits on cropped images rather than source images, so a test crop can come
from the very same satellite image (same date, same lighting/quality) as a
training crop of a different component on that image; and it mixes dates
randomly across train/test, whereas in the dashboard the model only ever
classifies images newer than everything it was trained on.

Every facility's capture dates are parsed from its crop filenames via
date_utils.parse_crop_date() (see that module for the per-facility filename
formats). TRAIN and VAL are drawn from every pre-cutoff image, stratified by
(facility, unit, class) the same way split_dataset.py/split_dataset_facility.py
already do (grouped, shuffled within each group, split by val_ratio) --
keeping every facility represented in both train and val. TEST is every
post-cutoff image, straight copy, no stratification needed.

Usage:
    # Step A: see the date distribution first (no files written) -- use this
    # to pick a cutoff before building anything:
    python split_dataset_chronological.py --list

    # Step B: build the split with a single global cutoff (YYYY-MM, inclusive
    # of everything from that month onward going to test):
    python split_dataset_chronological.py --cutoff 2021-01

    # Or a per-facility cutoff, if a single global date doesn't give every
    # facility a reasonable number of post-cutoff test images:
    python split_dataset_chronological.py --cutoffs Atlantis=2021-01,CapeFlats=2022-06,...

Output goes to ../cnn_dataset_split-chronological/{component}/{split}/{class}/
-- this naming keeps wwtw_utils.get_test_set_id() working unchanged (it
takes whatever follows the last "-" in the folder name as the test set id,
so this becomes "chronological").
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from date_utils import parse_crop_date, get_facility

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT = Path("../cnn_dataset_split-chronological")

# The chronological split is the only non-LOFO experiment -- it replaced the
# random 70/20/10 stratified split entirely (split_dataset.py still exists
# as a general-purpose utility, just unused by the pipeline now).
COMPONENTS = ["clarifier", "aerobic_zone"]

DEFAULT_VAL_RATIO = 0.20
MIN_GROUP_SIZE_FOR_GUARANTEE = 3
RANDOM_SEED = 42
# ---------------------------------------------------------

ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]


def get_facility_and_unit(filename: str) -> str:
    """Same convention as the other split scripts."""
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "Unknown_Unit"


def iter_component_images(component_dir: Path):
    """Yields (class_name, path, facility, year, month) for every image."""
    for class_dir in sorted(component_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.glob("*.*"):
            year, month = parse_crop_date(p.name)
            yield class_dir.name, p, get_facility(p.name), year, month


# ------------------------- Step A: --list ------------------------

def print_date_distribution():
    for component in COMPONENTS_FOR_LIST:
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


# ------------------------- Step B: build the split ------------------------

def split_group_train_val(images, val_ratio, min_group_size_for_guarantee=MIN_GROUP_SIZE_FOR_GUARANTEE):
    """Mirrors split_group_train_val() in split_dataset_facility.py."""
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
    return images[:n_train], images[n_train:n_train + n_val]


def resolve_cutoff(facility, global_cutoff, per_facility_cutoffs):
    if facility in per_facility_cutoffs:
        return per_facility_cutoffs[facility]
    if global_cutoff is not None:
        return global_cutoff
    raise ValueError(f"No cutoff given for facility '{facility}' -- pass --cutoff or --cutoffs")


def build_split(global_cutoff, per_facility_cutoffs, val_ratio, seed):
    random.seed(seed)
    if OUTPUT_ROOT.exists():
        print(f"Cleaning up existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    thin_val_coverage = []
    facility_empty_side = []  # (component, facility, "train"/"test") with zero images

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
            cutoff = resolve_cutoff(facility, global_cutoff, per_facility_cutoffs)
            if (year, month) >= cutoff:
                post_cutoff[class_name].append(path)
                facility_test_count[facility] += 1
            else:
                fac_unit = get_facility_and_unit(path.name)
                pre_cutoff_by_group[(fac_unit, class_name)].append(path)
                facility_train_count[facility] += 1

        # Pre-cutoff -> stratified train/val, same grouping as the other split scripts.
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

        # Post-cutoff -> straight to test.
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
                continue  # facility has no data for this component at all
            if facility_train_count.get(facility, 0) == 0:
                facility_empty_side.append((component, facility, "train/val (pre-cutoff)"))
            if facility_test_count.get(facility, 0) == 0:
                facility_empty_side.append((component, facility, "test (post-cutoff)"))

    print(f"\nDataset successfully split into: {OUTPUT_ROOT.resolve()}")

    if facility_empty_side:
        print(
            "\n=== WARNING: a facility is entirely missing from one side of the cutoff ===\n"
            "The chronological split only tells you something about a facility if it has\n"
            "images on BOTH sides. These do not:"
        )
        for component, facility, side in facility_empty_side:
            print(f"  - {component}/{facility}: no images in {side}")
    else:
        print("\nEvery facility has at least one image on both sides of its cutoff.")

    if thin_val_coverage:
        print("\n=== Thin val coverage warning ===")
        for line in thin_val_coverage:
            print(f"  - {line}")


def parse_cutoff(s: str) -> tuple:
    year, month = s.split("-")
    return int(year), int(month)


def parse_cutoffs(s: str) -> dict:
    out = {}
    for part in s.split(","):
        facility, cutoff = part.split("=")
        out[facility.strip()] = parse_cutoff(cutoff.strip())
    return out


def main():
    global COMPONENTS_FOR_LIST
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                         help="Print per-facility, per-year, per-class image counts and exit (no split performed)")
    parser.add_argument("--component", choices=["clarifier", "aerobic_zone"], default=None,
                         help="Only report/split this component (--list default: both; build default: clarifier only)")
    parser.add_argument("--cutoff", type=parse_cutoff, default=None,
                         help="Single global cutoff as YYYY-MM. Images from this month onward go to test.")
    parser.add_argument("--cutoffs", type=parse_cutoffs, default=None,
                         help="Per-facility cutoffs, e.g. Atlantis=2021-01,CapeFlats=2022-06 "
                              "(falls back to --cutoff for any facility not listed)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                         help=f"Fraction of the pre-cutoff pool to put in val (default: {DEFAULT_VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.list:
        COMPONENTS_FOR_LIST = [args.component] if args.component else ["clarifier", "aerobic_zone"]
        print_date_distribution()
        return

    if args.component:
        global COMPONENTS
        COMPONENTS = [args.component]

    if args.cutoff is None and not args.cutoffs:
        parser.error(
            "No cutoff given. Run with --list first to see the date distribution, then pass "
            "--cutoff YYYY-MM or --cutoffs facility=YYYY-MM,..."
        )

    build_split(args.cutoff, args.cutoffs or {}, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
