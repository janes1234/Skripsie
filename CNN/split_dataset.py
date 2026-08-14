"""
Splits the WWTW CNN dataset into train, validation, and test sets.

Stratifies by (facility, unit, class) so no single facility/unit dominates a
split. Within each (facility, unit, class) group, images are randomly shuffled 
to ensure a diverse mix of capture dates across train, val, and test splits.

Small-group guarantees: if a group is large enough to plausibly contain
a val/test image but the ratio-based split would round it down to 0,
this pulls one image over from train rather than silently leaving that
class/facility combo with zero evaluation coverage. Groups too small for
even that are reported explicitly at the end so you know which classes
have thin evaluation coverage.
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT = Path("../cnn_dataset_split")

# Updated to 70/20/10 split
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
COMPONENTS = ["aerobic_zone", "clarifier"]

# If a group has at least this many images, guarantee at least 1 ends up in
# val and 1 in test even if the ratio would round down to 0. Set to None to
# disable and fall back to pure ratio-based rounding.
MIN_GROUP_SIZE_FOR_GUARANTEE = 3

# Seed for reproducibility so the random splits are identical across runs
RANDOM_SEED = 42
# ---------------------------------------------------------


def get_facility_and_unit(filename: str) -> str:
    """
    Extracts the facility and unit from the filename.
    Format is expected to be: {facility_name}_{safe_unit_label}_{original_filename}_r{i}.jpg
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        # e.g., "CapeFlats" and "Images-CF1"
        return f"{parts[0]}_{parts[1]}"
    return "Unknown_Unit"


def split_group(images, min_size_for_guarantee=MIN_GROUP_SIZE_FOR_GUARANTEE):
    """
    Split a single (facility, unit, class) group of images (already shuffled)
    into train/val/test.

    Returns (train_imgs, val_imgs, test_imgs).
    """
    n_total = len(images)

    if n_total == 0:
        return [], [], []

    # --- base random block split (ratio-based) ---
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])
    # whatever's left goes to test, so rounding doesn't drop images
    n_test = n_total - n_train - n_val

    # --- small-group guarantee: don't let val/test silently round to 0 ---
    if min_size_for_guarantee is not None and n_total >= min_size_for_guarantee:
        if n_val == 0 and n_train > 1:
            n_val = 1
            n_train -= 1
        if n_test == 0 and n_train > 1:
            n_test = 1
            n_train -= 1

    train_end = n_train
    val_start = n_train
    val_end = n_train + n_val
    test_start = val_end

    train_imgs = images[:train_end]
    val_imgs = images[val_start:val_end]
    test_imgs = images[test_start:test_start + n_test]

    return train_imgs, val_imgs, test_imgs


def main():
    # Set the random seed to ensure consistent splits across runs
    random.seed(RANDOM_SEED)
    
    if OUTPUT_ROOT.exists():
        print(f"Cleaning up existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    # Track groups that end up with zero val or zero test images, so it's
    # obvious afterwards which classes have thin evaluation coverage.
    thin_coverage = []

    for component in COMPONENTS:
        component_dir = INPUT_ROOT / component
        if not component_dir.exists():
            continue

        print(f"\nProcessing {component}...")

        for class_dir in sorted(component_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            # Group images by (facility_unit) to stratify
            grouped_images = defaultdict(list)
            for img_path in class_dir.glob("*.*"):
                fac_unit = get_facility_and_unit(img_path.name)
                grouped_images[fac_unit].append(img_path)

            for fac_unit, images in sorted(grouped_images.items()):
                # Sort first to ensure deterministic ordering before shuffling,
                # which prevents OS-level file fetching variations from breaking the seed.
                images.sort(key=lambda p: p.name)
                # Now randomize the dates/images
                random.shuffle(images)

                train_imgs, val_imgs, test_imgs = split_group(images)

                if len(val_imgs) == 0 or len(test_imgs) == 0:
                    thin_coverage.append(
                        f"{component}/{class_name}/{fac_unit}: "
                        f"{len(images)} total -> train {len(train_imgs)}, "
                        f"val {len(val_imgs)}, test {len(test_imgs)}"
                    )

                # Copy files to their new split directories
                for split_name, split_imgs in [
                    ("train", train_imgs),
                    ("val", val_imgs),
                    ("test", test_imgs),
                ]:
                    split_dest = OUTPUT_ROOT / component / split_name / class_name
                    split_dest.mkdir(parents=True, exist_ok=True)

                    for img in split_imgs:
                        shutil.copy2(img, split_dest / img.name)

                print(
                    f"  {class_name:<15} | {fac_unit:<25} -> "
                    f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
                )

    print(f"\nDataset successfully split into: {OUTPUT_ROOT.resolve()}")

    if thin_coverage:
        print(
            "\n=== Thin coverage warning ===\n"
            "The following (component/class/facility_unit) groups have zero val "
            "and/or zero test images, even after the small-group guarantee. "
            "Metrics for these classes will be based on a subset of facilities — "
            "worth keeping in mind when interpreting per-class results:"
        )
        for line in thin_coverage:
            print(f"  - {line}")
    else:
        print("\nNo groups with zero val/test coverage.")


if __name__ == "__main__":
    main()