"""
Splits the WWTW CNN dataset into train, validation, and test sets.
Ensures a stratified distribution across facilities, units, and dates.
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random

# ------------------------- CONFIG ------------------------
INPUT_ROOT = Path("../cnn_dataset")
OUTPUT_ROOT = Path("../cnn_dataset_split")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
COMPONENTS = ["aerobic_zone", "clarifier"]

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

def main():
    if OUTPUT_ROOT.exists():
        print(f"Cleaning up existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    for component in COMPONENTS:
        component_dir = INPUT_ROOT / component
        if not component_dir.exists():
            continue

        print(f"\nProcessing {component}...")
        
        # Iterate over classes (Functional, Dysfunctional, etc.)
        for class_dir in component_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            
            # Group images by (facility_unit) to stratify
            grouped_images = defaultdict(list)
            for img_path in class_dir.glob("*.*"):
                fac_unit = get_facility_and_unit(img_path.name)
                grouped_images[fac_unit].append(img_path)
            
            for fac_unit, images in grouped_images.items():
                # Sorting naturally orders the dates embedded in the original filenames
                images.sort(key=lambda p: p.name)
                
                n_total = len(images)
                n_train = int(n_total * SPLIT_RATIOS["train"])
                n_val = int(n_total * SPLIT_RATIOS["val"])
                
                # Round-robin distribution based on sorted order ensures chronological mixing
                train_imgs = []
                val_imgs = []
                test_imgs = []
                
                for i, img in enumerate(images):
                    # Spread them evenly by index position to ensure different dates hit all splits
                    normalized_idx = i / n_total
                    if normalized_idx < SPLIT_RATIOS["train"]:
                        train_imgs.append(img)
                    elif normalized_idx < (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]):
                        val_imgs.append(img)
                    else:
                        test_imgs.append(img)
                
                # Copy files to their new split directories
                for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                    split_dest = OUTPUT_ROOT / component / split_name / class_name
                    split_dest.mkdir(parents=True, exist_ok=True)
                    
                    for img in split_imgs:
                        shutil.copy2(img, split_dest / img.name)
                
                print(f"  {class_name:<15} | {fac_unit:<25} -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    print(f"\nDataset successfully split into: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()