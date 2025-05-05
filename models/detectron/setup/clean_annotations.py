import json, os
from pathlib import Path

INPUT_DIR = Path("datasets/COCO_annotations")
OUTPUT_DIR = Path("tmp/detectron2_annotations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fix_segmentation(seg):
    if isinstance(seg, list):
        if seg and isinstance(seg[0], (int, float)):
            return [seg]  # flat list of coords into valid polygon
        elif seg and isinstance(seg[0], list):
            return seg  # already valid
    return None  # invalid

def fix_file(filename):
    with open(INPUT_DIR / filename, 'r') as f:
        data = json.load(f)

    # Fix annotations
    for ann in data["annotations"]:
        fixed = fix_segmentation(ann.get("segmentation"))
        if fixed:
            ann["segmentation"] = fixed
        else:
            print(f"[SKIPPED] Bad segmentation in ann {ann['id']}")
            ann["segmentation"] = []  # Optional: remove or skip

        # Update category_id
        if ann["category_id"] == 0:
            ann["category_id"] = 1

    # Replace categories list
    data["categories"] = [
        {
            "id": 1,
            "name": "Rubbish",
            "supercategory": "Waste"
        }
    ]

    with open(OUTPUT_DIR / filename, 'w') as f:
        json.dump(data, f)

    print(f"Fixed {filename}")

# Run for all splits
for split in ["train.json", "val.json", "test.json"]:
    fix_file(split)
