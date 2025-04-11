# ============================
# Converting COCO annotations to YOLOv8 format
# ============================
import os
import json
from pathlib import Path

# Converts COCO format to YOLOv8
def convert_coco_to_yolo(coco_json_path, annotation_name, output_label_dir):
    # Create output label folder if it doesn't exist
    os.makedirs(output_label_dir, exist_ok=True) # Make sure label folder exists

    with open(coco_json_path, 'r') as f:
        coco = json.load(f) # Open json COCO annotations

    # Map category IDs to 0-indexed class IDs
    categories = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # Map image ID to filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        file_name = image_id_to_filename[image_id]
        label_path = os.path.join(output_label_dir, Path(file_name).stem + ".txt")

        # Get class ID (0-based)
        class_id = categories[ann["category_id"]]

        # Get segmentation polygons (YOLOv8 expects normalized xyxy... format)
        if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0:
            # Handle different "segmentation" data types
            segments = ann.get("segmentation", []) # Get segmentation polygons

            if not isinstance(segments, list) or len(segments) == 0: # Skip if not a list (RLE mask) or if empty
                continue  # Skip non-polygon segmentations

            if isinstance(segments[0], (float, int)):  # Handle flat format [x1, y1, ..., xn, yn]
                # COCO annotations may contain multiple polygon lists (generally its just 1)
                segments = [segments] # Normalization (ensures list of lists)

            with open(label_path, "a") as f:
                for seg in segments:
                    normalized_points = []
                    img_info = next((img for img in coco["images"] if img["id"] == image_id), None) # Find first img dict where id matches annotation (default None)
                    
                    if img_info is None:
                        continue  # Skip this annotation if image not found
                    
                    w, h = img_info["width"], img_info["height"]

                    # Safely iterate through polygon point coordinates
                    for i in range(0, len(seg), 2):
                        # Get x,y relative values from 0.0 to 1.0
                        x = seg[i] / w
                        y = seg[i + 1] / h
                        normalized_points.append(f"{x:.6f} {y:.6f}") # Coordinates with 6 decimal places

                    points_str = " ".join(normalized_points)
                    f.write(f"{class_id} {points_str}\n")


    print(f"Converted: {annotation_name}")

# Converts all three splits (train/test/val)
root = "datasets"
convert_coco_to_yolo(
    coco_json_path=f"{root}/annotations/train.json",
    annotation_name="train",
    output_label_dir=f"{root}/labels/train"
)
convert_coco_to_yolo(
    coco_json_path=f"{root}/annotations/val.json",
    annotation_name="val",
    output_label_dir=f"{root}/labels/val"
)
convert_coco_to_yolo(
    coco_json_path=f"{root}/annotations/test.json",
    annotation_name="test",
    output_label_dir=f"{root}/labels/test"
)
