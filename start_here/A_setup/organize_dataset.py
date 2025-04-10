# ============================
# Splits MJU-Waste Images into train/test/val folders
# ============================

import os
import shutil

# Paths
root_dir = "datasets/images" # Output folder with sub folders (train, test, val)
jpeg_dir = "tmp/JPEGImages" # Big folder with ALL images
imagesets_dir = "tmp/ImageSets/Segmentation" # Specifies which image go in which folders

splits = ['train', 'val', 'test']

for split in splits:
    print(f"Splitting: {split.upper()}")
    split_file = os.path.join(imagesets_dir, f"{split}.txt")
    out_dir = os.path.join(root_dir, split)
    os.makedirs(out_dir, exist_ok=True) # Check if folders exist

    # Folder specific file names
    with open(split_file, "r") as f:
        lines = f.read().splitlines()

    # Copy images from big folder to specific folder (train, test, val)
    for image_id in lines:
        src_file = os.path.join(jpeg_dir, f"{image_id}.png")
        dst_file = os.path.join(out_dir, f"{image_id}.png")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            print(f"[!] Missing file: {src_file}")

print("All splits complete.")