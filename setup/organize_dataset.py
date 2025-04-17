# ============================
# Splits MJU-Waste Images into train/test/val folders
# ============================

import os
import shutil

# Paths
root_dir = "datasets/images" # Default image output folder with sub folders (train, test, val)
mask_output_dir = "datasets/masks"  # Mask image output folder with sub folders (train, test, val)
jpeg_dir = "tmp/JPEGImages" # Big folder with ALL images
imagesets_dir = "tmp/ImageSets/Segmentation" # Specifies which image go in which folders
segclass_mask_dir = "tmp/SegmentationClass" # Pre-generated masks for each image


splits = ['train', 'val', 'test']

def main():
    for split in splits:
        print(f"Splitting: {split}")

        split_file = os.path.join(imagesets_dir, f"{split}.txt") # TXT files specifying splits
        default_out_dir = os.path.join(root_dir, split)
        mask_out_dir = os.path.join(mask_output_dir, split)

        os.makedirs(default_out_dir, exist_ok=True) # Ensure "datasets/images" folder exists
        os.makedirs(mask_out_dir, exist_ok=True) # Ensure "datasets/masks" folder exists

        # Open TXT file, read folder specific file names (which images go in which folder)
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        # Copy images from collective folder to specific folder in datasets/(train, test, val)
        # Get image file name "image_id"
        for image_id in lines:
            default_src_file = os.path.join(jpeg_dir, f"{image_id}.png")
            mask_src_file = os.path.join(segclass_mask_dir, f"{image_id}.png")
            
            default_dst_file = os.path.join(default_out_dir, f"{image_id}.png")
            mask_dst_file = os.path.join(mask_out_dir, f"{image_id}.png")

            if os.path.exists(default_src_file):
                shutil.copy(default_src_file, default_dst_file)
            else:
                print(f"[!] Missing file: {default_src_file}")

            if os.path.exists(mask_src_file):
                shutil.copy(mask_src_file, mask_dst_file)
            else:
                print(f"[!] Missing file: {mask_src_file}")

    print("All splits complete.")

if __name__ == "__main__":
    main()