# =====================================================================
# Splits MJU-Waste original and mask images into train/test/val folders
# =====================================================================

import os
import shutil

# Paths
output_original_dir = "datasets/images" # Original images output folder with sub folders (train, test, val)
output_mask_dir = "datasets/masks"  # Mask images output folder with sub folders (train, test, val)
tmp_jpeg_dir = "tmp/JPEGImages" # Big folder with ALL images
tmp_seg_dir = "tmp/ImageSets/Segmentation" # Specifies which image go in which folders
tmp_mask_dir = "tmp/SegmentationClass" # Pre-generated masks for each image

splits = ['train', 'val', 'test']

def main():
    from tqdm import tqdm # For progress bars

    for split in splits:
        split_file = os.path.join(tmp_seg_dir, f"{split}.txt") # TXT files specifying splits
        original_out_dir = os.path.join(output_original_dir, split)
        mask_out_dir = os.path.join(output_mask_dir, split)

        os.makedirs(original_out_dir, exist_ok=True) # Ensure "datasets/images" folder exists
        os.makedirs(mask_out_dir, exist_ok=True) # Ensure "datasets/masks" folder exists

        # Open TXT file, read folder specific file names (which images go in which folder)
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        # Get image file name "image_id" and display progress bar of current split
        for image_id in tqdm(lines, desc=f"Splitting {split} images"):
            # Copy images from collective folder to specific folder in datasets/(train, test, val)
            original_src_file = os.path.join(tmp_jpeg_dir, f"{image_id}.png")
            mask_src_file = os.path.join(tmp_mask_dir, f"{image_id}.png")
            
            original_dst_file = os.path.join(original_out_dir, f"{image_id}.png")
            mask_dst_file = os.path.join(mask_out_dir, f"{image_id}.png")

            if os.path.exists(original_src_file):
                shutil.copy(original_src_file, original_dst_file)
            else:
                print(f"[!] Missing file: {original_src_file}")

            if os.path.exists(mask_src_file):
                shutil.copy(mask_src_file, mask_dst_file)
            else:
                print(f"[!] Missing file: {mask_src_file}")

    print("All splits complete.")

if __name__ == "__main__":
    main()