# ===========================
# Runs all python files into 1 go
# ===========================

from setup import download_data, organize_dataset, convert_coco_to_yolo
from train import train_baseline

def main():
    # Download imgs from Google Drive
    download_data.main()

    # Split dataset into (train, test, val)
    organize_dataset.main()

    # Convert COCO-style annotations into YOLOv8 format
    convert_coco_to_yolo.main()

    # Train model
    train_baseline.main()

if __name__ == "__main__":
    main()