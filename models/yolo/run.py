# ===========================
# Runs all python files into 1 go
# ===========================

from setup import convert_coco_to_yolo
from train import train_baseline

def main():
    # Convert COCO-style annotations into YOLOv8 format
    convert_coco_to_yolo.main()

    # Train model
    train_baseline.main()

if __name__ == "__main__":
    main()