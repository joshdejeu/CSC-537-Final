# ============================
# Trains the baseline model (to compare later)
# ============================

import os
import shutil
from ultralytics import YOLO
from pathlib import Path

# Pretrained model options
YOLO_MODELS = [
    "yolov8n-seg.pt",  # [0] Nano
    "yolov8s-seg.pt",  # [1] Small (default)
    "yolov8m-seg.pt",  # [2] Medium
    "yolov8l-seg.pt",  # [3] Large
    "yolov8x-seg.pt"   # [4] X-Large
]

ROOT_DIR = os.getcwd()
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models","yolo", "pt_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

def move_to_weights_dir(filename):
    src = os.path.join(ROOT_DIR, filename)
    dst = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    return dst


def main():
    print("Available pretrained YOLOv8 segmentation models:")
    print(" - [0] yolov8n-seg.pt  (Nano)")
    print(" - [1] yolov8s-seg.pt  (Small - default)")
    print(" - [2] yolov8m-seg.pt  (Medium)")
    print(" - [3] yolov8l-seg.pt  (Large)")
    print(" - [4] yolov8x-seg.pt  (X-Large)\n")

    # Get user selection
    selected = input("Select a pretrained model [1]: ").strip()
    if selected not in ["0", "1", "2", "3", "4"]:
        selected = "1"  # default

    weights = YOLO_MODELS[int(selected)]
    # Run name
    name = input("Enter run name [baseline_mju]: ").strip() or "baseline_mju"

    # Image size
    try:
        imgsz = int(input("Enter image size [640]: ").strip() or 640)
    except ValueError:
        imgsz = 640

    # Epochs
    try:
        epochs = int(input("Enter number of epochs [50]: ").strip() or 50)
    except ValueError:
        epochs = 50

    print(f"\nTraining with: {weights} | img size: {imgsz} | epochs: {epochs} | run name: {name}")

    # Move pretrained weights out of root directory
    weights_path = move_to_weights_dir(weights)

    # Train model using relocated weight path
    model = YOLO(weights_path)
    model.train(
        data="datasets/mju.yaml",
        project="output/yolo", # Output directory
        epochs=epochs,
        imgsz=imgsz, # Must be a multiple of 32
        name=name,
        batch=8,            # 12gb VRAM
        workers=4,          # 6 core/12 thread CPU
        lr0=0.002,         # Safer learning rate when using smaller batches
        optimizer="AdamW",  # Handles smaller batches better than SGD
        amp=False           # (optional, if you get mixed precision errors)
    )

if __name__ == "__main__":
    main()

