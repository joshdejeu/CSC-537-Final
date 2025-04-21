# ==========================================================
# Use pre-trained YOLO weights to make predictions on images
# ==========================================================

import os
import shutil
from ultralytics import YOLO

# NOTE : You have to import your own images and manually change the name each time you run this script

ROOT_DIR = os.getcwd()
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models", "yolo", "pt_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Moves downloaded pretrained weights to a permanent directory
def move_to_weights_dir(filename):
    src = os.path.join(ROOT_DIR, filename)
    dst = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    return dst

# Pretrained model options
YOLO_MODELS = [
    "yolov8n-seg.pt",  # [0] Nano
    "yolov8s-seg.pt",  # [1] Small (default)
    "yolov8m-seg.pt",  # [2] Medium
    "yolov8l-seg.pt",  # [3] Large
    "yolov8x-seg.pt"   # [4] X-Large
]

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

# Move pretrained weights out of root directory
weights_path = move_to_weights_dir(weights)
model = YOLO(weights_path)

# Get confidence threshold from user
# Default confidence threshold is 0.1
confident_input = input("\nEnter confidence threshold (default 0.1): ")
if confident_input.strip() == "":
    confident_input = 0.1
else:
    try:
        confident_input = float(confident_input)
    except ValueError:
        print("[!] Invalid input, defaulting to 0.1.")
        confident_input = 0.1

# NOTE : You have to import your own images and manually change the name each time you run this script
results = model(r"input\box.jpg", conf=confident_input) # TODO : Make this dynamic for all images

results = model(r"input\box.jpg", conf=0.1) # Make this dynamic for all images

results[0].show()  # show prediction with boxes/masks

# Print all detections
# print(results[0].boxes) # Bounding boxes
# print(results[0].masks) # Segmentation masks
