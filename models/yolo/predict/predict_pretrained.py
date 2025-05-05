# ==========================================================
# Use pre-trained YOLO weights to make predictions on images
# ==========================================================

import os
import shutil
from ultralytics import YOLO
from glob import glob

ROOT_DIR = os.getcwd()
INPUT_DIR = os.path.join(ROOT_DIR, "input")
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models", "yolo", "pt_weights")
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "predictions", "yolo", "pretrained")
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
for idx, name in enumerate(YOLO_MODELS):
    print(f" - [{idx}] {name}")

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
        
# Find all images
input_images = glob(os.path.join(INPUT_DIR, "*.*"))
if not input_images:
    print("[!] No images found in input/. Exiting.")
    exit(1)

# Set output save path
save_dir = os.path.join(BASE_OUTPUT_DIR, weights.replace(".pt", ""))
os.makedirs(save_dir, exist_ok=True)


# Predict on each image and save
for img_path in input_images:
    print(f"Predicting {img_path}...")
    results = model(img_path, conf=confident_input)
    img_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, img_name)
    results[0].save(filename=save_path)

# Print all detections
# print(results[0].boxes) # Bounding boxes
# print(results[0].masks) # Segmentation masks
