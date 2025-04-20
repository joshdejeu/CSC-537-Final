# ======================================================
# Use trained YOLO weights to make predictions on images
# ======================================================

import os
import shutil
from ultralytics import YOLO

# NOTE : You have to import your own images and manually change the name each time you run this script

# Base directory for all training runs
base_dir = "output/yolo/"

ROOT_DIR = os.getcwd()
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models", "yolo", "pt_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# List all subdirectories in the base directory (each representing a different training run)
runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Moves downloaded pretrained weights to a permanent directory
def move_to_weights_dir(filename):
    src = os.path.join(ROOT_DIR, filename)
    dst = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    return dst


# Display the available training runs
# Returns true if there are any runs, false otherwise
def are_available_runs(runs):
    if not runs:
        return False
    else:
        print("Available training runs:")
        for idx, run in enumerate(runs):
            print(f"{idx}: {run}")
        return True

# Choose a previous run to get weights from, exit on invalid input
# Return run_choice if the run choice is valid, -1 otherwise
def get_run_choice(runs):
    # Choose a previous run to get weights from, exit on invalid input
    try:
        run_choice = int(input(f"\nChoose a training run to get weights from (0-{len(runs) - 1}): "))
        if run_choice >= len(runs):
            raise ValueError("Out of range, invalid input")
        else:
            return run_choice
    except ValueError:
        print("[!] Invalid input.")
        exit(1)

# List all saved weights in specific run
# Return true if there are any saved weights, false otherwise
def display_available_runs(saved_weights):
    if not saved_weights:
        return False
    else:
        print("Saved model weights found:")
        for idx, weight_file in enumerate(saved_weights):
                print(f"{idx}: {weight_file}")
        return True

# Choose weights from previous run, default to best.pt
def get_weight_choice():
    # Ask user to choose the specific weights to use, default to best.pt
    user_input = input(f"\nChoose a weight (0-{len(saved_weights) - 1}): ")
    if user_input.strip() == "":
        return -1
    else:
        try:
            weight_choice = int(user_input)
            # Ensure valid weight choice
            if weight_choice < 0 or weight_choice >= len(saved_weights):
                return -1
            else:
                return weight_choice
        except ValueError:
            return -1


if not are_available_runs(runs):
    print("No previous training runs found. Using default pretrained weights")
    weights_path = move_to_weights_dir("yolov8s-seg.pt")
    model = YOLO(weights_path) # Use small pretrained model
else:
    chosen_run = get_run_choice(runs)
    if chosen_run != -1:
        selected_run = runs[chosen_run]
        print(f"\nSelected training run: {selected_run}\n")

        # List all the saved weights in that specific run
        weights_dir = os.path.join(base_dir, selected_run, "weights")
        saved_weights = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]

        if display_available_runs(saved_weights):
            chosen_weights = get_weight_choice()
            if chosen_weights == -1:
                print("[!] Invalid input, defaulting to 0.")
                model = YOLO(os.path.join(weights_dir, saved_weights[0]))  # Default to best.pt
            else:
                model = YOLO(os.path.join(weights_dir, saved_weights[chosen_weights]))  # Resume from chosen model
        else:
            # No saved weights found, use default pretrained weights
            print("No saved weights found in this run. Using default pretrained weights")
            weights_path = move_to_weights_dir("yolov8s-seg.pt")
            model = YOLO(weights_path) # Use small pretrained model

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
results = model(r"predictions\box.jpg", conf=confident_input) # TODO : Make this dynamic for all images

results[0].show()  # show prediction with boxes/masks

# Print all detections
# print(results[0].boxes) # Bounding boxes
# print(results[0].masks) # Segmentation masks
