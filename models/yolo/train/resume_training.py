# =====================
# Use this file to resume training your previous models
# (they must have previously trained for atleast 1 epoch)
# =====================

import os
import shutil
from ultralytics import YOLO

ROOT_DIR = os.getcwd()
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models", "yolo", "pt_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Base directory for all training runs
base_dir = "output/yolo/"

# Moves downloaded pretrained weights to a permanent directory
def move_to_weights_dir(filename):
    src = os.path.join(ROOT_DIR, filename)
    dst = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    return dst

def main():
    # List all subdirectories in the base directory (each representing a different training run)
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Display the available training runs
    if not runs:
        print("No previous training runs found. Starting fresh...")
        weights_path = move_to_weights_dir("yolov8s-seg.pt")
        model = YOLO(weights_path) # Use small pretrained model
    else:
        print("Available training runs:")
        for idx, run in enumerate(runs):
            print(f"{idx}: {run}")

        # Choose which training run to continue
        run_choice = int(input(f"Choose a training run to resume from (0-{len(runs) - 1}): "))

        # Ensure valid run choice
        if run_choice < 0 or run_choice >= len(runs):
            print("Invalid choice, starting fresh...")
            weights_path = move_to_weights_dir("yolov8s-seg.pt")
            model = YOLO(weights_path) # Use small pretrained model
        else:
            # Get the selected run's directory
            selected_run = runs[run_choice]
            print(f"Selected training run: {selected_run}")

            # List all the saved weights in that specific run
            weights_dir = os.path.join(base_dir, selected_run, "weights")
            saved_weights = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]

            if not saved_weights:
                print("No saved weights found in this run. Starting fresh...")
                weights_path = move_to_weights_dir("yolov8s-seg.pt")
                model = YOLO(weights_path) # Use small pretrained model
            else:
                # Display saved weights and ask user to choose which checkpoint to resume from
                print("Saved model weights found:")
                for idx, weight_file in enumerate(saved_weights):
                    print(f"{idx}: {weight_file}")

                # Ask user to choose the specific weight to resume from
                weight_choice = int(input(f"Choose a model to resume from (0-{len(saved_weights) - 1}): "))

                # Ensure valid weight choice
                if weight_choice < 0 or weight_choice >= len(saved_weights):
                    print("Invalid choice, resuming from the most recent model...")
                    model = YOLO(os.path.join(weights_dir, saved_weights[-1]))  # Default to most recent model
                else:
                    model = YOLO(os.path.join(weights_dir, saved_weights[weight_choice]))  # Resume from chosen model

    # Train the model (if needed, this will pick up from the selected model and continue training)
    model.train(
        data="datasets/mju.yaml",
        project="output/yolo", # Output directory
        epochs=50,
        imgsz=640,  # Adjust if needed
        # TODO : get the correct folder name and continue training from there
        name="RESUMED",  # Save to the same folder (or change if you want a new name)
        resume=True  # Automatically resume from the selected checkpoint
    )

if __name__ == "__main__":
    main()