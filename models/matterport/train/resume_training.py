# ======================================
# Resumes training from MRCNN checkpoint
# ======================================

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import sys

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # adjust if needed
sys.path.append(PROJECT_ROOT)


from mrcnn import model as modellib
from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset


# Config and directories
config = GarbageDetectionConfig()
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "output/mrcnn")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

# List all subdirectories in the base directory (each representing a different training run)
runs = [
    d for d in os.listdir(MODEL_DIR)
    if os.path.isdir(os.path.join(MODEL_DIR, d)) and
       not d.startswith(".")  # Skip .gitkeep
]

# Display the available training runs
if not runs:
    print("No previous training runs found.")
    exit(1)

print("\nAvailable training runs:")
for idx, run in enumerate(runs):
    print(f"{idx}: {run}")
# Choose which training run to continue
run_choice = int(input(f"\nChoose a training run to resume from (0-{len(runs) - 1}): "))

# Ensure valid run choice
if run_choice < 0 or run_choice >= len(runs):
    print("Invalid choice")
    exit(1)
else:
    # Get the selected run's directory
    selected_run = runs[run_choice]
    run_path = os.path.join(MODEL_DIR, selected_run)
    print(f"Selected training run: {selected_run}")

    # TODO : Make this dynamic, find the best run (lowest loss) and resume from that
    # List all the saved weights in that specific run
    weights = sorted([f for f in os.listdir(run_path) if f.endswith(".h5")]) # List all .h5 files
    if not weights:
        print("No saved weights found in this run.")
        exit(1)

    # Display saved weights and ask user to choose which checkpoint to resume from
    print("Saved model weights found:")
    for idx, weight_file in enumerate(weights):
        print(f"{idx+1}: {weight_file}")

    # Ask user to choose the specific weight to resume from
    weight_choice = int(input(f"Choose a model to resume from (1-{len(weights)}): "))
    weight_choice -= 1  # Adjust for zero-based index
    weight_path = os.path.join(run_path, weights[weight_choice])

    # Ensure valid weight choice
    if weight_choice < 0 or weight_choice >= len(weights):
        print("Invalid choice")
        exit(1)
    else:
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR) # Resume from chosen model
        model.load_weights(weight_path, by_name=True)

# Prepare datasets
dataset_train = GarbageDataset()
dataset_train.load_garbage(DATASET_DIR, "train")
dataset_train.prepare()

dataset_val = GarbageDataset()
dataset_val.load_garbage(DATASET_DIR, "val")
dataset_val.prepare()

print("")
# Choose to train heads or all layers with dynamic epochs
train_choice = input("Train heads or all layers? (heads/all): ").strip().lower()
if train_choice == "all":
    mode = 1
    try:
        all_epochs = int(input("Epochs for training all layers [default: 50]: ") or 50)
    except ValueError:
        print("[!] Invalid input. Using default (50 epochs).")
        all_epochs = 50
else:
    mode = 0  # Default or "heads" or anything else
    try:
        head_epochs = int(input("Epochs for training heads [default: 50]: ") or 30)
    except ValueError:
        print("[!] Invalid input. Using default (50 epochs).")
        head_epochs = 50

# Run training based on mode
if mode == 1:
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=all_epochs,
        layers="all")
else:
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=head_epochs,
        layers="heads")