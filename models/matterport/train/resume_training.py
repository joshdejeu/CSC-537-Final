# ======================================
# Resumes training from MRCNN checkpoint
# ======================================

import os

# Hide TensorFlow logs (0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Disable TensorFlow logger
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys
import gc
from keras import backend as K

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # adjust if needed
sys.path.append(PROJECT_ROOT)

from mrcnn import model as modellib
from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset
from setup.best_weights_callback import BestWeightsManager
from keras.callbacks import ModelCheckpoint

# Config and directories
config = GarbageDetectionConfig()
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "output\mrcnn")
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
    print("\nSaved model weights found:")
    for idx, weight_file in enumerate(weights):
        print(f"{idx+1}: {weight_file}")

    # Ask user to choose the specific weight to resume from
    weight_choice = int(input(f"\nChoose a model to resume from (1-{len(weights)}): "))
    weight_choice -= 1  # Adjust for zero-based index
    weight_path = os.path.join(run_path, weights[weight_choice])

    # Ensure valid weight choice
    if weight_choice < 0 or weight_choice >= len(weights):
        print("Invalid choice")
        exit(1)
    else:
        config.NAME = selected_run.split("_")[0] + "_" + selected_run.split("_")[1]  # Use full folder name including timestamp

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR) # Resume from chosen model
        model.load_weights(weight_path, by_name=True)

        # Override model directory (this prevents new folder creation)
        model.set_log_dir(weight_path)  # Resets log_dir & checkpoint naming
        model.log_dir = run_path  # The original run dir (Example: test_20250420T2315)
        model.checkpoint_path = os.path.join(run_path, f"mask_rcnn_{config.NAME.lower()}_" + "{epoch:04d}.h5")

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
        all_epochs = int(input("Epochs for training all layers [default: 50]: ") or 50)
    except ValueError:
        print("[!] Invalid input. Using default (50 epochs).")
        head_epochs = 50
        all_epochs = 50


# Manually set the run folder to continue in same directory
checkpoint_cb = ModelCheckpoint(
    filepath=model.checkpoint_path,
    save_weights_only=True,
    verbose=0,
    period=1  # Saves every epoch
)

best_weights_cb = BestWeightsManager(model_dir=run_path, run_name=config.NAME)

callbacks = [checkpoint_cb, best_weights_cb]

# Run training based on mode
if mode == 1:
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=all_epochs,
        layers="all",
        custom_callbacks=callbacks)
else:
    # Resume training heads first
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=head_epochs,
        layers="heads",
        custom_callbacks=callbacks)

    # Clear session and cleanup to free up memory and prevent OOM errors
    # NOTE : This removes the model from memory, so we need to recreate it
    K.clear_session()
    gc.collect()

    # Recreate model + reload weights
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    model.load_weights(os.path.join(run_path, "last.h5"), by_name=True)

    # Reapply forced paths
    model.log_dir = run_path
    model.checkpoint_path = os.path.join(run_path, f"mask_rcnn_{config.NAME.lower()}_" + "{epoch:04d}.h5")

    # Then train all layers
    # layers=4+ just train from stage 4 onward to prevent OOM errors
    # NOTE : This is a workaround for OOM errors, not the best practice
    # If you want to train all layers, set layers="all" and increase batch size
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=all_epochs, # Default value, can be changed
        layers="4+",
        custom_callbacks=callbacks)