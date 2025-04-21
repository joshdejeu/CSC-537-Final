# =======================================================
# Trains Matterport model and saves best and last weights
# Initially trains heads, then all layers
# =======================================================

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
from mrcnn import model as modellib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)

from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset
from setup.best_weights_callback import BestWeightsManager
from keras.callbacks import ModelCheckpoint

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "output\mrcnn")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
PRETRAINED_DIR = os.path.join(ROOT_DIR, "models","matterport", "pt_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)
COCO_MODEL_PATH = os.path.join(PRETRAINED_DIR, "mask_rcnn_coco.h5")

# Download COCO weights if not present
if not os.path.exists(COCO_MODEL_PATH):
    from mrcnn.utils import download_trained_weights
    download_trained_weights(COCO_MODEL_PATH)

# Load configuration
config = GarbageDetectionConfig()

print("")

# Get folder name for the run
run_name = input("Name for run: ").strip()
if not run_name.endswith("_"):
    run_name += "_"
config.NAME = run_name

# Create the model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR) # Create model in training mode, makes a new dir with timestamp
run_dir = model.log_dir  # Example: output/mrcnn/test_20250420T0302

print(run_dir)

# exit(1)

# Load weights
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Prepare datasets
dataset_train = GarbageDataset()
dataset_train.load_garbage(DATASET_DIR, "train")
dataset_train.prepare()

dataset_val = GarbageDataset()
dataset_val.load_garbage(DATASET_DIR, "val")
dataset_val.prepare()

print("")

# Dynamically choose epochs for training heads and all layers
try:
    head_epochs = int(input("Epochs for training heads [default 50]: ") or 50)
    all_epochs = int(input("Epochs for training all layers [default 50]: ") or 50)
except ValueError:
    print("[!] Invalid input. Using defaults (50 for heads, 50 for all).")
    head_epochs = 50
    all_epochs = 50

checkpoint_path = os.path.join(run_dir, f"mask_rcnn_{config.NAME}" + "{epoch:04d}.h5") # Example: mask_rcnn_name_0001.h5
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=0,
    period=1  # Saves every epoch
)

best_weights_cb = BestWeightsManager(model_dir=run_dir, run_name=config.NAME)

callbacks = [checkpoint_cb, best_weights_cb]

# Train heads first
model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=head_epochs,
    layers='heads',
    custom_callbacks=callbacks) # Only train on top layers of network

# NOTE : To train all layers, uncomment the following lines, run resume_training.py, or comment the above line and uncomment the below one

# TODO : Make this work like resume_training.py
# Fine-tune entire model
# model.train(dataset_train, dataset_val,
#     learning_rate=config.LEARNING_RATE / 10,
#     epochs=all_epochs,
#     layers="all",
#     custom_callbacks=callbacks  ) # Trains entire model including ResNet + all heads
