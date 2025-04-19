# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
from mrcnn import model as modellib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)

from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "output/mrcnn")
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

# Set folder name
run_name = input("Name for run: ").strip()
# Create model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
if not run_name.endswith("_"):
    run_name += "_"
config.NAME = run_name

# Load weights
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

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

# Train heads first
model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=head_epochs,
    layers='heads') # Only train on top layers of network

# Fine-tune entire model
model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE / 10,
    epochs=all_epochs,
    layers="all") # Trains entire model including ResNet + all heads
