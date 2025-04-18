import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
from mrcnn import model as modellib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)

from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset

# Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO weights if not present
if not os.path.exists(COCO_MODEL_PATH):
    from mrcnn.utils import download_trained_weights
    download_trained_weights(COCO_MODEL_PATH)

# Load configuration
config = GarbageDetectionConfig()

# Create model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

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

# Train
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')