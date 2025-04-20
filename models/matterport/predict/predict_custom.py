# =======================================================
# Use trained MRCNN weights to make predictions on images
# =======================================================

# TODO : Display all runs to select from
# TODO : Find the best weight from that run

# NOTE : This is a bit of a hack, might have some problems and lots of warnings in the console

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)

from mrcnn import model as modellib
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from setup.dataset_class import GarbageDataset
from setup.config import GarbageDetectionConfig 

# Setup paths
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, "output", "mrcnn", "real_20250420T0302", "real20250420T0302", "mask_rcnn_run__0050.h5")
IMAGE_PATH = os.path.join(ROOT_DIR, "predictions", "juicebox.png")


# Load config and model
config = GarbageDetectionConfig()
config.GPU_COUNT = 1
config.IMAGES_PER_GPU = 1

model = modellib.MaskRCNN(mode="inference", model_dir=ROOT_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

# Load image
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = model.detect([image], verbose=1)
r = results[0]

# Visualize 
display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'Rubbish'], r['scores'])
