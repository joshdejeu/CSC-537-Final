# ====================================================================
# Loads weights from previous training run, configues prediction model
# ====================================================================

import os
# Hide TensorFlow logs (0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

# Surpress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable TensorFlow logger
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)


from mrcnn import model as modellib
from setup.dataset_class import GarbageDataset
from setup.config import GarbageDetectionConfig 

# Creates a model with weights from a previous training run
# Returns the model object
def load_weights(model_dir, weights_dir):
    print(f"\nLoading weights from: {weights_dir}")
    config = GarbageDetectionConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(weights_dir, by_name=True)
    return model