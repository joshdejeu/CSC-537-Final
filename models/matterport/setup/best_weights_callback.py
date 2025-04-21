# =====================================================
# Hooks into training loop to evaluate the best weights
# =====================================================

import os
import shutil
from keras.callbacks import Callback

# Hide TensorFlow logs (0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import sys

# Surpress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable TensorFlow logger
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

from keras.callbacks import Callback

class BestWeightsManager(Callback):
    def __init__(self, model_dir, run_name):
        super().__init__()
        self.model_dir = model_dir
        self.run_name = run_name
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        epoch_filename = f"mask_rcnn_{self.run_name}_{epoch+1:04d}.h5"
        epoch_path = os.path.join(self.model_dir, epoch_filename)

        if not os.path.exists(epoch_path):
            print(f"[!] Skipped: {epoch_filename} not found")
            return

        # Save as last.h5
        last_path = os.path.join(self.model_dir, "last.h5")
        shutil.copy(epoch_path, last_path)

        # Save as best.h5 if this is the best val_loss so far
        best_path = os.path.join(self.model_dir, "best.h5")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            shutil.copy(epoch_path, best_path)
            print(f"Epoch {epoch+1}: New best.h5 (val_loss={val_loss:.4f})")

        # Delete all other .h5 files except best/last
        for f in os.listdir(self.model_dir):
            if f.endswith(".h5") and f not in ["best.h5", "last.h5"]:
                try:
                    os.remove(os.path.join(self.model_dir, f))
                except Exception as e:
                    print(f"[!] Couldn't delete {f}: {e}")
