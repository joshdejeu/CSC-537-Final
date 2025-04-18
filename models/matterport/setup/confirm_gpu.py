# =================
# Confirm GPU Setup
# =================

# Should ouput:
# =====================
# TF version: 1.13.1
# Built with CUDA: True
# GPU available: True
# =====================


import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())

