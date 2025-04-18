from mrcnn.config import Config

class GarbageDetectionConfig(Config):
    NAME = "run_"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + garbage
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

    # Optional for smaller GPUs
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512