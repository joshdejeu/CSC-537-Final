from mrcnn.config import Config

class GarbageDetectionConfig(Config):
    NAME = "waste"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + garbage
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9