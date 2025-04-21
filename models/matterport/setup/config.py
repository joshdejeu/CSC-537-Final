from mrcnn.config import Config

# TODO: Make these options dynamic for every training run
class GarbageDetectionConfig(Config):
    NAME = "run_"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + garbage
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8

    # Optional for smaller GPUs
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Reduces memory usage
    TRAIN_ROIS_PER_IMAGE = 64       # Default is 200
    MAX_GT_INSTANCES = 50            # Default is 100 Number of objects per image
    DETECTION_MAX_INSTANCES = 50     # Default is 100
    POST_NMS_ROIS_TRAINING = 500     # Default is 2000
    POST_NMS_ROIS_INFERENCE = 250    # Default is 1000

    BACKBONE = "resnet50"  # ResNet101 default
    # MASK_SHAPE = [14, 14]  # default is [28, 28]
