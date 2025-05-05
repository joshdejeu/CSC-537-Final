import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Root working directory
root_dir = os.getcwd()

# Root working directory
root_dir = os.getcwd()

# Paths to annotation and image directories
ann_dir = os.path.join(root_dir, "tmp", "detectron2_annotations")
img_dir = os.path.join(root_dir, "datasets", "images")

# Register datasets
register_coco_instances("mju_train", {}, os.path.join(ann_dir, "train.json"), os.path.join(img_dir, "train"))
register_coco_instances("mju_val", {}, os.path.join(ann_dir, "val.json"), os.path.join(img_dir, "val"))

# Config setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("mju_train",)
cfg.DATASETS.TEST = ("mju_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.OUTPUT_DIR = os.path.join(root_dir, "output", "weights","detectron2")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Match Matterport memory/speed-friendly settings
cfg.SOLVER.IMS_PER_BATCH = 1  # IMAGES_PER_GPU = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []  # Disable LR step decay

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # TRAIN_ROIS_PER_IMAGE
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # garbage only

# Smaller input size like 512x512
cfg.INPUT.MIN_SIZE_TRAIN = (512,)
cfg.INPUT.MAX_SIZE_TRAIN = 512
cfg.INPUT.MIN_SIZE_TEST = 512
cfg.INPUT.MAX_SIZE_TEST = 512

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # DETECTION_MIN_CONFIDENCE

# Reduce instance count to save VRAM
cfg.TEST.DETECTIONS_PER_IMAGE = 50  # DETECTION_MAX_INSTANCES


# Start training
if __name__ == "__main__":
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
