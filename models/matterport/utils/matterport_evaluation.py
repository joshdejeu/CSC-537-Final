import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATTERPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(MATTERPORT_DIR)

from mrcnn import utils, model as modellib
from setup.config import GarbageDetectionConfig
from setup.dataset_class import GarbageDataset

def choose_run_and_weight(base_dir="output/mrcnn"):
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs:
        print("No training runs found.")
        exit(1)

    print("Available training runs:")
    for idx, run in enumerate(runs):
        print(f"{idx}: {run}")

    try:
        run_idx = int(input("Select a run (index): "))
        run_dir = os.path.join(base_dir, runs[run_idx])
    except:
        print("Invalid run index.")
        exit(1)

    weights = [f for f in os.listdir(run_dir) if f.endswith(".h5")]
    if not weights:
        print("No weight files found in selected run.")
        exit(1)

    print("Available weight files:")
    for idx, w in enumerate(weights):
        print(f"{idx}: {w}")

    try:
        weight_idx = int(input("Select a weight file (index): "))
        weight_path = os.path.join(run_dir, weights[weight_idx])
    except:
        print("Invalid weight index.")
        exit(1)

    return weight_path

def evaluate_model(weight_path, dataset_dir="datasets", subset="val"):
    class InferenceConfig(GarbageDetectionConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    dataset = GarbageDataset()
    dataset.load_garbage(dataset_dir, subset)
    dataset.prepare()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(weight_path))
    model.load_weights(weight_path, by_name=True)

    print(f"\nEvaluating {os.path.basename(weight_path)} on {subset} set ({len(dataset.image_ids)} images)...")
    APs = []
    TP = 0
    FP = 0
    FN = 0

    for image_id in tqdm(dataset.image_ids, desc="Evaluating"):
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False
        )
        results = model.detect([image], verbose=0)
        r = results[0]

        AP, _, _, _ = utils.compute_ap(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r["class_ids"], r["scores"], r["masks"]
        )
        APs.append(AP)

        gt_match, pred_match, _ = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r["class_ids"], r["scores"], r["masks"],
            iou_threshold=0.5
        )
        TP += np.sum(pred_match > -1)
        FP += np.sum(pred_match == -1)
        FN += np.sum(gt_match == -1)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_ap = np.mean(APs)

    print(f"\nEvaluation Results:")
    print(f"  True Positives : {TP}")
    print(f"  False Positives: {FP}")
    print(f"  False Negatives: {FN}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"mAP@0.5  : {mean_ap:.4f}")

if __name__ == "__main__":
    weight_path = choose_run_and_weight()
    evaluate_model(weight_path)
