# ===========================================================================================
# This script evaluates a trained YOLO model on best-performing epoch based on mAP@0.5 (mask)
# ===========================================================================================

import os
import pandas as pd

def find_best_yolo_run(base_dir="output/yolo"):
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs:
        print("No YOLO runs found.")
        return

    print("Available YOLO runs:")
    for idx, run in enumerate(runs):
        print(f"{idx}: {run}")

    try:
        run_idx = int(input("Select a run (index): "))
        run_name = runs[run_idx]
        csv_path = os.path.join(base_dir, run_name, "results.csv")
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    if not os.path.exists(csv_path):
        print(f"results.csv not found in {run_name}")
        return

    df = pd.read_csv(csv_path)

    if "metrics/mAP50(M)" not in df.columns:
        print("Column 'metrics/mAP50(M)' not found in CSV.")
        return

    best_epoch = df["metrics/mAP50(M)"].idxmax()
    best_metrics = df.loc[best_epoch]

    print(f"\nBest model in run '{run_name}' at epoch {best_epoch + 1}")
    print(f"mAP@0.5 (mask): {best_metrics['metrics/mAP50(M)']:.4f}")
    print(f"Precision     : {best_metrics['metrics/precision(M)']:.4f}")
    print(f"Recall        : {best_metrics['metrics/recall(M)']:.4f}")

if __name__ == "__main__":
    find_best_yolo_run()
