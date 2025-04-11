# ============================
# Trains the baseline model (to compare later)
# ============================

from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")

# Train the model
# Note: To save last.pt it must run for atleast 1 epoch.
model.train(
    data="datasets/mju.yaml",
    epochs=50,
    imgsz=640, # Reduce to 416 or 320 if training too slow
    name="baseline_mju"
)


