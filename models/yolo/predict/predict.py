# =================================================
# Use trained weights to make predictions on images
# =================================================

from ultralytics import YOLO

# TODO : display all runs and weights
# TODO : create a place to store new test images

model = YOLO(r"output\yolo\baseline_mju\weights\best.pt")
results = model("test.png", conf=0.1)

results[0].show()  # show prediction with boxes/masks

# Print all detections
print(results[0].boxes) # Bounding boxes
print(results[0].masks) # Segmentation masks
