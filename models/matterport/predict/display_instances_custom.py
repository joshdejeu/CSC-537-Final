# ===========================================================================
# Custom version of display_instances that overlays masks, bounding boxes, 
# class labels, and confidence scores directly onto an image using OpenCV. 
# Returns the result as a NumPy array instead of displaying with matplotlib.
# ===========================================================================

import numpy as np
import cv2
import random

def display_instances_custom(image, boxes, masks, class_ids, class_names, scores=None, show_mask=True, show_bbox=True):
    """Draw bounding boxes and masks on the image, and return the result as a NumPy array."""
    n_instances = boxes.shape[0]
    if not n_instances:
        return image

    colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(n_instances)]
    img = image.copy()

    for i in range(n_instances):
        color = colors[i]
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if show_mask:
            mask = masks[:, :, i]
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask, color[c], 0)
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

        label = class_names[class_ids[i]]
        score = f"{scores[i]:.2f}" if scores is not None else ""
        caption = f"{label} {score}"
        cv2.putText(img, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img
