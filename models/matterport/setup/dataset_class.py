import os
import numpy as np
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
import cv2

class GarbageDataset(Dataset):
    # Load a subset of the garbage dataset.
    def load_garbage(self, dataset_dir, subset):
        # Add class (background is always class 0)
        self.add_class("garbage", 1, "garbage")

        # Define dataset paths
        image_dir = os.path.join(dataset_dir, "images", subset)
        mask_dir = os.path.join(dataset_dir, "masks", subset)

        # List all image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        for image_id, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)

            # Add image
            self.add_image(
                "garbage",
                image_id=image_id,
                path=image_path,
                mask_path=mask_path
            )

    # Load instance masks for the given image.
    def load_mask(self, image_id):
        # Get image info
        info = self.image_info[image_id]
        mask_path = info['mask_path']

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert mask to binary (1 for object, 0 for background)
        mask = mask.astype(np.uint8)  # Prevents warnings

        # Expand dimensions to match (height, width, instances)
        mask = np.expand_dims(mask, axis=-1)

        # Return mask and class IDs (all masks belong to class 1)
        class_ids = np.array([1], dtype=np.int32)
        return mask, class_ids


    # Return the path of the image.
    def image_reference(self, image_id):
        return self.image_info[image_id]["path"]