# =================================================================
# Visualize a random image and its corresponding segmentation mask
# =================================================================

# Gets default image and mask image from a random file in the test folder
def visualize_random_image_and_mask(image_dir, mask_dir):
    import matplotlib.pyplot as plt
    import cv2
    import os
    import random

    # Grab all image filenames from the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    if not image_files:
        print("[!] No image files found in:", image_dir)
        return

    # Pick a random image
    image_filename = random.choice(image_files)

    # Paths
    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, image_filename)  # assume the same name for mask

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"[!] Missing file: {image_path} or {mask_path}")
        return

    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("[!] Could not load image or mask")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot both
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title(f"Original Image\n{image_filename}")
    axs[1].imshow(mask, cmap='Reds')
    axs[1].set_title("Segmentation Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
visualize_random_image_and_mask(
    image_dir="datasets/images/test",
    mask_dir="datasets/masks/test"
)

