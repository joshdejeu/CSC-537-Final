# =======================================================
# Use trained MRCNN weights to make predictions on images
# =======================================================

import os
from load_weights import load_weights
from display_instances_custom import display_instances_custom

# Base dir for Matterport runs
base_dir = "output\mrcnn"
output_dir = "output\predictions"

# List of all directories in the base dir
matterport_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Check if there are any Matterport runs
# Return true if runs exist, else false (exits for now)
def display_available_runs(runs):
    # Check if there are any runs available
    if runs:
        print("Available training runs:")
        for idx, run in enumerate(runs):
            print(f"{idx}: {run}")
        return True
    else:
        print("No previous training runs found. Using default pretrained weights")
        # TODO : Get the default pretrained weights for Matterport
        exit(1)
        return False

# Choose a run from the list of available runs
# Return the chosen run dir if valid, else return the best run dir
# TODO : Implement a function to choose the best run based on some metric (in train.py / resume.py)
def choose_run(runs):
    chosen_input = input(f"\nChoose a training run to get weights from (0-{len(runs) - 1}): ")
    try:
        chosen_run = int(chosen_input)
        if chosen_run >= len(runs):
            raise ValueError("Out of range, invalid input")
        else:
            print(f"\Chosen run weights: {runs[chosen_run]}\n")
            run_dir = os.path.join(base_dir, runs[chosen_run])
            return run_dir
    except ValueError:
        print("[!] Invalid input.")
        exit(1)

# Display all saved weights in the chosen run
# Return list of available weights if weights exist, else exit
def display_available_weights(chosen_run):
    # List all the saved weights in that specific run
    saved_weights = [f for f in os.listdir(chosen_run) if f.endswith(".h5")]

    if not saved_weights:
        print("No saved model weights found.")
        exit(1)
    else:
        print("Saved model weights found:")
        for idx, weight_file in enumerate(saved_weights):
            print(f"{idx+1}: {weight_file}")
        return saved_weights

# Choose a weight from the list of available weights in the chosen run
# Return the chosen weight dir if valid, else return the best weight dir
# TODO : Implement a function to choose the best weight based on some metric (in train.py / resume.py)
def choose_weight(weights, run_dir):
    chosen_input = input(f"\nChoose a weight file (1-{len(weights)}): ")
    try:
        chosen_weight = int(chosen_input)
        if chosen_weight > len(weights) or chosen_weight < 1:
            raise ValueError("Out of range, invalid input")
        else:
            chosen_weight_file = weights[chosen_weight - 1] # Adjust for 0-based index
            print(f"Chosen run weights: {chosen_weight_file}\n")
            weight_dir = os.path.join(run_dir, chosen_weight_file)
            return weight_dir
    except ValueError:
        # TODO : Return the best weight
        # NOTE : For now, return the last weight in the list
        default_weight = weights[-1] # Choose the last weight
        print(f"[!] Invalid input. Choosing the last weight: {default_weight}")
        weight_dir = os.path.join(run_dir, default_weight) # Choose the last weight
        return weight_dir

# Run prediction on all images in the predictions folder
# Output result images to output folder
def predict_images(model, chosen_weight_dir):
    import cv2
    from mrcnn.visualize import display_instances

    images_dir = "input"

    # Get list of all images in the predictions folder
    images_to_predict = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]

    # Run detection on all images
    print(f"\nRunning predictions on {len(images_to_predict)} images...")

    # Get run folder and weight filename (no extension)
    run_dir = os.path.dirname(chosen_weight_dir)          # output\mrcnn\test_20250420T0302
    run_name = os.path.basename(run_dir)            # test_20250420T0302
    weight_file = os.path.basename(chosen_weight_dir)     # mask_rcnn_test__0250.h5
    weight_name = os.path.splitext(weight_file)[0]  # mask_rcnn_test__0250

    # Create prediction output path
    prediction_path = os.path.join("output", "predictions", "mrcnn", run_name, weight_name)
    os.makedirs(prediction_path, exist_ok=True)  # Create if it doesn't exist

    # Loop through each image one at a time and save output
    # TODO : use tqdm progress bar for better visualization
    for i, filename in enumerate(images_to_predict):
        print(f"[{i+1}/{len(images_to_predict)}] Processing {filename}")

        # Load and preprocess image
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection (must pass a list, even for 1 image)
        results = model.detect([image], verbose=0)
        r = results[0]

        # Visualize result
        result_img = display_instances_custom(
            image, r['rois'], r['masks'], r['class_ids'],
            ['BG', 'Rubbish'], r['scores'], show_mask=True, show_bbox=True
        )

        # Convert color and save output
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        output_filename = "predicted_" + filename
        cv2.imwrite(os.path.join(prediction_path, output_filename), result_img)


# If available runs exist, display them
if display_available_runs(matterport_runs):
    # Choose a run from the list of available runs
    chosen_run_dir = choose_run(matterport_runs)

    # List all the saved weights in that specific run
    list_of_available_weights = display_available_weights(chosen_run_dir)

    if list_of_available_weights:
        # Choose a weight from the list of available weights in the chosen run
        chosen_weight_dir = choose_weight(list_of_available_weights, chosen_run_dir)
        
        # Load a model with chosen weights from the chosen run
        model = load_weights(chosen_run_dir, chosen_weight_dir)

        # Run prediction on all images in the predictions folder
        predict_images(model, chosen_weight_dir)


