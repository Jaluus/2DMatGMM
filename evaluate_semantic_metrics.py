import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from GMMDetector import MaterialDetector
from Utils import ConfusionMatrix

# Constants
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRAST_PATH_ROOT = os.path.join(FILE_DIR, "GMMDetector", "trained_parameters")
METRIC_PATH = os.path.join(FILE_DIR, "Metrics")
DATASET_ROOT = os.path.join(FILE_DIR, "Datasets", "GMMDetectorDatasets")

FP_RANGE = np.linspace(0, 1, 11)

MATERIALS = ["WSe2", "Graphene"]
NUM_CLASSES = {"WSe2": 3, "Graphene": 4}

os.makedirs(METRIC_PATH, exist_ok=True)

for material in MATERIALS:
    contrast_path = os.path.join(CONTRAST_PATH_ROOT, f"{material}_GMM.json")
    image_dir = os.path.join(DATASET_ROOT, material, "test_images")
    mask_dir = os.path.join(DATASET_ROOT, material, "test_semantic_masks")

    if not os.path.exists(contrast_path):
        print(
            f"Contrast parameters for {material} not found in {contrast_path}. Skipping {material} Evaluation."
        )
        continue

    # Read the contrast parameters
    with open(contrast_path) as f:
        contrast_dict = json.load(f)

    myDetector = MaterialDetector(
        contrast_dict=contrast_dict,
        standard_deviation_threshold=5,
        size_threshold=200,
        used_channels="BGR",
    )
    # set up the confusion matrices
    confusion_matrices = {
        fp: ConfusionMatrix(
            num_classes=NUM_CLASSES[material] + 1,
            ignore_label=NUM_CLASSES[material] + 2,
        )
        for fp in FP_RANGE
    }

    start_time = time.time()

    image_names = [
        image_name
        for image_name in os.listdir(image_dir)
        if image_name.endswith(".jpg")
    ]
    for idx, image_name in enumerate(image_names):
        time_per_image = (time.time() - start_time) / (idx + 1)
        approx_time_left = (len(image_names) - idx) * time_per_image
        approx_time_left = time.strftime("%H:%M:%S", time.gmtime(approx_time_left))

        image = cv2.imread(os.path.join(image_dir, image_name))
        true_mask = cv2.imread(
            os.path.join(mask_dir, image_name.replace(".jpg", ".png")),
            cv2.IMREAD_GRAYSCALE,
        )

        # ~120ms
        detected_flakes = myDetector.detect_flakes(image)

        # generate the semantic mask
        detected_masks = {
            fp: np.zeros_like(true_mask, dtype=np.uint8) for fp in FP_RANGE
        }

        for flake in detected_flakes:
            # sweep through the false positive range and add the flake to the mask if it is within fp range
            for sweep_val in FP_RANGE:
                if flake.false_positive_probability > sweep_val:
                    continue
                else:
                    detected_masks[sweep_val][flake.mask != 0] = int(flake.thickness)

        for sweep_val in FP_RANGE:
            confusion_matrices[sweep_val].add(
                detected_masks[sweep_val].flatten(), true_mask.flatten()
            )

        printed_string = f"{image_name} || {idx:5}/{len(image_names):5} ({idx / len(image_names):6.1%}) | {approx_time_left}"
        print(printed_string, end="\t\r")

    precisions = {sweep_val: [] for sweep_val in FP_RANGE}
    accuracies = {sweep_val: [] for sweep_val in FP_RANGE}
    recalls = {sweep_val: [] for sweep_val in FP_RANGE}
    IOUs = {sweep_val: [] for sweep_val in FP_RANGE}

    for sweep_val, conf_mat in confusion_matrices.items():
        cm = conf_mat.value()

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        accuracy = (TP + TN) / (TP + FP + FN + TN + 0.0001)
        precision = TP / (TP + FP + 0.0001)
        recall = TP / (TP + FN + 0.0001)
        IOU = TP / (TP + FP + FN + 0.0001)

        precisions[sweep_val] = precision
        accuracies[sweep_val] = accuracy
        recalls[sweep_val] = recall
        IOUs[sweep_val] = IOU

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for idx in range(NUM_CLASSES[material]):
        prec = np.array(list(precisions.values()))[:, idx + 1]
        rec = np.array(list(recalls.values()))[:, idx + 1]
        iou = np.array(list(IOUs.values()))[:, idx + 1]

        prec = prec[1:]
        rec = rec[1:]
        iou = iou[1:]

        x_idx = idx // 2
        y_idx = idx % 2

        axs[x_idx, y_idx].plot(FP_RANGE[1:], prec, label="Precision")
        axs[x_idx, y_idx].plot(FP_RANGE[1:], rec, label="Recall")
        axs[x_idx, y_idx].plot(FP_RANGE[1:], iou, label="IOU")

        # axs[x_idx,y_idx].grid()
        axs[x_idx, y_idx].set_xlabel("False Positive Treshold")
        axs[x_idx, y_idx].set_ylabel("Score")
        axs[x_idx, y_idx].legend()
        axs[x_idx, y_idx].set_title(f"Layer {idx+1}")

        axs[x_idx, y_idx].set_ylim(-0.05, 1.05)
        axs[x_idx, y_idx].set_xlim(-0.05, 1.05)

    plt.savefig(f"Metrics/{material}_semantic_metrics.png", dpi=300)
