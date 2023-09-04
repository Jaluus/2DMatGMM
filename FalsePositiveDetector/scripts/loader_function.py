import json
import os

import cv2
import numpy as np


def load_and_partition_data(data_path):
    mask_paths = {}
    annotations = {}

    # Loading the Data
    solidities = {"train": [], "test": []}
    arcareas = {"train": [], "test": []}
    labels = {"train": [], "test": []}

    # fmt: off
    for split in ["train", "test"]:
        print(f"Processing {split} split")
        annotations[split] = json.load(
            open(os.path.join(data_path, split, "annotations.json"), "r")
        )
        mask_paths[split] = [
            os.path.join(data_path, split, "masks", image_name + ".png")
            for image_name in annotations[split].keys()
        ]
        labels[split] = [
            annotations[split][image_name] for image_name in annotations[split].keys()
        ]
        for idx, mask_path in enumerate(mask_paths[split]):
            print(f"{idx}/{len(mask_paths[split])}", end="\r")

            mask = cv2.imread(mask_path, 0)

            contours, _ = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]

            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            arclength = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            solidity = float(area) / convex_hull_area

            solidities[split].append(solidity)
            arcareas[split].append(arclength / area**0.5)
    # fmt: on

    X_train = np.array([arcareas["train"], solidities["train"]]).T
    y_train = np.array(labels["train"])

    X_test = np.array([arcareas["test"], solidities["test"]]).T
    y_test = np.array(labels["test"])

    return X_train, y_train, X_test, y_test
