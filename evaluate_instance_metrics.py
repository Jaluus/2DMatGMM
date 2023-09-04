import json
import os

from Datasets import register_all_datasets
from GMMDetector import MaterialDetector

# import the coco evaluator
from Utils.custom_COCO_evaluator import DetectionWrapper, evaluate_on_dataset

register_all_datasets()

# Constants
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRAST_PATH_ROOT = os.path.join(FILE_DIR, "GMMDetector", "trained_parameters")
METRIC_PATH = os.path.join(FILE_DIR, "Metrics")
os.makedirs(METRIC_PATH, exist_ok=True)

MATERIALS = [
    "WSe2",
    "Graphene",
]

for material in MATERIALS:
    dataset_name = f"{material}_test"
    metrics_name = f"{material}_COCO.json"

    contrast_path = os.path.join(CONTRAST_PATH_ROOT, f"{material}_GMM.json")
    if not os.path.exists(contrast_path):
        print(
            f"Contrast parameters for {material} not found in {contrast_path}. Skipping {material} Evaluation."
        )
        continue

    # Read the contrast parameters
    with open(contrast_path) as f:
        contrast_dict = json.load(f)

    # initialize the detector and wrap it to be able to be used by the COCO evaluator
    myDetector = MaterialDetector(
        contrast_dict=contrast_dict,
        standard_deviation_threshold=5,
        size_threshold=200,
        used_channels="BGR",
    )
    model = DetectionWrapper(myDetector)

    # evaluate the model on the dataset
    results = evaluate_on_dataset(
        model,
        dataset_name,
        output_dir=METRIC_PATH,
    )

    with open(os.path.join(METRIC_PATH, metrics_name), "w") as f:
        json.dump(results, f, indent=4)
