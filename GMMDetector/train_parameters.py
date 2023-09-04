import json
import os

import numpy as np
from scripts.fitting_functions import fit_set
from scripts.plotting_functions import plot_gaussians
from scripts.postprocessing_functions import format_components
from scripts.preprocessor_functions import get_contrasts_from_dir

FILE_DIR = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(FILE_DIR, "..", "Datasets", "GMMDetectorDatasets")
PARAM_ROOT = os.path.join(FILE_DIR, "trained_parameters")
PLOTS_ROOT = os.path.join(FILE_DIR, "plots")

MATERIALS = ["Graphene", "WSe2"]

# define some plotting constants
AXIS_NAMES = ["Blue Contrast", "Green Contrast", "Red Contrast"]
UPPER_BOUNDS = [0, 0, 0]  # B,G,R
LOWER_BOUNDS = [-1, -1, -1]  # B,G,R

# defining material specific constants
PARAMS = {
    "Graphene": {
        "Number of Components": 4,
        "Number of Noise Components": 1,
    },
    "WSe2": {
        "Number of Components": 3,
        "Number of Noise Components": 1,
    },
}

for material in MATERIALS:
    np.random.seed(42)
    data_directory = os.path.join(DATASET_ROOT, material)
    if not os.path.exists(data_directory):
        print(f"Data directory for {material} does not exist. Skipping...")
        continue

    image_directory = os.path.join(data_directory, "train_images")
    mask_directory = os.path.join(data_directory, "train_semantic_masks")

    datapoints_contrast = get_contrasts_from_dir(
        image_directory=image_directory,
        mask_directory=mask_directory,
    )

    # cropping the data so they lie in a certain range
    datapoints_contrast_cropped = datapoints_contrast[
        (datapoints_contrast[:, 0] > LOWER_BOUNDS[0])
        & (datapoints_contrast[:, 0] < UPPER_BOUNDS[0])
        & (datapoints_contrast[:, 1] > LOWER_BOUNDS[1])
        & (datapoints_contrast[:, 1] < UPPER_BOUNDS[1])
        & (datapoints_contrast[:, 2] > LOWER_BOUNDS[2])
        & (datapoints_contrast[:, 2] < UPPER_BOUNDS[2])
    ]

    (
        all_means_gauss,
        all_covariances_gauss,
        all_weights_gauss,
        sampled_data,
        predicted_labels,
    ) = fit_set(
        data=datapoints_contrast_cropped,
        num_components=PARAMS[material]["Number of Components"],
        num_additional_noise_comp=PARAMS[material]["Number of Noise Components"],
        cov_type="full",
        sample_size=30000,
    )

    figures, axes = plot_gaussians(
        data=sampled_data,
        predicted_labels=predicted_labels,
        gauss_means=all_means_gauss,
        gauss_weights=all_weights_gauss,
        gauss_covariances=all_covariances_gauss,
        axis_names=AXIS_NAMES,
        upper_bounds=UPPER_BOUNDS,
        lower_bounds=LOWER_BOUNDS,
        bins=100,
        fig_size=(10, 10),
        plot_type="heatmap",
        heatmap_bins=100,
        heatmap_sigma=1,
    )

    for i, fig in enumerate(figures):
        fig_path = os.path.join(PLOTS_ROOT, f"{material}_GMM_{i}.png")
        fig.savefig(fig_path)

    component_dict = format_components(all_means_gauss, all_covariances_gauss)
    component_path = os.path.join(PARAM_ROOT, f"{material}_GMM.json")
    with open(component_path, "w") as f:
        json.dump(component_dict, f, indent=4, sort_keys=True)
