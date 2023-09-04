import numpy as np


def sort_components(component_means, component_covariances):
    sorted_indices = np.argsort(component_means[:, 2])[::-1]
    sorted_means = component_means[sorted_indices]
    sorted_covariances = component_covariances[sorted_indices]
    return sorted_means, sorted_covariances


def format_components(all_means_gauss, all_covariances_gauss):
    component_dict = {}

    all_means_gauss_sorted, all_covariances_gauss_sorted = sort_components(
        all_means_gauss, all_covariances_gauss
    )

    for component in range(all_means_gauss_sorted.shape[0]):
        component_dict[component + 1] = {}
        component_dict[component + 1]["contrast"] = {
            "r": all_means_gauss_sorted[component][2],
            "g": all_means_gauss_sorted[component][1],
            "b": all_means_gauss_sorted[component][0],
        }
        component_dict[component + 1][
            "covariance_matrix"
        ] = all_covariances_gauss_sorted[component].tolist()

    return component_dict
