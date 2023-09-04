import numpy as np
from sklearn.mixture import GaussianMixture


def fit_set(
    data,
    num_components,
    cov_type,
    num_additional_noise_comp=0,
    sample_size=30000,
    used_channels="BGR",
    initial_means=None,
    **kwargs,
):
    assert cov_type in ["full", "tied", "diag"], "Invalid covariance type"

    np.random.seed(42)
    sum_components = num_components + num_additional_noise_comp
    used_channel_indices = ["BGR".index(c) for c in used_channels]

    # sample the data
    if sample_size is None or sample_size >= data.shape[0]:
        sampled_data = data
    else:
        assert sample_size > 0, "Sample size must be greater than 0"
        sampled_data = data[
            np.random.choice(data.shape[0], sample_size, replace=False), :
        ]

    # only use the channels we want
    sampled_data_fit = sampled_data[:, used_channel_indices]

    gaussian_model = GaussianMixture(
        n_components=sum_components,
        covariance_type=cov_type,
        max_iter=100,
        init_params="kmeans",
        n_init=10,
        tol=1e-4,
        reg_covar=1e-7,
        warm_start=False,
        means_init=initial_means,
        **kwargs,
    )

    gaussian_model.fit(sampled_data_fit)

    all_means_gauss = np.round(gaussian_model.means_, 3)
    all_weights_gauss = gaussian_model.weights_
    predicted_labels = gaussian_model.predict(sampled_data_fit)
    if cov_type == "tied":
        all_covariances_gauss = np.array([gaussian_model.covariances_] * sum_components)
    elif cov_type == "diag":
        all_covariances_gauss = np.array(
            [np.diag(cov) for cov in gaussian_model.covariances_]
        )
    else:
        all_covariances_gauss = gaussian_model.covariances_

    # pad the respective dimensions with zeros and ones
    if len(used_channels) == 2:
        # if we use "BG" the used channels are 0 and 1 thus we need to pad the the last dimension
        # if we use "GR" the used channels are 1 and 2 thus we need to pad the first dimension
        # if we use "BR" the used channels are 0 and 2 thus we need to pad the second dimension

        if used_channels == "GR":
            pad_dim = 0
        if used_channels == "BR":
            pad_dim = 1
        if used_channels == "BG":
            pad_dim = 2
        all_means_gauss = np.insert(all_means_gauss, pad_dim, 0, axis=1)

        all_covariances_gauss = np.insert(all_covariances_gauss, pad_dim, 0, axis=1)
        all_covariances_gauss = np.insert(all_covariances_gauss, pad_dim, 0, axis=2)
        all_covariances_gauss[:, pad_dim, pad_dim] = 1

    # Remove the additional component for noise if we used it
    # It has the largest eigenvalue by default as its ellipse is the biggest
    for i in range(num_additional_noise_comp):
        max_ev = 0
        max_ev_idx = 0
        # get the eigenvalues and eigenvectors
        for idx, cov in enumerate(all_covariances_gauss):
            eig_vals, _ = np.linalg.eig(cov)
            if np.sum(eig_vals) > max_ev:
                max_ev = np.sum(eig_vals)
                max_ev_idx = idx

        # remove the largest eigenvalue
        all_covariances_gauss = np.delete(all_covariances_gauss, max_ev_idx, axis=0)
        all_means_gauss = np.delete(all_means_gauss, max_ev_idx, axis=0)
        all_weights_gauss = np.delete(all_weights_gauss, max_ev_idx, axis=0)

    # reweight the gaussians to sum to 1
    all_weights_gauss = all_weights_gauss / np.sum(all_weights_gauss)

    return (
        all_means_gauss,
        all_covariances_gauss,
        all_weights_gauss,
        sampled_data,
        predicted_labels,
    )
