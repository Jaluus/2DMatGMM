import itertools

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter


def confidence_ellipse(
    ax,
    mean,
    cov,
    n_std=3.0,
    facecolor="none",
    **kwargs,
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    mean_x = mean[0]
    mean_y = mean[1]

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        linewidth=2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_heatmap_plot(
    data,
    axis_names=None,
    sigma=3,
    bins=100,
    processing_function=lambda x: np.log(x + 1),
    upper_bounds=None,
    lower_bounds=None,
    title="Full 3D Contrast Heatmap",
    used_channels="BGR",
):
    assert data.shape[1] == 3, "Data must be 3D"
    assert len(used_channels) in [2, 3], "Number of used channels must be 2 or 3"

    if not (axis_names is None):
        assert len(axis_names) == 3, "There must be 3 axis names or None"

    # change the font size of the plot
    plt.rcParams.update({"font.size": 20})

    # extract the used channels and all possible combinations
    # these are either 3 or 1 combinations
    used_channel_indices = ["BGR".index(channel) for channel in used_channels]
    channel_combinations = list(itertools.combinations(used_channel_indices, 2))

    fig, axis = plt.subplots(
        1,
        len(channel_combinations),
        figsize=(7 * len(channel_combinations), 7),
        dpi=300,
    )
    fig.suptitle(title)

    # if only one channel is used, we need to add an axis to the list
    if len(channel_combinations) == 1:
        axis = [axis]

    for idx, (i, j) in enumerate(channel_combinations):
        # Generate some test data
        x = data[:, i]
        y = data[:, j]

        if lower_bounds is not None and upper_bounds is not None:
            x_lower = lower_bounds[i]
            x_upper = upper_bounds[i]
            y_lower = lower_bounds[j]
            y_upper = upper_bounds[j]
            img, extent = create_heatmap(
                x,
                y,
                sigma=sigma,
                bins=bins,
                extent=[x_lower, x_upper, y_lower, y_upper],
            )
        else:
            img, extent = create_heatmap(x, y, sigma=sigma, bins=bins)

        img = processing_function(img)

        axis[idx].imshow(
            img, extent=extent, origin="lower", cmap=cm.plasma, aspect="auto"
        )

        axis[idx].set_xlabel(axis_names[i])
        axis[idx].set_ylabel(axis_names[j])

        axis[idx].grid()

    # set space between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def plot_gaussians(
    data,
    predicted_labels,
    gauss_means,
    gauss_weights,
    gauss_covariances,
    lower_bounds,
    upper_bounds,
    axis_names=["Blue Contrast", "Green Contrast", "Red Contrast"],
    heatmap_sigma=3,
    heatmap_bins=200,
    plot_type="scatter",
    bins=50,
    fig_size=(10, 10),
    used_channels="BGR",
) -> plt.Figure:
    assert plot_type in ["scatter", "heatmap"], "Type must be either scatter or heatmap"

    # set the font size of the plot
    plt.rcParams.update({"font.size": 20})

    def gauss(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
            sigma * np.sqrt(2 * np.pi)
        )

    used_channel_indices = ["BGR".index(channel) for channel in used_channels]
    channel_combinations = list(itertools.combinations(used_channel_indices, 2))

    figures = []
    axes = []

    for idx, (i, j) in enumerate(channel_combinations):
        x_lower = lower_bounds[i]
        x_upper = upper_bounds[i]
        y_lower = lower_bounds[j]
        y_upper = upper_bounds[j]

        X_DATA = data[:, [i, j]]
        means = gauss_means[:, [i, j]]
        covariances = gauss_covariances[:, :, [i, j]][:, [i, j], :]
        weights = gauss_weights

        fig = plt.figure(figsize=fig_size, dpi=300)

        # Define the plot windows
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(7, 2),
            height_ratios=(2, 7),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.1,
            hspace=0.1,
        )

        # add subplots for the data
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        ax_histx.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax_histy.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )

        ax_histy.grid()
        ax_histx.grid()

        # add the histograms
        ax_histx.hist(X_DATA[:, 0], bins=bins, density=True)

        ax_histy.hist(X_DATA[:, 1], bins=bins, orientation="horizontal", density=True)
        ax_histx.set_ylabel("Density")
        ax_histy.set_xlabel("Density")

        # add the confidence ellipse and gaussians to the histograms
        for k in range(means.shape[0]):
            gauss_color = "red"  # cm.get_cmap("Dark2")(j / means.shape[0])

            component_covariance = covariances[k]
            component_mean = means[k]
            component_weight = weights[k]

            x_range = np.linspace(
                component_mean[0] - 3 * np.sqrt(component_covariance[0, 0]),
                component_mean[0] + 3 * np.sqrt(component_covariance[0, 0]),
                bins * 10,
            )
            y_range = np.linspace(
                component_mean[1] - 3 * np.sqrt(component_covariance[1, 1]),
                component_mean[1] + 3 * np.sqrt(component_covariance[1, 1]),
                bins * 10,
            )

            x_gauss = gauss(
                x_range,
                component_mean[0],
                np.sqrt(component_covariance[0, 0]),
            )
            y_gauss = gauss(
                y_range,
                component_mean[1],
                np.sqrt(component_covariance[1, 1]),
            )

            ax_histx.plot(
                x_range, x_gauss * component_weight, color=gauss_color, linewidth=3
            )
            ax_histy.plot(
                y_gauss * component_weight, y_range, color=gauss_color, linewidth=3
            )

            for h in range(3):
                confidence_ellipse(
                    ax,
                    mean=component_mean,
                    cov=component_covariance,
                    n_std=h + 1,
                    facecolor="none",
                    edgecolor="white",
                    alpha=0.5,
                )

        if plot_type == "scatter":
            ax.scatter(
                X_DATA[:, 0],
                X_DATA[:, 1],
                s=10,
                marker="s",
                alpha=0.04,
                ec="None",
                c=predicted_labels,
                cmap="Dark2",
            )
        else:
            heatmap, extent = create_heatmap(
                X_DATA[:, 0],
                X_DATA[:, 1],
                sigma=heatmap_sigma,
                bins=heatmap_bins,
                extent=[x_lower, x_upper, y_lower, y_upper],
            )

            # normalize the heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap[heatmap > 0.5] = 0.5

            ax.imshow(
                heatmap, extent=extent, origin="lower", cmap=cm.plasma, aspect="auto"
            )

        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlabel(axis_names[i])
        ax.set_ylabel(axis_names[j])

        figures.append(fig)
        axes.append(ax)

    return figures, axes


def create_heatmap(
    x,
    y,
    sigma,
    bins=1000,
    extent=None,
):
    if extent is None:
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    else:
        hist_range = [[extent[0], extent[1]], [extent[2], extent[3]]]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range)

    heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap.T, extent
