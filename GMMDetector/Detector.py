import copy
import os
from textwrap import dedent
from typing import List

import cv2
import numpy as np
from joblib import load
from numba import jit, prange
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression

from .structures import Flake


class MaterialDetector:
    """
    The 2D Material Detector of the 2nd Insitute of Physics A, RWTH Aachen University\n
    The implementation is based on the following paper:\n
    "An open-source robust machine learning platform for real-time detection and classification of 2D material flakes"\n
    https://arxiv.org/abs/2306.14845
    """

    def __init__(
        self,
        contrast_dict: dict,
        size_threshold: int = 1000,
        standard_deviation_threshold: float = 5,
        used_channels: str = "BGR",
        false_positive_detector_path: str = None,
        **kwargs,
    ):
        """
        Initialize the Material Detector\n

        Args:
            contrast_dict (dict): The contrast dictionary of the material, Keys are the layer names, values are the contrast and the covariance matrix
            size_threshold (int, optional): The minimal size of a flake in pixels. Defaults to 1000, this is about 150 μm² in a 20x image.
            standard_deviation_threshold (float, optional): The maximal standard deviation threshold for the GMM of the contrast in a flake. Defaults to 5.
            used_channels (str, optional): The used channels for the detection. Defaults to "BGR" meaning all channels are used, BG would mean only the Blue and Green channel.
            false_positive_detector_path (str, optional): The path to the false positive detector model. Defaults to r"..\FalsePositiveDetector\models\classifier_L2_logistic.joblib". This is relative to the location of the Detector.py file.
        """

        self.contrast_dict = copy.deepcopy(contrast_dict)
        self.size_threshold = size_threshold
        self.standard_deviation_threshold = standard_deviation_threshold
        self.used_channels = used_channels

        self._try_loading_fp_detector(false_positive_detector_path)

        # add some more keys to the contrast_dict
        # the inverse of the cholesky decomposition of the covariance matrix in order to speed up the calculation of the distance
        # and the mean of the contrast to make it easier to handle
        # also handle the values of mean etc.. internally differently
        self.contrast_means = []
        self.inv_cholesky_matrices = []
        self.layer_index_lookup = {}
        self.layer_name_lookup = {}

        for layer_index, layer_name in enumerate(self.contrast_dict.keys()):
            # Save the mean of the contrast in the openCV format (BGR instead of RGB)
            contrast_mean = np.array(
                [
                    self.contrast_dict[layer_name]["contrast"]["b"],
                    self.contrast_dict[layer_name]["contrast"]["g"],
                    self.contrast_dict[layer_name]["contrast"]["r"],
                ]
            )
            self.contrast_means.append(contrast_mean)

            # Calculate the inverse of the cholesky decomposition of the covariance matrix
            # This is done in order to speed up the calculation of the distance in the Gaussian Mixtures
            covariance_matrix = np.array(
                self.contrast_dict[layer_name]["covariance_matrix"]
            )
            inv_cholesky = np.linalg.inv(np.linalg.cholesky(covariance_matrix))
            self.inv_cholesky_matrices.append(inv_cholesky)

            # Save a Lookup Table for the Layer Indexes and Layer Names
            self.layer_index_lookup[layer_name] = layer_index + 1
            self.layer_name_lookup[layer_index] = layer_name

        self.contrast_means = np.array(self.contrast_means)
        self.inv_cholesky_matrices = np.array(self.inv_cholesky_matrices)

    def _get_used_channel_indexes(self) -> List[int]:
        """
        Interprets the used_channels string and returns the indexes of the used channels\n
        An example:\n
        "BGR" -> [0,1,2]\n
        "GR" -> [1,2]\n

        Returns:
            List[int]: The indexes of the used channels
        """
        used_channel_indexes = []
        for channel in self.used_channels:
            if channel.lower() == "b":
                used_channel_indexes.append(0)
            elif channel.lower() == "g":
                used_channel_indexes.append(1)
            elif channel.lower() == "r":
                used_channel_indexes.append(2)
        used_channel_indexes.sort()
        return np.array(used_channel_indexes)

    def _try_loading_fp_detector(self, path: str) -> None:
        try:
            if path is None:
                detector_path = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "FalsePositiveDetector",
                    "models",
                    "classifier_L2_logistic.joblib",
                )
            else:
                detector_path = path

            self.FP_Detector: LogisticRegression = load(detector_path)

        except FileNotFoundError as e:
            print(
                dedent(
                    f"""
                    The False Positive Detector Model could not be found at {os.path.join(os.path.dirname(__file__), path)}.
                    Please make sure that the path is correct and the model is available.
                    If the model is not yet created consider running the False Positive Detector training script (FalsePositiveDetector/train_false_positive_detector.py) first.
                    Continuing without the False Positive Detector.
                    """
                )
            )
            self.FP_Detector = None

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Flake]:
        return self.detect_flakes(image)

    @staticmethod
    def get_mean_background_values(
        image: np.ndarray,
        radius: int = 5,
        min_value: int = 20,
        max_value: int = 230,
    ) -> np.ndarray:
        """
        Calculates the mean background values for each channel\n
        Takes the mean around the mode of the histogram of the image\n

        Args:
            image (NxMx3 Numpy Array): The image to calculate the mean background values from.
            radius (int, optional): The size of the area around the mode of the histogram used for the calcuations. Defaults to 5.
            min_value (int, optional): The minimum value of the histogram used for the calcuations, everything under this value will not be used. Defaults to 20.
            max_value (int, optional): The maximum value of the histogram used for the calcuations, everything above this value will not be used. Defaults to 230.

        Returns:
            np.ndarray: The mean background values for each channel in form BGR, dtype=np.uint8
        """
        masks = []

        for c in range(3):
            # masking out any large and small values
            # overexposed pixels are larger than 230 and underexposed pixels are smaller than 20
            image_channel = image[:, :, c]
            image_mask = cv2.inRange(image_channel, min_value, max_value)

            hist_mode = cv2.calcHist([image], [c], image_mask, [256], [0, 256]).argmax()

            masked_background_channel = cv2.inRange(
                image[:, :, c],
                int(hist_mode - radius),
                int(hist_mode + radius),
            )
            # masked_background_channel = cv2.erode(masked_background_channel, np.ones((3, 3)))
            masks.append(masked_background_channel)

        # combine the masks
        final_mask = cv2.bitwise_and(masks[0], masks[1])
        final_mask = cv2.bitwise_and(final_mask, masks[2])

        # find the mean b g r values of the background ~2 ms
        mean_background_values = np.array(
            cv2.mean(
                image,
                mask=final_mask,
            )[:-1],
            dtype=np.float32,
        )

        return mean_background_values

    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=True, nogil=True)
    def get_mean_background_values_numba(
        image: np.ndarray, radius: int = 5, min_value: int = 20, max_value: int = 230
    ) -> np.ndarray:
        """
        Calculates the mean background values for each channel\n
        Takes the mean around the mode of the histogram of the image\n

        Args:
            image (NxMx3 Numpy Array): The image to calculate the mean background values from.
            radius (int, optional): The size of the area around the mode of the histogram used for the calcuations. Defaults to 5.
            min_value (int, optional): The minimum value of the histogram used for the calcuations, everything under this value will not be used. Defaults to 20.
            max_value (int, optional): The maximum value of the histogram used for the calcuations, everything above this value will not be used. Defaults to 230.

        Returns:
            np.ndarray: The mean background values for each channel in form BGR, dtype=np.uint8
        """
        # Output array to store means for each channel
        means = np.zeros(3, dtype=np.float32)

        # Process each channel
        for channel in prange(3):
            channel_data = image[:, :, channel].flatten()
            count = np.zeros(256, dtype=np.uint32)  # 256 for all possible uint8 values

            # Build histogram
            for value in channel_data:
                if min_value <= value <= max_value:
                    count[value] += 1

            # Restrict mode calculation to within min_value and max_value
            mode_value = np.argmax(count[min_value : max_value + 1]) + min_value

            # Compute mean around the mode
            lower_bound = max(min_value, mode_value - radius)
            upper_bound = min(max_value, mode_value + radius)

            sum_values = 0
            num_values = 0
            for value in channel_data:
                if lower_bound <= value <= upper_bound:
                    sum_values += value
                    num_values += 1

            means[channel] = sum_values / num_values if num_values > 0 else 0

        return means

    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=True, nogil=True)
    def calculate_contrast_image(
        image: np.ndarray,
        mean_background_values: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the contrast image from the image and the mean background values\n
        Sped up by using numba\n

        Args:
            image (NxMx3 Numpy Array): The image to calculate the contrast image from.
            mean_background_values (NxMx3 Numpy Array): The mean background values for each channel in form BGR, dtype=np.uint8

        Returns:
            contrast_image (NxMx3 Numpy Array): The contrast image of the image
        """
        contrast_image = image / mean_background_values - 1
        return contrast_image

    def get_closest_component(self, pixel: np.ndarray) -> int:
        return MaterialDetector.assign_components_to_pixel(
            pixel,
            self.contrast_means,
            self.inv_cholesky_matrices,
            self.standard_deviation_threshold,
        )

    @staticmethod
    @jit(fastmath=True, nopython=True, nogil=True)
    def assign_components_to_pixel(
        pixel: np.ndarray,
        means: np.ndarray,
        inv_choleskys: np.ndarray,
        standard_deviations: float,
    ):
        """
        Assigns one pixel of format BGR to a component of the Gaussian Mixture\n

        Args:
            pixel (Kx1 Numpy Array): The contrast image of the image with K channels, dtype=np.float32
            means (Cx1 Numpy Array): The means of the Gaussian Mixture with C components
            inv_choleskys (CxKxK Numpy Array): The inverse of the cholesky decomposition of the covariance matrix of the Gaussian Mixture with C components
            standard_deviations (float): The maximum standard deviation of the Gaussian Mixture for which a pixel is still assigned to a component

        Returns:
            Flake Component Mask (NxM Numpy Array): The mask of the flake components, dtype=np.uint8, the value of each pixel is the index of the component it is assigned to; 0 means that the pixel is not assigned to any component
        """
        maximum_squared_stddev = standard_deviations**2
        tmp = 0
        smallest_distance = -1
        current_closest_layer = 0
        # calculate the distance for each thickness and select the one with the smallest distance
        for component_index in range(means.shape[0]):
            # mh_dist : Mahalanobis Distance of the current pixel to the current gaussian
            mh_dist = 0

            # run the following loop to calculate the distance
            # calculate the Mahalanobis distance by utilizing the speedup of the inverse Cholesky decomposition
            ###################
            # tmp_1 = inv_cholesky[0, 0] * diff[0]
            # tmp_2 = inv_cholesky[1, 0] * diff[0] + inv_cholesky[1, 1] * diff[1]
            # tmp_3 = inv_cholesky[2, 0] * diff[0] + inv_cholesky[2, 1] * diff[1] + inv_cholesky[2, 2] * diff[2]
            # dist = tmp_1 **2 + tmp_2 **2 + tmp_3 **2
            ###################
            for k in range(pixel.shape[0]):
                tmp = 0
                for h in range(k + 1):
                    tmp += (means[component_index, h] - pixel[h]) * inv_choleskys[
                        component_index, k, h
                    ]
                mh_dist += tmp**2

                # we can break the loop if the distance is already bigger than the maximal allowed standard deviation
                # we can do this as the mh distance only increases with more iterations
                if mh_dist > maximum_squared_stddev:
                    break

            # skip further calculations if the distance is already bigger than the maximal allowed standard deviation
            if mh_dist > maximum_squared_stddev:
                continue

            # Set the current layer as the closest layer if the distance is smaller than the current smallest distance
            # or if the current smallest distance is -1 (which means that it is the first iteration)
            if mh_dist < smallest_distance or smallest_distance == -1:
                smallest_distance = mh_dist
                current_closest_layer = component_index + 1
        return current_closest_layer

    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=True, nogil=True)
    def assign_components_to_pixels(
        contrast_image: np.ndarray,
        means: np.ndarray,
        inv_choleskys: np.ndarray,
        standard_deviations: float,
    ) -> np.ndarray:
        """
        Assigns each pixel of the input image to a component of the Gaussian Mixture\n

        Args:
            contrast_image (NxMxK Numpy Array): The contrast image of the image with K channels, dtype=np.float32
            means (Cx1 Numpy Array): The means of the Gaussian Mixture with C components
            inv_choleskys (CxKxK Numpy Array): The inverse of the cholesky decomposition of the covariance matrix of the Gaussian Mixture with C components
            standard_deviations (float): The maximum standard deviation of the Gaussian Mixture for which a pixel is still assigned to a component

        Returns:
            Flake Component Mask (NxM Numpy Array): The mask of the flake components, dtype=np.uint8, the value of each pixel is the index of the component it is assigned to; 0 means that the pixel is not assigned to any component
        """
        maximum_squared_stddev = standard_deviations**2
        tmp = 0
        smallest_distance = -1
        current_closest_layer = 0
        flake_component_mask = np.zeros(
            shape=(contrast_image.shape[0], contrast_image.shape[1]),
            dtype=np.uint8,
        )

        # Iterate over all pixels
        for i in prange(flake_component_mask.shape[0]):
            for j in prange(flake_component_mask.shape[1]):
                # use -1 as a standin for infinity
                smallest_distance = -1
                current_closest_layer = 0

                # calculate the distance for each thickness and select the one with the smallest distance
                for component_index in range(means.shape[0]):
                    # mh_dist : Mahalanobis Distance of the current pixel to the current gaussian
                    mh_dist = 0

                    # run the following loop to calculate the distance
                    # calculate the Mahalanobis distance by utilizing the speedup of the inverse Cholesky decomposition
                    ###################
                    # tmp_1 = inv_cholesky[0, 0] * diff[0]
                    # tmp_2 = inv_cholesky[1, 0] * diff[0] + inv_cholesky[1, 1] * diff[1]
                    # tmp_3 = inv_cholesky[2, 0] * diff[0] + inv_cholesky[2, 1] * diff[1] + inv_cholesky[2, 2] * diff[2]
                    # dist = tmp_1 **2 + tmp_2 **2 + tmp_3 **2
                    ###################
                    for k in range(contrast_image.shape[2]):
                        tmp = 0
                        for h in range(k + 1):
                            tmp += (
                                means[component_index, h] - contrast_image[i, j, h]
                            ) * inv_choleskys[component_index, k, h]
                        mh_dist += tmp**2

                        # we can break the loop if the distance is already bigger than the maximal allowed standard deviation
                        # we can do this as the mh distance only increases with more iterations
                        if mh_dist > maximum_squared_stddev:
                            break

                    # skip further calculations if the distance is already bigger than the maximal allowed standard deviation
                    if mh_dist > maximum_squared_stddev:
                        continue

                    # Set the current layer as the closest layer if the distance is smaller than the current smallest distance
                    # or if the current smallest distance is -1 (which means that it is the first iteration)
                    if mh_dist < smallest_distance or smallest_distance == -1:
                        smallest_distance = mh_dist
                        current_closest_layer = component_index + 1

                flake_component_mask[i, j] = current_closest_layer

        return flake_component_mask

    def _get_fp_probability(
        self,
        flake_contour: np.ndarray,
    ) -> float:
        """
        Calculates the probability of the flake being a false positive.\n
        Uses the False Positive Detector.

        Args:
            flake_contour (np.ndarray): A CV2 contour of the flake

        Returns:
            float: The probability of the flake being a false positive; between 0 and 1
        """
        convex_hull = cv2.convexHull(flake_contour)
        convex_hull_area = cv2.contourArea(convex_hull)
        arclength = cv2.arcLength(flake_contour, True)
        area = cv2.contourArea(flake_contour)
        solidity = float(area) / convex_hull_area
        arcarea = arclength / area**0.5

        return round(self.FP_Detector.predict_proba([[arcarea, solidity]])[0][0], 3)

    def _get_mean_entropy(
        self,
        image: np.ndarray,
        masked_flake: np.ndarray,
        flake_contour: np.ndarray,
    ) -> float:
        """
        Calculates the mean shannon entropy of the flake.\n

        Args:
            image (np.ndarray): The original image
            masked_flake (np.ndarray): The mask of the flake
            flake_contour (np.ndarray): The CV2 contour of the flake

        Returns:
            float: The mean shannon entropy of the flake
        """
        x, y, w, h = cv2.boundingRect(flake_contour)

        # We use a bounding box to speed up the calculation
        # As shannon entropy is a local property, we can do this
        # But it is really slow to calculate it for the whole image
        # So using a local bounding box is a good compromise

        # Expand the Bounding Box
        x_min = max(x - 20, 0)
        x_max = min(x + w + 20, image.shape[1])
        y_min = max(y - 20, 0)
        y_max = min(y + h + 20, image.shape[0])

        # Cut out the Bounding boxes
        cut_out_flake = image[
            y_min:y_max,
            x_min:x_max,
        ]
        cut_out_flake_mask = masked_flake[
            y_min:y_max,
            x_min:x_max,
        ]

        # Erode the mask to not accidentally have the Edges in the mean
        cut_out_flake_mask = cv2.erode(cut_out_flake_mask, disk(2), iterations=2)

        # go to Gray as we only need the gray Entropy
        entropy_area_gray = cv2.cvtColor(cut_out_flake, cv2.COLOR_BGR2GRAY)

        # Do the entropy function, really quick as we only use a small part
        entropied_image_area = entropy(
            entropy_area_gray,
            footprint=disk(2),
            mask=cut_out_flake_mask,
        )

        return cv2.mean(
            entropied_image_area,
            mask=cut_out_flake_mask,
        )[0]

    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=True, nogil=True)
    def _generate_mh_distance_map(
        image: np.ndarray,
        means: np.ndarray,
        inv_choleskys: np.ndarray,
        max_mode_value: int = 230,
        min_mode_value: int = 20,
        mode_radius: int = 5,
    ) -> np.ndarray:
        """Generate the Mahalanobis Distance Map of the image given the Gaussian Mixture Componentes\n
        You should call this function via the `generate_mh_distance_map` wrapper function of the MaterialDetector class\n

        Args:
            image (np.ndarray): The image of shape H x W x C, dtype=np.uint8
            means (np.ndarray): The means of the Gaussian Mixture with C components.
            inv_choleskys (np.ndarray): The inverse of the cholesky decomposition of the covariance matrix of the Gaussian Mixture with C components.
            max_mode_value (int, optional): The Maximum Value to evaluate the mode to, might come in handy if your image has large white areas such as overexposed flakes. Defaults to 230.
            min_mode_value (int, optional): The Minimum Value to evaluate the mode to, might come in handy if your image has large black areas such as shadows. Defaults to 20.
            mode_radius (int, optional): The Radius around the mode to calculate the mean background color. Defaults to 5.

        Returns:
            np.ndarray: An array of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
            The values of the array are the Mahalanobis Distances of the pixels to the components
        """

        ############### get the mean background values ~ 6ms
        # Output array to store means for each channel
        background_mean_values = np.zeros(3, dtype=np.float32)

        # Process each channel
        for channel in prange(3):
            channel_data = image[:, :, channel].flatten()
            count = np.zeros(256, dtype=np.uint32)  # 256 for all possible uint8 values

            # Build histogram
            for value in channel_data:
                if min_mode_value <= value <= max_mode_value:
                    count[value] += 1

            # Restrict mode calculation to within min_mode_value and max_mode_value
            mode_value = (
                np.argmax(count[min_mode_value : max_mode_value + 1]) + min_mode_value
            )

            # Compute mean around the mode
            lower_bound = max(min_mode_value, mode_value - mode_radius)
            upper_bound = min(max_mode_value, mode_value + mode_radius)

            sum_values = 0
            num_values = 0
            for value in channel_data:
                if lower_bound <= value <= upper_bound:
                    sum_values += value
                    num_values += 1

            background_mean_values[channel] = (
                sum_values / num_values if num_values > 0 else 0
            )
        ############### calucalute the contast imate

        assert (
            (background_mean_values[0] != 0)
            and (background_mean_values[1] != 0)
            and (background_mean_values[2] != 0)
        ), "The mean background values are 0, this is not possible, there seems to be a problem with the image"

        contrast_image = image / background_mean_values - 1

        mh_distance_map = np.zeros(
            shape=(means.shape[0], contrast_image.shape[0], contrast_image.shape[1]),
            dtype=np.float32,
        )

        # Iterate over all pixels
        for component_index in prange(means.shape[0]):
            for i in prange(contrast_image.shape[0]):
                for j in prange(contrast_image.shape[1]):
                    # run the following loop to calculate the distance
                    # calculate the Mahalanobis distance by utilizing the speedup of the inverse Cholesky decomposition
                    ###################
                    # tmp_1 = inv_cholesky[0, 0] * diff[0]
                    # tmp_2 = inv_cholesky[1, 0] * diff[0] + inv_cholesky[1, 1] * diff[1]
                    # tmp_3 = inv_cholesky[2, 0] * diff[0] + inv_cholesky[2, 1] * diff[1] + inv_cholesky[2, 2] * diff[2]
                    # dist = tmp_1 **2 + tmp_2 **2 + tmp_3 **2
                    ###################

                    # mh_dist : Mahalanobis Distance of the current pixel to the current gaussian
                    mh_dist = 0
                    for k in range(contrast_image.shape[2]):
                        tmp = 0
                        for h in range(k + 1):
                            tmp += (
                                means[component_index, h] - contrast_image[i, j, h]
                            ) * inv_choleskys[component_index, k, h]
                        mh_dist += tmp**2

                    mh_distance_map[component_index, i, j] = np.sqrt(mh_dist)

        return mh_distance_map

    def generate_mh_distance_map(
        self,
        image: np.ndarray,
        max_mode_value: int = 230,
        min_mode_value: int = 20,
        mode_radius: int = 5,
    ) -> np.ndarray:
        """Generate the Mahalanobis Distance Map of the image given the Gaussian Mixture Componentes\n

        Args:
            image (np.ndarray): The image of shape H x W x C, dtype=np.uint8
            max_mode_value (int, optional): The Maximum Value to evaluate the mode to, might come in handy if your image has large white areas such as overexposed flakes. Defaults to 230.
            min_mode_value (int, optional): The Minimum Value to evaluate the mode to, might come in handy if your image has large black areas such as shadows. Defaults to 20.
            mode_radius (int, optional): The Radius around the mode to calculate the mean background color. Defaults to 5.

        Returns:
            np.ndarray: An array of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
            The values of the array are the Mahalanobis Distances of the pixels to the components
        """
        return MaterialDetector._generate_mh_distance_map(
            image,
            self.contrast_means,
            self.inv_cholesky_matrices,
            max_mode_value,
            min_mode_value,
            mode_radius,
        )

    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=True, nogil=True)
    def _generate_mh_distance_map_from_contrast_image(
        contrast_image: np.ndarray,
        means: np.ndarray,
        inv_choleskys: np.ndarray,
    ) -> np.ndarray:
        """Generate the Mahalanobis Distance Map of the image given the Gaussian Mixture Componentes\n
        You should call this function via the `generate_mh_distance_map_from_contrast_image` wrapper function of the MaterialDetector class\n

        Args:
            contrast_image (np.ndarray): The image of shape H x W x C, dtype=np.uint8
            means (np.ndarray): The means of the Gaussian Mixture with C components.
            inv_choleskys (np.ndarray): The inverse of the cholesky decomposition of the covariance matrix of the Gaussian Mixture with C components.

        Returns:
            np.ndarray: An array of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
            The values of the array are the Mahalanobis Distances of the pixels to the components
        """

        mh_distance_map = np.zeros(
            shape=(means.shape[0], contrast_image.shape[0], contrast_image.shape[1]),
            dtype=np.float32,
        )

        # Iterate over all pixels
        for component_index in prange(means.shape[0]):
            for i in prange(contrast_image.shape[0]):
                for j in prange(contrast_image.shape[1]):
                    # run the following loop to calculate the distance
                    # calculate the Mahalanobis distance by utilizing the speedup of the inverse Cholesky decomposition
                    ###################
                    # tmp_1 = inv_cholesky[0, 0] * diff[0]
                    # tmp_2 = inv_cholesky[1, 0] * diff[0] + inv_cholesky[1, 1] * diff[1]
                    # tmp_3 = inv_cholesky[2, 0] * diff[0] + inv_cholesky[2, 1] * diff[1] + inv_cholesky[2, 2] * diff[2]
                    # dist = tmp_1 **2 + tmp_2 **2 + tmp_3 **2
                    ###################

                    # mh_dist : Mahalanobis Distance of the current pixel to the current gaussian
                    mh_dist = 0
                    for k in range(contrast_image.shape[2]):
                        tmp = 0
                        for h in range(k + 1):
                            tmp += (
                                means[component_index, h] - contrast_image[i, j, h]
                            ) * inv_choleskys[component_index, k, h]
                        mh_dist += tmp**2

                    mh_distance_map[component_index, i, j] = np.sqrt(mh_dist)

        return mh_distance_map

    def generate_mh_distance_map_from_contrast_image(
        self,
        contrast_image: np.ndarray,
    ) -> np.ndarray:
        """Generate the Mahalanobis Distance Map of the Contrast image given the Gaussian Mixture Componentes\n
        If you want to directly get the MH Distance Map you should call `generate_mh_distance_map` with the original image.

        Args:
            contrast_image (np.ndarray): The Contrast image of shape H x W x C, dtype=np.uint8

        Returns:
            np.ndarray: An array of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
            The values of the array are the Mahalanobis Distances of the pixels to the components
        """
        if len(self.used_channels) == 3:
            return MaterialDetector._generate_mh_distance_map_from_contrast_image(
                contrast_image,
                self.contrast_means,
                self.inv_cholesky_matrices,
            )
        used_channel_indexes = self._get_used_channel_indexes()
        return MaterialDetector._generate_mh_distance_map_from_contrast_image(
            contrast_image[:, :, used_channel_indexes],
            self.contrast_means[:, used_channel_indexes],
            self.inv_cholesky_matrices[:, used_channel_indexes, :][
                :, :, used_channel_indexes
            ],
        )

    def postprocess_mh_map(
        self, distance_map: np.ndarray, distance_threshold: float = 5
    ) -> np.ndarray:
        """Postprocesses the Mahalanobis distance map to get the semantic map of the flakes\n
        This generates a semantic map of flakes with no overlap\n

        Args:
            distance_map (np.ndarray): The Mahalanobis distance map of the image of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
            distance_threshold (float, optional): The Maximum Distance a value can have in Standard deviation. Defaults to 5.

        Returns:
            np.ndarray: The semantic map of the flakes of shape (K x H x W) with K being the number of components and H and W being the height and width of the image
        """
        # Set every value that is not the minimum to the standard deviation threshold
        # this is done along the K axis
        distance_map[distance_map != distance_map.min(axis=0)] = distance_threshold

        # finally we can threshold the map to get the semantic map
        # this map has the shape (K x H x W) with K being the number of components and
        semantic_masks = (distance_map < distance_threshold).astype(np.uint8)

        return semantic_masks

    def detect_flakes(
        self,
        image: np.ndarray,
    ) -> List[Flake]:
        """
        Detects Flakes in the given Image.\n
        Expects images without vignette.\n

        Args:
            image (NxMx3 Numpy Array): The original image without vignette, Expected to be in format BGR

        Returns:
            (Kx1 Numpy Array): An Array of Flakes
        """

        assert (
            image.shape[2] == 3
        ), f"The Image has to have the shape of NxMx3, the shape is {image.shape}"
        assert image.dtype == np.uint8, "The Image has to be of type uint8"

        detected_flakes = []

        image = cv2.medianBlur(image, 5)

        mean_background_values = MaterialDetector.get_mean_background_values_numba(
            image,
        )

        contrast_image = MaterialDetector.calculate_contrast_image(
            image,
            mean_background_values,
        )

        mh_distance_map = self.generate_mh_distance_map_from_contrast_image(
            contrast_image
        )

        semantic_masks = self.postprocess_mh_map(
            mh_distance_map, distance_threshold=self.standard_deviation_threshold
        )

        for layer_index, layer_mask in enumerate(semantic_masks):
            layer_name = self.layer_name_lookup[layer_index]

            # Remove small outliers
            layer_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_OPEN, disk(2))

            # Skip the layer if there are not enough pixels
            if cv2.countNonZero(layer_mask) < self.size_threshold:
                continue

            # label each connected 'blob' on the mask with an individual number
            # each of these blobs is a flake candidate
            num_labels, labeled_mask = cv2.connectedComponents(
                layer_mask,
                connectivity=4,
            )

            # iterate over all flake candidates IDs, the 0 ID is the background
            for i in range(1, num_labels + 1):
                # mask out only the pixels of certain flakes, quite fast ~1ms
                masked_flake = cv2.inRange(labeled_mask, i, i)

                # if the flake has less pixel than the Threshold skip it
                flake_size = cv2.countNonZero(masked_flake)
                if flake_size < self.size_threshold:
                    continue

                contours, hierarchy = cv2.findContours(
                    image=masked_flake,
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_NONE,
                )

                # extract the toplevel contour by finding the contour with no parents
                top_level_contour = [
                    contours[i]
                    for i in range(len(contours))
                    if hierarchy[0, i, 3] == -1
                ]

                # Fill all the holes in the contour by redrawing the contour
                masked_flake = cv2.drawContours(
                    masked_flake, top_level_contour, -1, 255, -1
                )

                #### Calculate the Contrast of the Flakes
                mean_contrast, _ = cv2.meanStdDev(contrast_image, mask=masked_flake)
                mean_contrast = mean_contrast[:, 0].tolist()

                #### Calculate the false positive probability
                if self.FP_Detector is not None:
                    false_positive_probability = self._get_fp_probability(
                        top_level_contour[0]
                    )
                else:
                    false_positive_probability = 0

                #### Calculate the Aspect Ratio and Center Position of the Flake
                ((center_x, center_y), (width_r, height_r), _) = cv2.minAreaRect(
                    top_level_contour[0]
                )

                #### Calculate the Entropy
                flake_entropy = self._get_mean_entropy(
                    image,
                    masked_flake,
                    top_level_contour[0],
                )

                flake = Flake(
                    mask=masked_flake,
                    thickness=layer_name,
                    size=flake_size,
                    mean_contrast=mean_contrast,
                    center=(int(center_x), int(center_y)),
                    min_sidelength=min(width_r, height_r),
                    max_sidelength=max(width_r, height_r),
                    false_positive_probability=false_positive_probability,
                    entropy=flake_entropy,
                )

                detected_flakes.append(flake)

        return np.array(detected_flakes)
