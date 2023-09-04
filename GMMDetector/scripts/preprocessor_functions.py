import os

import cv2
import numpy as np


def get_contrasts_from_dir(
    image_directory,
    mask_directory,
    flatfield_path=None,
    use_flatfield=False,
):
    contrasts = []

    if use_flatfield and flatfield_path is not None:
        flatfield = cv2.imread(flatfield_path)
        assert (
            flatfield is not None
        ), f"Could not load flatfield at '{flatfield_path}', have you selected the correct path?"

    mask_names = os.listdir(mask_directory)

    for idx, mask_name in enumerate(mask_names):
        print(f"{idx + 1}/{len(mask_names)} read", end="\r")

        # check if the image is either in the png or jpg format
        image_path = os.path.join(image_directory, mask_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(
                image_directory, mask_name.replace(".png", ".jpg")
            )
        if not os.path.exists(image_path):
            print(f"Could not find image corresponding to mask '{mask_name}', skipping")

        mask_path = os.path.join(mask_directory, mask_name)

        mask = cv2.imread(mask_path, 0)
        image = cv2.imread(image_path)

        assert mask is not None, f"Could not load mask {mask_path}"
        assert image is not None, f"Could not load image {image_path}"

        mask = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )

        if cv2.countNonZero(mask) < 10:
            continue

        if use_flatfield and flatfield_path is not None:
            image = remove_vignette(image, flatfield)

        flake_color = np.array(image[mask != 0])
        background_color = np.array(calculate_background_color(image, 10))

        if np.any(background_color == 0):
            print(f"Error with image {mask_name}; Invalid Background, skipping")
            continue

        flake_contrast = (flake_color / background_color) - 1

        contrasts.extend(flake_contrast)

    contrasts = np.array(contrasts)

    return contrasts


def remove_vignette(
    image,
    flatfield,
    max_background_value: int = 241,
):
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flatfield (NxMx3 Array): the Flat Field in RGB
        max_background_value (int): the maximum value of the background

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """
    image_no_vigentte = image / flatfield * cv2.mean(flatfield)[:-1]
    image_no_vigentte[image_no_vigentte > max_background_value] = max_background_value
    return np.asarray(image_no_vigentte, dtype=np.uint8)


def calculate_background_color(img, radius=5):
    masks = []

    for i in range(3):
        img_channel = img[:, :, i]
        mask = cv2.inRange(img_channel, 20, 230)
        hist = cv2.calcHist([img_channel], [0], mask, [256], [0, 256])
        hist_mode = np.argmax(hist)
        thresholded_image = cv2.inRange(
            img_channel, int(hist_mode - radius), int(hist_mode + radius)
        )
        background_mask_channel = cv2.erode(
            thresholded_image, np.ones((3, 3)), iterations=3
        )
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])

    return cv2.mean(img, mask=final_mask)[:3]
