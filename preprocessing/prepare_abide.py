"""
ABIDE-I structural MRI preprocessing utilities.

The preprocessing pipeline follows the procedures described
in the manuscript:

    T1-weighted MRI
          ↓
    Skull stripping
          ↓
    Min-max intensity normalization
          ↓
    Central midsagittal slice extraction
          ↓
    Background cropping
          ↓
    Resize to 224 x 224
          ↓
    Grayscale-to-RGB conversion
          ↓
    Quality control
"""

from pathlib import Path

import numpy as np
from PIL import Image


def min_max_normalize(image):
    """
    Normalize image intensities to the range [0, 1].

    Parameters
    ----------
    image : numpy.ndarray
        MRI slice.

    Returns
    -------
    numpy.ndarray
        Intensity-normalized image.
    """

    image = image.astype(np.float32)

    image_min = np.min(image)
    image_max = np.max(image)

    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (
        (image - image_min)
        / (image_max - image_min)
    )

    return normalized


def extract_midsagittal_slice(volume):
    """
    Extract the central midsagittal slice.

    Parameters
    ----------
    volume : numpy.ndarray
        3-D structural MRI volume.

    Returns
    -------
    numpy.ndarray
        Central sagittal slice.
    """

    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3-D MRI volume."
        )

    center_index = volume.shape[0] // 2

    slice_2d = volume[
        center_index,
        :,
        :
    ]

    return slice_2d


def crop_background(image):
    """
    Crop zero-valued background from a 2-D MRI slice.

    Parameters
    ----------
    image : numpy.ndarray
        2-D MRI slice.

    Returns
    -------
    numpy.ndarray
        Cropped MRI slice.
    """

    nonzero = np.argwhere(
        image > 0
    )

    if nonzero.size == 0:
        return image

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)

    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return cropped


def resize_image(
    image,
    image_size=(224, 224)
):
    """
    Resize an MRI slice to 224 x 224.

    Parameters
    ----------
    image : numpy.ndarray
        2-D normalized MRI slice.

    image_size : tuple
        Target image size.

    Returns
    -------
    numpy.ndarray
        Resized image.
    """

    image_uint8 = (
        np.clip(image, 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image_uint8,
        mode="L"
    )

    pil_image = pil_image.resize(
        image_size,
        Image.Resampling.BILINEAR
    )

    resized = np.asarray(
        pil_image,
        dtype=np.float32
    ) / 255.0

    return resized


def grayscale_to_rgb(image):
    """
    Replicate a grayscale MRI slice into three channels.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image with shape [H, W].

    Returns
    -------
    numpy.ndarray
        Three-channel image with shape [H, W, 3].
    """

    if image.ndim != 2:
        raise ValueError(
            "Expected a 2-D grayscale image."
        )

    rgb_image = np.stack(
        [image, image, image],
        axis=-1
    )

    return rgb_image


def preprocess_slice(
    image,
    image_size=(224, 224)
):
    """
    Apply the image-level preprocessing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        2-D MRI slice.

    image_size : tuple
        Target image size.

    Returns
    -------
    numpy.ndarray
        Preprocessed RGB image:
        [224, 224, 3]
    """

    # Intensity normalization.
    image = min_max_normalize(
        image
    )

    # Remove background.
    image = crop_background(
        image
    )

    # Resize to 224 x 224.
    image = resize_image(
        image,
        image_size=image_size
    )

    # Convert grayscale to three channels.
    image = grayscale_to_rgb(
        image
    )

    return image


def quality_control(image):
    """
    Perform basic quality checks.

    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed image.

    Returns
    -------
    bool
        True if the image passes the basic checks.
    """

    if image.shape != (224, 224, 3):
        return False

    if not np.isfinite(image).all():
        return False

    if image.min() < 0.0:
        return False

    if image.max() > 1.0:
        return False

    return True


def prepare_dataset():
    """
    Entry point for dataset preparation.

    The actual ABIDE-I file discovery and NIfTI loading
    should be connected to the local dataset separately.
    """

    raise NotImplementedError(
        "Connect this function to the local ABIDE-I "
        "NIfTI dataset after verifying file paths and "
        "subject-to-phenotype matching."
    )


if __name__ == "__main__":

    # Simple preprocessing test using a dummy MRI slice.
    dummy_slice = np.random.rand(
        256,
        256
    ).astype(np.float32)

    processed = preprocess_slice(
        dummy_slice
    )

    print(
        "Processed image shape:",
        processed.shape
    )

    print(
        "Minimum intensity:",
        processed.min()
    )

    print(
        "Maximum intensity:",
        processed.max()
    )

    print(
        "Quality control:",
        quality_control(processed)
    )
