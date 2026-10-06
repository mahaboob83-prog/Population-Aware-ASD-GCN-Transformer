"""
ABIDE-I structural MRI preprocessing utilities.

Preprocessing pipeline:
    NIfTI MRI
        -> load volume
        -> skull stripping / brain masking
        -> min-max normalization
        -> midsagittal slice extraction
        -> background cropping
        -> resize to 224 x 224
        -> grayscale to 3-channel conversion
        -> quality control

The resulting image can be passed to create_patches.py for
32 x 32 non-overlapping patch extraction.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nilearn import image as nilearn_image


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224


# ---------------------------------------------------------------------
# NIfTI loading
# ---------------------------------------------------------------------

def load_nifti(nifti_path):
    """
    Load a NIfTI MRI file.

    Parameters
    ----------
    nifti_path : str or pathlib.Path
        Path to a .nii or .nii.gz file.

    Returns
    -------
    np.ndarray
        MRI volume as a floating-point NumPy array.
    """

    nifti_path = Path(nifti_path)

    if not nifti_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {nifti_path}"
        )

    if nifti_path.suffix not in {".nii", ".gz"} and not str(
        nifti_path
    ).endswith(".nii.gz"):
        raise ValueError(
            f"Expected a NIfTI file (.nii or .nii.gz), got: {nifti_path}"
        )

    nii = nib.load(str(nifti_path))

    volume = nii.get_fdata(dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, got shape {volume.shape}"
        )

    return volume


# ---------------------------------------------------------------------
# Skull stripping / brain masking
# ---------------------------------------------------------------------

def skull_strip(nifti_path):
    """
    Apply a brain mask to a NIfTI MRI volume.

    Parameters
    ----------
    nifti_path : str or pathlib.Path
        Path to the input NIfTI image.

    Returns
    -------
    np.ndarray
        Skull-stripped MRI volume.

    Notes
    -----
    This function uses Nilearn's automatic brain-mask estimation.
    The exact preprocessing used to generate the reported experimental
    results should be verified before using this function to reproduce
    those results.
    """

    nii = nib.load(str(nifti_path))

    if len(nii.shape) != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, got shape {nii.shape}"
        )

    # Compute a brain mask from the structural MRI.
    mask_img = nilearn_image.compute_brain_mask(nii)

    # Apply the mask to the MRI volume.
    masked_img = nilearn_image.math_img(
        "img * mask",
        img=nii,
        mask=mask_img
    )

    volume = masked_img.get_fdata(dtype=np.float32)

    return volume


# ---------------------------------------------------------------------
# Min-max normalization
# ---------------------------------------------------------------------

def min_max_normalize(volume):
    """
    Apply min-max normalization to an MRI volume.

    Parameters
    ----------
    volume : np.ndarray
        MRI volume.

    Returns
    -------
    np.ndarray
        Volume scaled to [0, 1].
    """

    volume = np.asarray(volume, dtype=np.float32)

    finite_mask = np.isfinite(volume)

    if not np.any(finite_mask):
        raise ValueError(
            "MRI volume does not contain any finite values."
        )

    valid_values = volume[finite_mask]

    minimum = valid_values.min()
    maximum = valid_values.max()

    if maximum == minimum:
        return np.zeros_like(volume, dtype=np.float32)

    normalized = (volume - minimum) / (maximum - minimum)

    normalized[~finite_mask] = 0.0

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------
# Midsagittal slice extraction
# ---------------------------------------------------------------------

def extract_midsagittal_slice(volume):
    """
    Extract the central midsagittal slice.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume with shape [X, Y, Z].

    Returns
    -------
    np.ndarray
        2D midsagittal slice.

    Notes
    -----
    This function assumes that the first volume axis corresponds to
    the sagittal direction. This assumption should match the orientation
    of the preprocessed ABIDE-I volumes used in the study.
    """

    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {volume.shape}"
        )

    sagittal_index = volume.shape[0] // 2

    slice_2d = volume[sagittal_index, :, :]

    return slice_2d.astype(np.float32)


# ---------------------------------------------------------------------
# Background cropping
# ---------------------------------------------------------------------

def crop_background(slice_2d, threshold=0.01):
    """
    Crop zero/background regions from a 2D MRI slice.

    Parameters
    ----------
    slice_2d : np.ndarray
        2D MRI slice normalized to approximately [0, 1].

    threshold : float
        Intensity threshold used to identify foreground pixels.

    Returns
    -------
    np.ndarray
        Cropped MRI slice.
    """

    image = np.asarray(slice_2d, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape}"
        )

    foreground = image > threshold

    if not np.any(foreground):
        # If no foreground is detected, return the original image.
        return image

    rows = np.where(np.any(foreground, axis=1))[0]
    columns = np.where(np.any(foreground, axis=0))[0]

    row_start = rows[0]
    row_end = rows[-1] + 1

    col_start = columns[0]
    col_end = columns[-1] + 1

    cropped = image[
        row_start:row_end,
        col_start:col_end
    ]

    return cropped.astype(np.float32)


# ---------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------

def resize_slice(slice_2d, image_size=IMAGE_SIZE):
    """
    Resize a 2D MRI slice to the required input size.

    Parameters
    ----------
    slice_2d : np.ndarray
        2D MRI slice.

    image_size : int
        Target height and width.

    Returns
    -------
    np.ndarray
        Resized image of shape [image_size, image_size].
    """

    image = np.asarray(slice_2d, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape}"
        )

    # Convert [0, 1] to 8-bit for PIL resizing.
    image_uint8 = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image_uint8,
        mode="L"
    )

    resized = pil_image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR
    )

    resized_array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    return resized_array


# ---------------------------------------------------------------------
# Grayscale -> 3-channel
# ---------------------------------------------------------------------

def grayscale_to_rgb(image):
    """
    Convert a grayscale MRI image into a 3-channel representation.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.

    Returns
    -------
    np.ndarray
        3-channel image with shape [3, H, W].
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    rgb = np.stack(
        [image, image, image],
        axis=0
    )

    return rgb.astype(np.float32)


# ---------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------

def quality_control(image, image_size=IMAGE_SIZE):
    """
    Perform basic quality-control checks on a processed MRI image.

    Parameters
    ----------
    image : np.ndarray
        Processed 2D or 3-channel image.

    image_size : int
        Expected spatial dimension.

    Returns
    -------
    bool
        True if the image passes the basic checks.
    """

    image = np.asarray(image)

    # Check dimensions.
    if image.ndim == 2:
        if image.shape != (image_size, image_size):
            return False

    elif image.ndim == 3:
        if image.shape[1:] != (image_size, image_size):
            return False

        if image.shape[0] != 3:
            return False

    else:
        return False

    # Check for NaN or Inf.
    if not np.all(np.isfinite(image)):
        return False

    # Check that the image contains some non-zero information.
    if np.max(image) <= 0:
        return False

    return True


# ---------------------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------------------

def preprocess_nifti(
    nifti_path,
    image_size=IMAGE_SIZE,
    apply_skull_stripping=True,
    crop=True
):
    """
    Complete preprocessing pipeline for one ABIDE-I MRI volume.

    Parameters
    ----------
    nifti_path : str or pathlib.Path
        Path to the input NIfTI MRI.

    image_size : int
        Target spatial dimension. Default is 224.

    apply_skull_stripping : bool
        Whether to apply Nilearn brain masking.

    crop : bool
        Whether to crop background before resizing.

    Returns
    -------
    np.ndarray
        Preprocessed 3-channel image with shape [3, 224, 224].
    """

    # ---------------------------------------------------------------
    # 1. Load MRI
    # ---------------------------------------------------------------

    if apply_skull_stripping:
        volume = skull_strip(nifti_path)
    else:
        volume = load_nifti(nifti_path)

    # ---------------------------------------------------------------
    # 2. Min-max normalization
    # ---------------------------------------------------------------

    volume = min_max_normalize(volume)

    # ---------------------------------------------------------------
    # 3. Extract central midsagittal slice
    # ---------------------------------------------------------------

    slice_2d = extract_midsagittal_slice(volume)

    # ---------------------------------------------------------------
    # 4. Crop background
    # ---------------------------------------------------------------

    if crop:
        slice_2d = crop_background(slice_2d)

    # ---------------------------------------------------------------
    # 5. Resize to 224 x 224
    # ---------------------------------------------------------------

    slice_2d = resize_slice(
        slice_2d,
        image_size=image_size
    )

    # ---------------------------------------------------------------
    # 6. Convert grayscale to 3 channels
    # ---------------------------------------------------------------

    image = grayscale_to_rgb(slice_2d)

    # ---------------------------------------------------------------
    # 7. Quality control
    # ---------------------------------------------------------------

    if not quality_control(
        image,
        image_size=image_size
    ):
        raise ValueError(
            f"Quality-control check failed for: {nifti_path}"
        )

    return image


# ---------------------------------------------------------------------
# NumPy -> PyTorch
# ---------------------------------------------------------------------

def image_to_tensor(image):
    """
    Convert a processed image to a PyTorch tensor.

    Parameters
    ----------
    image : np.ndarray
        Image of shape [3, H, W].

    Returns
    -------
    torch.Tensor
        Float tensor of shape [3, H, W].
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(
            f"Expected [C, H, W], got shape {image.shape}"
        )

    tensor = torch.from_numpy(image.copy())

    return tensor.float()


# ---------------------------------------------------------------------
# Save processed image
# ---------------------------------------------------------------------

def save_processed_image(image, output_path):
    """
    Save a processed 3-channel MRI representation as a PNG image.

    Parameters
    ----------
    image : np.ndarray
        Processed image of shape [3, H, W].

    output_path : str or pathlib.Path
        Output PNG path.
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            "Expected a 3-channel image with shape [3, H, W]."
        )

    # Since all three channels contain the same grayscale information,
    # save the first channel as the visual representation.
    grayscale = np.clip(
        image[0] * 255.0,
        0,
        255
    ).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        grayscale,
        mode="L"
    ).save(output_path)


# ---------------------------------------------------------------------
# Simple command-line test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("ABIDE-I preprocessing module")
    print("--------------------------------")
    print(f"Target image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Output channels: 3")
    print("--------------------------------")
    print(
        "Use preprocess_nifti(path) to preprocess "
        "an individual ABIDE-I NIfTI file."
    )
