"""
Patch extraction utilities for 2D structural MRI slices.

The proposed framework uses a 224x224 MRI representation divided into
49 non-overlapping 32x32 patches arranged in a 7x7 grid.
"""

import torch


IMAGE_SIZE = 224
PATCH_SIZE = 32
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE


def extract_patches(image, patch_size=PATCH_SIZE):
    """
    Divide a 2D MRI image into non-overlapping square patches.

    Parameters
    ----------
    image : torch.Tensor
        2D image tensor of shape [H, W].

    patch_size : int, optional
        Size of each square patch. Default is 32.

    Returns
    -------
    torch.Tensor
        Patch tensor of shape [N, patch_size, patch_size],
        where N is the number of extracted patches.

    Raises
    ------
    TypeError
        If image is not a PyTorch tensor.

    ValueError
        If the input is not a 2D image or its dimensions are not
        divisible by the patch size.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image tensor [H, W], got shape {tuple(image.shape)}."
        )

    height, width = image.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image dimensions ({height}, {width}) must be divisible "
            f"by patch size ({patch_size})."
        )

    patches = image.unfold(0, patch_size, patch_size)
    patches = patches.unfold(1, patch_size, patch_size)

    # [grid_h, grid_w, patch_h, patch_w]
    patches = patches.contiguous().view(
        -1, patch_size, patch_size
    )

    return patches


def extract_patches_from_rgb(image, patch_size=PATCH_SIZE):
    """
    Divide a 3-channel MRI image into non-overlapping patches.

    Parameters
    ----------
    image : torch.Tensor
        RGB image tensor of shape [C, H, W].

    patch_size : int, optional
        Size of each square patch. Default is 32.

    Returns
    -------
    torch.Tensor
        Patch tensor of shape [N, C, patch_size, patch_size].
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")

    if image.ndim != 3:
        raise ValueError(
            f"Expected an RGB tensor [C, H, W], got shape {tuple(image.shape)}."
        )

    channels, height, width = image.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image dimensions ({height}, {width}) must be divisible "
            f"by patch size ({patch_size})."
        )

    patches = image.unfold(1, patch_size, patch_size)
    patches = patches.unfold(2, patch_size, patch_size)

    # [C, grid_h, grid_w, patch_h, patch_w]
    patches = patches.permute(1, 2, 0, 3, 4)

    # [N, C, patch_h, patch_w]
    patches = patches.contiguous().view(
        -1, channels, patch_size, patch_size
    )

    return patches


def get_patch_grid(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """
    Return the number of patches along each image dimension.

    Returns
    -------
    tuple
        (rows, columns, total_patches)
    """

    if image_size % patch_size != 0:
        raise ValueError(
            f"Image size ({image_size}) must be divisible "
            f"by patch size ({patch_size})."
        )

    rows = image_size // patch_size
    columns = image_size // patch_size
    total = rows * columns

    return rows, columns, total


if __name__ == "__main__":
    # Test with a single 224x224 grayscale MRI slice.
    image = torch.rand(IMAGE_SIZE, IMAGE_SIZE)

    patches = extract_patches(image)

    rows, columns, total = get_patch_grid()

    print(f"Image shape:        {tuple(image.shape)}")
    print(f"Patch grid:         {rows} x {columns}")
    print(f"Number of patches:  {total}")
    print(f"Patch tensor shape: {tuple(patches.shape)}")

    assert patches.shape == (
        NUM_PATCHES,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    # Test with a 3-channel image.
    rgb_image = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)

    rgb_patches = extract_patches_from_rgb(rgb_image)

    print(f"RGB image shape:    {tuple(rgb_image.shape)}")
    print(f"RGB patches shape:  {tuple(rgb_patches.shape)}")

    assert rgb_patches.shape == (
        NUM_PATCHES,
        3,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    print("Patch extraction tests passed.")
