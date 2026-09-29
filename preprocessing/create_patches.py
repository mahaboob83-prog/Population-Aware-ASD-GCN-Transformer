"""
Non-overlapping patch extraction for structural MRI slices.

The proposed framework represents each 224 x 224 structural MRI
slice using 32 x 32 non-overlapping patches, resulting in a
7 x 7 grid and 49 patches per subject.
"""

import torch


def extract_patches(image, patch_size=32):
    """
    Divide a 224 x 224 MRI image into non-overlapping patches.

    Parameters
    ----------
    image : torch.Tensor
        MRI image with shape:
        [C, H, W]

    patch_size : int
        Spatial size of each square patch.

    Returns
    -------
    torch.Tensor
        Patch tensor with shape:
        [num_patches, C, patch_size, patch_size]

    Notes
    -----
    For a 224 x 224 image and a patch size of 32,
    the resulting grid is 7 x 7 and contains 49 patches.
    """

    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [C, H, W]."
        )

    channels, height, width = image.shape

    if height != 224 or width != 224:
        raise ValueError(
            f"Expected a 224 x 224 image, "
            f"but received {height} x {width}."
        )

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "Image dimensions must be divisible by patch_size."
        )

    patches = image.unfold(
        dimension=1,
        size=patch_size,
        step=patch_size
    ).unfold(
        dimension=2,
        size=patch_size,
        step=patch_size
    )

    # Current shape:
    # [C, grid_rows, grid_cols, patch_size, patch_size]

    patches = patches.permute(
        1, 2, 0, 3, 4
    )

    # Shape:
    # [grid_rows, grid_cols, C, patch_size, patch_size]

    grid_rows = patches.shape[0]
    grid_cols = patches.shape[1]

    patches = patches.reshape(
        grid_rows * grid_cols,
        channels,
        patch_size,
        patch_size
    )

    return patches


if __name__ == "__main__":

    # Example shape test
    dummy_image = torch.randn(1, 224, 224)

    patches = extract_patches(
        dummy_image,
        patch_size=32
    )

    print("Input shape :", dummy_image.shape)
    print("Patch shape :", patches.shape)
    print("Number of patches:", patches.shape[0])
