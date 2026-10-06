"""
ABIDE-I structural MRI dataset loader.

The dataset contains preprocessed 2D structural MRI images.
Each image is represented as a 224 x 224 grayscale image and
converted to three channels for the ResNet18 feature extractor.

Labels:
    0 -> TD
    1 -> ASD

Pipeline:
    224 x 224 MRI image
        -> 3-channel representation
        -> 49 non-overlapping 32 x 32 patches

The dataset class does not perform:
    - NIfTI loading
    - CSV loading
    - population graph construction
    - GCN processing
    - Transformer processing
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.create_patches import (
    extract_patches_from_rgb,
)


IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = 49


class ABIDEDataset(Dataset):
    """
    Dataset loader for preprocessed ABIDE-I sMRI images.

    Expected directory structure
    ----------------------------
    root_dir/
        ASD/
            image_001.png
            image_002.png
            ...
        TD/
            image_003.png
            image_004.png
            ...

    Labels
    ------
    0 : TD
    1 : ASD
    """

    VALID_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    CLASS_LABELS = {
        "TD": 0,
        "ASD": 1,
    }

    def __init__(self, root_dir):
        """
        Parameters
        ----------
        root_dir : str or pathlib.Path
            Root directory containing ASD and TD folders.
        """

        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        if not self.root_dir.is_dir():
            raise ValueError(
                f"Dataset path is not a directory: {self.root_dir}"
            )

        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid MRI images were found in: {self.root_dir}"
            )

    def _collect_samples(self):
        """
        Collect image paths and corresponding labels.
        """

        for class_name, label in self.CLASS_LABELS.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            if not class_dir.is_dir():
                raise ValueError(
                    f"Class path is not a directory: {class_dir}"
                )

            image_files = sorted(
                [
                    path
                    for path in class_dir.iterdir()
                    if (
                        path.is_file()
                        and path.suffix.lower()
                        in self.VALID_EXTENSIONS
                    )
                ]
            )

            for image_path in image_files:
                self.samples.append(
                    (
                        image_path,
                        label,
                    )
                )

    def __len__(self):
        """
        Return the number of subjects.
        """

        return len(self.samples)

    def _load_image(self, image_path):
        """
        Load one preprocessed 2D MRI image.

        Parameters
        ----------
        image_path : pathlib.Path
            Path to the MRI image.

        Returns
        -------
        torch.Tensor
            Tensor with shape [3, 224, 224].
        """

        try:
            image = Image.open(image_path).convert("L")
        except Exception as error:
            raise RuntimeError(
                f"Unable to load MRI image: {image_path}"
            ) from error

        # Ensure the required spatial resolution.
        if image.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                resample=Image.Resampling.BILINEAR,
            )

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        # Convert pixel intensities to [0, 1].
        image = image / 255.0

        # Convert grayscale image to three identical channels.
        image = np.stack(
            [
                image,
                image,
                image,
            ],
            axis=0,
        )

        image_tensor = torch.from_numpy(
            image.copy()
        ).float()

        return image_tensor

    def __getitem__(self, index):
        """
        Load one subject and extract its patches.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        dict
            image:
                Tensor of shape [3, 224, 224]

            patches:
                Tensor of shape [49, 3, 32, 32]

            label:
                Binary class label

            image_path:
                Source image path
        """

        if not isinstance(index, int):
            index = int(index)

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        image_path, label = self.samples[index]

        image_tensor = self._load_image(
            image_path
        )

        patches = extract_patches_from_rgb(
            image_tensor,
            patch_size=PATCH_SIZE,
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "patches": patches,
            "label": label_tensor,
            "image_path": str(image_path),
        }

    def get_labels(self):
        """
        Return labels for all subjects.

        Returns
        -------
        np.ndarray
            Integer labels with shape [N].
        """

        return np.asarray(
            [
                label
                for _, label in self.samples
            ],
            dtype=np.int64,
        )

    def get_image_paths(self):
        """
        Return image paths for all subjects.

        Returns
        -------
        list
            List of image paths.
        """

        return [
            image_path
            for image_path, _ in self.samples
        ]


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Dataset test
    # ---------------------------------------------------------------

    # Change this path to the actual location of your ABIDE-I
    # preprocessed 2D MRI images before running this test.
    dataset_root = "E:/ABIDE1"

    try:

        dataset = ABIDEDataset(
            root_dir=dataset_root
        )

        print(
            "Number of subjects:",
            len(dataset)
        )

        labels = dataset.get_labels()

        print(
            "TD subjects:",
            int(np.sum(labels == 0))
        )

        print(
            "ASD subjects:",
            int(np.sum(labels == 1))
        )

        sample = dataset[0]

        print(
            "Image shape:",
            tuple(sample["image"].shape)
        )

        print(
            "Patch shape:",
            tuple(sample["patches"].shape)
        )

        print(
            "Label:",
            int(sample["label"].item())
        )

        print(
            "Image path:",
            sample["image_path"]
        )

        assert sample["image"].shape == (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        assert sample["patches"].shape == (
            NUM_PATCHES,
            3,
            PATCH_SIZE,
            PATCH_SIZE,
        )

        assert sample["label"].item() in (0.0, 1.0)

        print("ABIDE-I dataset test passed.")

    except FileNotFoundError as error:

        print(error)

        print(
            "Update dataset_root to the actual ABIDE-I "
            "image directory before running the test."
        )
