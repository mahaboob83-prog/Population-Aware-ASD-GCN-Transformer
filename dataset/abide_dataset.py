"""
ABIDE-I dataset loader for structural MRI.

Each subject is loaded from a NIfTI file and processed using the
preprocessing pipeline defined in preprocessing/prepare_abide.py.

Pipeline:

    NIfTI MRI
        -> preprocessing
        -> 3 x 224 x 224 image
        -> 49 non-overlapping patches
        -> 49 x 3 x 32 x 32 patches

The dataset class does not perform model inference, graph
construction, GCN processing, or Transformer processing.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.create_patches import (
    extract_patches_from_rgb,
)
from preprocessing.prepare_abide import (
    image_to_tensor,
    preprocess_nifti,
)


class ABIDEDataset(Dataset):
    """
    PyTorch dataset for ABIDE-I structural MRI subjects.

    Expected metadata columns
    -------------------------
    image_path : path to the subject's NIfTI file
    label      : binary class label

    Labels
    ------
    0 : TD
    1 : ASD

    Returns
    -------
    dict
        image:
            Tensor of shape [3, 224, 224]

        patches:
            Tensor of shape [49, 3, 32, 32]

        label:
            Tensor containing the binary class label

        image_path:
            Original MRI path
    """

    def __init__(
        self,
        dataframe,
        image_column="image_path",
        label_column="label",
        apply_skull_stripping=True,
        crop=True,
    ):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing MRI paths and labels.

        image_column : str
            Name of the column containing NIfTI paths.

        label_column : str
            Name of the column containing binary labels.

        apply_skull_stripping : bool
            Whether to use the preprocessing module's
            skull-stripping step.

        crop : bool
            Whether to crop background before resizing.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas.DataFrame."
            )

        required_columns = {
            image_column,
            label_column,
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if len(dataframe) == 0:
            raise ValueError(
                "The dataset dataframe is empty."
            )

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.image_column = image_column
        self.label_column = label_column
        self.apply_skull_stripping = (
            apply_skull_stripping
        )
        self.crop = crop

    def __len__(self):
        """
        Return the number of subjects.
        """

        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Load and preprocess one subject.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        dict
            Processed subject information.
        """

        if not isinstance(index, int):
            index = int(index)

        row = self.dataframe.iloc[index]

        image_path = Path(
            str(row[self.image_column])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"MRI file not found: {image_path}"
            )

        label = int(row[self.label_column])

        if label not in (0, 1):
            raise ValueError(
                f"Expected binary label 0 or 1, got {label}."
            )

        # -----------------------------------------------------------
        # MRI preprocessing
        # -----------------------------------------------------------

        image = preprocess_nifti(
            image_path,
            apply_skull_stripping=(
                self.apply_skull_stripping
            ),
            crop=self.crop,
        )

        # -----------------------------------------------------------
        # Convert image to PyTorch tensor
        # -----------------------------------------------------------

        image_tensor = image_to_tensor(
            image
        )

        # -----------------------------------------------------------
        # Extract 49 non-overlapping 32x32 patches
        # -----------------------------------------------------------

        patches = extract_patches_from_rgb(
            image_tensor
        )

        # -----------------------------------------------------------
        # Subject label
        # -----------------------------------------------------------

        label_tensor = torch.tensor(
            label,
            dtype=torch.float32
        )

        return {
            "image": image_tensor,
            "patches": patches,
            "label": label_tensor,
            "image_path": str(image_path),
        }


def create_dataset_from_csv(
    csv_path,
    image_column="image_path",
    label_column="label",
    apply_skull_stripping=True,
    crop=True,
):
    """
    Create an ABIDEDataset from a CSV file.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the dataset metadata CSV.

    image_column : str
        Column containing NIfTI paths.

    label_column : str
        Column containing binary labels.

    apply_skull_stripping : bool
        Whether to apply the preprocessing module's
        skull-stripping step.

    crop : bool
        Whether to crop background before resizing.

    Returns
    -------
    ABIDEDataset
        Configured ABIDE-I dataset.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path
    )

    return ABIDEDataset(
        dataframe=dataframe,
        image_column=image_column,
        label_column=label_column,
        apply_skull_stripping=apply_skull_stripping,
        crop=crop,
    )


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Dataset structure test
    # ---------------------------------------------------------------
    #
    # This test checks the expected dataframe interface without
    # requiring the ABIDE-I dataset to be included in the repository.
    # ---------------------------------------------------------------

    test_dataframe = pd.DataFrame(
        {
            "image_path": [
                "subject_001.nii.gz",
                "subject_002.nii.gz",
            ],
            "label": [
                0,
                1,
            ],
        }
    )

    dataset = ABIDEDataset(
        dataframe=test_dataframe,
        image_column="image_path",
        label_column="label",
        apply_skull_stripping=True,
        crop=True,
    )

    print(
        "Number of subjects:",
        len(dataset)
    )

    print(
        "Image column:",
        dataset.image_column
    )

    print(
        "Label column:",
        dataset.label_column
    )

    assert len(dataset) == 2

    assert dataset.image_column == "image_path"

    assert dataset.label_column == "label"

    print("ABIDE-I dataset structure test passed.")
