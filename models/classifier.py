"""
MLP classifier for ASD versus typically developing (TD) classification.

The classifier follows the BMC Medical Imaging manuscript:
    128 -> 64 -> 1
with ReLU, dropout, and sigmoid output.
"""

import torch
import torch.nn as nn


class ASDClassifier(nn.Module):
    """
    Two-layer MLP classifier.

    Input:
        [N, 128]

    Output:
        [N, 1]

    Output represents the probability of ASD.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=64,
        dropout=0.3
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input representation:
            [N, 128]

        Returns
        -------
        torch.Tensor
            ASD probability:
            [N, 1]
        """

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x


if __name__ == "__main__":

    num_subjects = 20

    dummy_input = torch.randn(
        num_subjects,
        128
    )

    model = ASDClassifier(
        input_dim=128,
        hidden_dim=64,
        dropout=0.3
    )

    probability = model(
        dummy_input
    )

    print(
        "Input shape:",
        dummy_input.shape
    )

    print(
        "Output shape:",
        probability.shape
    )

    print(
        "Minimum probability:",
        probability.min().item()
    )

    print(
        "Maximum probability:",
        probability.max().item()
    )
