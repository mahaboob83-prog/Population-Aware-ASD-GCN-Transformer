"""
MLP classifier for ASD versus typically developing (TD) classification.

The classifier receives the 128-dimensional representations produced
by the population-level representation learning module and produces
two class logits corresponding to ASD and TD.
"""

import torch
import torch.nn as nn


class ASDClassifier(nn.Module):
    """
    Multilayer perceptron classifier.

    Input:
        [N, 128]

    Output:
        [N, 2]

    Class indices:
        0 -> TD
        1 -> ASD
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=64,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

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
            Class logits:
            [N, 2]
        """

        return self.classifier(x)


if __name__ == "__main__":

    # Simple shape test.
    num_subjects = 20
    embedding_dim = 128

    dummy_input = torch.randn(
        num_subjects,
        embedding_dim
    )

    model = ASDClassifier(
        input_dim=128,
        hidden_dim=64,
        num_classes=2,
        dropout=0.3
    )

    logits = model(
        dummy_input
    )

    print(
        "Input shape :",
        dummy_input.shape
    )

    print(
        "Output shape:",
        logits.shape
    )

    print(
        "Expected output:",
        "[N, 2]"
    )
