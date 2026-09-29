"""
Transformer encoder for population-level contextual representation.

The Transformer operates on the graph-refined subject representations
produced by the GCN module and uses multi-head self-attention to model
higher-order and long-range interactions.
"""

import torch
import torch.nn as nn


class PopulationTransformer(nn.Module):
    """
    Transformer encoder for graph-refined subject representations.

    Input:
        [N, embedding_dim]

    Output:
        [N, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.3
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Graph-refined node representations:
            [N, embedding_dim]

        Returns
        -------
        torch.Tensor
            Transformer-refined representations:
            [N, embedding_dim]
        """

        # Treat the population as one sequence.
        x = x.unsqueeze(0)

        x = self.encoder(x)

        # Remove the artificial batch dimension.
        x = x.squeeze(0)

        return x


if __name__ == "__main__":

    # Simple shape test.
    num_subjects = 20
    embedding_dim = 128

    dummy_input = torch.randn(
        num_subjects,
        embedding_dim
    )

    model = PopulationTransformer(
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.3
    )

    output = model(
        dummy_input
    )

    print(
        "Input shape :",
        dummy_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )
