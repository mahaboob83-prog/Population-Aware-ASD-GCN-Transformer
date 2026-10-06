"""
Transformer encoder for population-level contextual representation learning.

The Transformer operates on the graph-refined subject representations
produced by the GCN module.

The implementation follows the revised manuscript:

    Input representation : 128-dimensional
    Transformer layers    : 2
    Attention heads       : 4
    Feed-forward dimension: 256
    Dropout               : 0.1
    Positional embedding  : Learnable

Input:
    [N, 128]

Output:
    [N, 128]
"""

import torch
import torch.nn as nn


class PopulationTransformer(nn.Module):
    """
    Transformer encoder for graph-refined subject representations.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the subject representation.

    num_heads : int
        Number of self-attention heads.

    num_layers : int
        Number of Transformer encoder layers.

    feedforward_dim : int
        Dimensionality of the Transformer feed-forward network.

    dropout : float
        Dropout probability used in the Transformer encoder.

    max_nodes : int
        Maximum number of subjects that can be processed in one
        population sequence.

    Architecture
    ------------
    Input:
        N x 128

    Positional embedding:
        N x 128

    Transformer encoder:
        2 layers
        4 attention heads
        Feed-forward dimension = 256

    Output:
        N x 128
    """

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_nodes=1112
    ):
        super().__init__()

        # -----------------------------------------------------
        # Validate configuration
        # -----------------------------------------------------

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if max_nodes <= 0:
            raise ValueError(
                "max_nodes must be greater than zero."
            )

        # -----------------------------------------------------
        # Learnable positional embeddings
        # -----------------------------------------------------

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_nodes,
                embedding_dim
            )
        )

        # -----------------------------------------------------
        # Transformer encoder layer
        # -----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        # -----------------------------------------------------
        # Transformer encoder
        # -----------------------------------------------------

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Store configuration for reproducibility.
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout
        self.max_nodes = max_nodes

    def forward(self, x):
        """
        Apply positional embeddings and Transformer encoding.

        Parameters
        ----------
        x : torch.Tensor
            Graph-refined subject representations with shape:

                [N, embedding_dim]

            where N is the number of subjects.

        Returns
        -------
        torch.Tensor
            Transformer-refined subject representations with shape:

                [N, embedding_dim]
        """

        # -----------------------------------------------------
        # Validate input
        # -----------------------------------------------------

        if x.ndim != 2:
            raise ValueError(
                "Expected input with shape [N, embedding_dim]."
            )

        num_nodes, feature_dim = x.shape

        if feature_dim != self.embedding_dim:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.embedding_dim}, "
                f"but received {feature_dim}."
            )

        if num_nodes > self.max_nodes:
            raise ValueError(
                f"Number of nodes ({num_nodes}) exceeds "
                f"the configured maximum ({self.max_nodes})."
            )

        # -----------------------------------------------------
        # Add batch dimension
        #
        # Transformer input:
        # [batch_size, sequence_length, embedding_dim]
        #
        # Here, the population is treated as one sequence.
        # -----------------------------------------------------

        x = x.unsqueeze(0)

        # -----------------------------------------------------
        # Add learnable positional embeddings
        # -----------------------------------------------------

        x = x + self.positional_embedding[
            :, :num_nodes, :
        ]

        # -----------------------------------------------------
        # Transformer encoding
        # -----------------------------------------------------

        x = self.encoder(x)

        # -----------------------------------------------------
        # Remove artificial batch dimension
        # -----------------------------------------------------

        x = x.squeeze(0)

        return x


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Simple shape test
    # ---------------------------------------------------------

    num_subjects = 20
    embedding_dim = 128

    # Dummy graph-refined subject representations.
    dummy_input = torch.randn(
        num_subjects,
        embedding_dim
    )

    # Create Transformer.
    model = PopulationTransformer(
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_nodes=1112
    )

    # Forward pass.
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

    print(
        "Number of attention heads:",
        model.num_heads
    )

    print(
        "Number of Transformer layers:",
        model.num_layers
    )

    print(
        "Feed-forward dimension:",
        model.feedforward_dim
    )

    print(
        "Maximum population size:",
        model.max_nodes
    )
