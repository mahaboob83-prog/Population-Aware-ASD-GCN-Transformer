"""
Graph Convolutional Network for population-level representation learning.

The module implements a two-layer GCN using a symmetrically
normalized population adjacency matrix:

    A_tilde = A + I

    A_hat = D^(-1/2) A_tilde D^(-1/2)

The input and output node representations are 128-dimensional.
"""

import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Single graph convolution layer.

    Parameters
    ----------
    in_features : int
        Input node feature dimension.

    out_features : int
        Output node feature dimension.
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x, normalized_adjacency):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix:
            [N, in_features]

        normalized_adjacency : torch.Tensor
            Symmetrically normalized adjacency matrix:
            [N, N]

        Returns
        -------
        torch.Tensor
            Updated node representations:
            [N, out_features]
        """

        # Aggregate information from neighboring nodes.
        aggregated = torch.matmul(
            normalized_adjacency,
            x
        )

        # Learnable linear transformation.
        output = self.linear(
            aggregated
        )

        return output


class PopulationGCN(nn.Module):
    """
    Two-layer GCN for population-level relational learning.

    Architecture
    ------------
    Input:
        N x 128 subject-level representations

    GCN Layer 1:
        128 -> 128

    ReLU + Dropout:
        Dropout = 0.3

    GCN Layer 2:
        128 -> 128

    Output:
        N x 128 graph-refined representations
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        dropout=0.3
    ):
        super().__init__()

        # First graph convolution layer.
        self.gcn1 = GraphConvolution(
            input_dim,
            hidden_dim
        )

        # Second graph convolution layer.
        self.gcn2 = GraphConvolution(
            hidden_dim,
            output_dim
        )

        # Non-linear activation.
        self.activation = nn.ReLU()

        # Dropout specified for the GCN module.
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        normalized_adjacency
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Subject-level node features:
            [N, 128]

        normalized_adjacency : torch.Tensor
            Symmetrically normalized population graph:
            [N, N]

        Returns
        -------
        torch.Tensor
            Graph-refined node representations:
            [N, output_dim]
        """

        # First GCN layer.
        x = self.gcn1(
            x,
            normalized_adjacency
        )

        # Non-linear transformation.
        x = self.activation(x)

        # Dropout after the first GCN layer.
        x = self.dropout(x)

        # Second GCN layer.
        x = self.gcn2(
            x,
            normalized_adjacency
        )

        # The second GCN output is returned directly,
        # following the manuscript formulation.
        return x


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Simple shape test
    # ---------------------------------------------------------

    num_subjects = 20
    feature_dim = 128

    # Dummy subject-level features.
    dummy_features = torch.randn(
        num_subjects,
        feature_dim
    )

    # Dummy symmetric adjacency matrix.
    dummy_adjacency = torch.rand(
        num_subjects,
        num_subjects
    )

    dummy_adjacency = (
        dummy_adjacency
        + dummy_adjacency.t()
    ) / 2.0

    # Add self-loops.
    identity = torch.eye(
        num_subjects
    )

    adjacency_with_self_loops = (
        dummy_adjacency
        + identity
    )

    # Compute node degrees.
    degree = (
        adjacency_with_self_loops
        .sum(dim=1)
    )

    # Compute D^(-1/2).
    degree_inv_sqrt = torch.pow(
        degree,
        -0.5
    )

    degree_inv_sqrt[
        torch.isinf(degree_inv_sqrt)
    ] = 0.0

    # Symmetric normalized adjacency:
    # D^(-1/2) A D^(-1/2)
    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(1)
        * adjacency_with_self_loops
        * degree_inv_sqrt.unsqueeze(0)
    )

    # Create the GCN.
    model = PopulationGCN(
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        dropout=0.3
    )

    # Forward pass.
    output = model(
        dummy_features,
        normalized_adjacency
    )

    print(
        "Input feature shape:",
        dummy_features.shape
    )

    print(
        "Normalized adjacency shape:",
        normalized_adjacency.shape
    )

    print(
        "Output feature shape:",
        output.shape
    )
