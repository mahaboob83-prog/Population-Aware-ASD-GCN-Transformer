"""
Graph Convolutional Network for population-level representation learning.

The module implements two graph convolution layers using the
symmetrically normalized adjacency matrix:

    A_tilde = A + I

    A_hat = D^(-1/2) A_tilde D^(-1/2)

The input node representation is 128-dimensional.
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

        aggregated = torch.matmul(
            normalized_adjacency,
            x
        )

        output = self.linear(
            aggregated
        )

        return output


class PopulationGCN(nn.Module):
    """
    Two-layer GCN for population-level relational learning.

    Input:
        N x 128 subject-level representations

    Output:
        N x hidden_dim graph-refined representations
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        dropout=0.0
    ):
        super().__init__()

        self.gcn1 = GraphConvolution(
            input_dim,
            hidden_dim
        )

        self.gcn2 = GraphConvolution(
            hidden_dim,
            output_dim
        )

        self.activation = nn.ReLU()

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
            Normalized population graph:
            [N, N]

        Returns
        -------
        torch.Tensor
            Graph-refined node representations:
            [N, output_dim]
        """

        x = self.gcn1(
            x,
            normalized_adjacency
        )

        x = self.activation(x)

        x = self.dropout(x)

        x = self.gcn2(
            x,
            normalized_adjacency
        )

        x = self.activation(x)

        return x


if __name__ == "__main__":

    # Shape test
    num_subjects = 20
    feature_dim = 128

    dummy_features = torch.randn(
        num_subjects,
        feature_dim
    )

    dummy_adjacency = torch.rand(
        num_subjects,
        num_subjects
    )

    # Symmetrize the example adjacency.
    dummy_adjacency = (
        dummy_adjacency
        + dummy_adjacency.t()
    ) / 2.0

    identity = torch.eye(
        num_subjects
    )

    adjacency_with_self_loops = (
        dummy_adjacency
        + identity
    )

    degree = (
        adjacency_with_self_loops
        .sum(dim=1)
    )

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5
    )

    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(1)
        * adjacency_with_self_loops
        * degree_inv_sqrt.unsqueeze(0)
    )

    model = PopulationGCN(
        input_dim=128,
        hidden_dim=128,
        output_dim=128
    )

    output = model(
        dummy_features,
        normalized_adjacency
    )

    print(
        "Input feature shape:",
        dummy_features.shape
    )

    print(
        "Output feature shape:",
        output.shape
    )
