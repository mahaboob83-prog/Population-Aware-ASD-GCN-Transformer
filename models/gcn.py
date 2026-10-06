"""
Two-layer Graph Convolutional Network for population-level
structural representation learning.

The implementation follows the revised manuscript formulation:

    H^(0) = X

    H^(1) = ReLU(A_hat H^(0) W^(0))

    Z = A_hat H^(1) W^(1)

where:

    X       : subject-level structural representations
    A_hat   : symmetrically normalized population adjacency matrix
    H^(1)   : hidden graph representation
    Z       : graph-refined subject representation

The input and output representations are 128-dimensional.
"""

import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Single graph convolution layer.

    The layer first aggregates information from neighboring
    subjects using the normalized adjacency matrix and then
    applies a learnable linear transformation.

    Parameters
    ----------
    in_features : int
        Input node feature dimension.

    out_features : int
        Output node feature dimension.
    """

    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=False
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
            Node feature matrix with shape:

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

        # -----------------------------------------------------
        # Validate input dimensions
        # -----------------------------------------------------

        if x.ndim != 2:
            raise ValueError(
                "Expected node features with shape "
                "[N, in_features]."
            )

        if normalized_adjacency.ndim != 2:
            raise ValueError(
                "Expected normalized adjacency with shape "
                "[N, N]."
            )

        if normalized_adjacency.size(0) != x.size(0):
            raise ValueError(
                "The number of nodes in the adjacency matrix "
                "must match the number of nodes in x."
            )

        # -----------------------------------------------------
        # Graph neighborhood aggregation
        #
        # A_hat X
        # -----------------------------------------------------

        aggregated = torch.matmul(
            normalized_adjacency,
            x
        )

        # -----------------------------------------------------
        # Learnable transformation
        #
        # A_hat X W
        # -----------------------------------------------------

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
        N x 128

    First GCN layer:
        128 -> 128
        followed by ReLU

    Second GCN layer:
        128 -> 128

    Output:
        N x 128

    The second GCN layer is followed directly by the output,
    consistent with the mathematical formulation in the
    revised manuscript.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=128
    ):
        super().__init__()

        # -----------------------------------------------------
        # First GCN layer
        # -----------------------------------------------------

        self.gcn1 = GraphConvolution(
            in_features=input_dim,
            out_features=hidden_dim
        )

        # -----------------------------------------------------
        # Second GCN layer
        # -----------------------------------------------------

        self.gcn2 = GraphConvolution(
            in_features=hidden_dim,
            out_features=output_dim
        )

        # ReLU applied only after the first GCN layer.
        self.activation = nn.ReLU()

    def forward(
        self,
        x,
        normalized_adjacency
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Subject-level node representations:

                [N, 128]

        normalized_adjacency : torch.Tensor
            Symmetrically normalized population graph:

                [N, N]

        Returns
        -------
        torch.Tensor
            Graph-refined subject representations:

                [N, 128]
        """

        # -----------------------------------------------------
        # First GCN layer
        #
        # H^(1) = ReLU(A_hat H^(0) W^(0))
        # -----------------------------------------------------

        x = self.gcn1(
            x,
            normalized_adjacency
        )

        x = self.activation(x)

        # -----------------------------------------------------
        # Second GCN layer
        #
        # Z = A_hat H^(1) W^(1)
        # -----------------------------------------------------

        x = self.gcn2(
            x,
            normalized_adjacency
        )

        return x


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Simple shape test
    # ---------------------------------------------------------

    num_subjects = 20
    feature_dim = 128

    # Dummy subject-level representations.
    dummy_features = torch.randn(
        num_subjects,
        feature_dim
    )

    # ---------------------------------------------------------
    # Create a symmetric dummy adjacency matrix
    # ---------------------------------------------------------

    dummy_adjacency = torch.rand(
        num_subjects,
        num_subjects
    )

    dummy_adjacency = (
        dummy_adjacency
        + dummy_adjacency.t()
    ) / 2.0

    # ---------------------------------------------------------
    # Add self-loops
    # ---------------------------------------------------------

    identity = torch.eye(
        num_subjects
    )

    adjacency_with_self_loops = (
        dummy_adjacency
        + identity
    )

    # ---------------------------------------------------------
    # Compute node degrees
    # ---------------------------------------------------------

    degree = (
        adjacency_with_self_loops
        .sum(dim=1)
    )

    # ---------------------------------------------------------
    # Compute D^(-1/2)
    # ---------------------------------------------------------

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5
    )

    degree_inv_sqrt[
        torch.isinf(degree_inv_sqrt)
    ] = 0.0

    # ---------------------------------------------------------
    # Symmetric normalization
    #
    # A_hat = D^(-1/2) A_tilde D^(-1/2)
    # ---------------------------------------------------------

    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(1)
        * adjacency_with_self_loops
        * degree_inv_sqrt.unsqueeze(0)
    )

    # ---------------------------------------------------------
    # Create GCN
    # ---------------------------------------------------------

    model = PopulationGCN(
        input_dim=128,
        hidden_dim=128,
        output_dim=128
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

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
