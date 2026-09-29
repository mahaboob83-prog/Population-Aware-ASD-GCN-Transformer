"""
Adaptive similarity-based population graph construction.

The graph is constructed from subject-level 128-dimensional
embeddings using pairwise cosine similarity.

The implementation follows the revised manuscript description:

    subject embeddings
            ↓
    cosine similarity
            ↓
    continuous similarity-based edge weights
            ↓
    self-loops
            ↓
    symmetric normalized adjacency matrix
"""

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(x):
    """
    Compute pairwise cosine similarity between subject embeddings.

    Parameters
    ----------
    x : torch.Tensor
        Subject feature matrix with shape:
        [N, feature_dim]

    Returns
    -------
    torch.Tensor
        Pairwise similarity matrix with shape:
        [N, N]
    """

    if x.ndim != 2:
        raise ValueError(
            "Expected input with shape [N, feature_dim]."
        )

    # Normalize each subject embedding.
    x_normalized = F.normalize(
        x,
        p=2,
        dim=1
    )

    # Pairwise cosine similarity.
    similarity = torch.mm(
        x_normalized,
        x_normalized.t()
    )

    return similarity


def build_adaptive_adjacency(x):
    """
    Construct a continuous similarity-weighted population graph.

    Negative cosine similarities are removed so that the adjacency
    weights lie in the range [0, 1], consistent with the graph
    formulation used in the revised manuscript and Fig. 6.

    Parameters
    ----------
    x : torch.Tensor
        Subject feature matrix:
        [N, feature_dim]

    Returns
    -------
    adjacency : torch.Tensor
        Weighted adjacency matrix:
        [N, N]

    similarity : torch.Tensor
        Original cosine similarity matrix:
        [N, N]
    """

    similarity = cosine_similarity_matrix(x)

    # Retain non-negative similarity values as continuous
    # edge weights.
    adjacency = torch.clamp(
        similarity,
        min=0.0,
        max=1.0
    )

    # Add self-loops.
    adjacency.fill_diagonal_(1.0)

    return adjacency, similarity


def normalize_adjacency(adjacency):
    """
    Compute the symmetric normalized adjacency matrix.

    A_tilde = A + I
    D_ii = sum_j A_tilde_ij

    A_hat = D^(-1/2) A_tilde D^(-1/2)

    Parameters
    ----------
    adjacency : torch.Tensor
        Weighted adjacency matrix [N, N].

    Returns
    -------
    torch.Tensor
        Symmetrically normalized adjacency matrix [N, N].
    """

    num_nodes = adjacency.size(0)

    identity = torch.eye(
        num_nodes,
        device=adjacency.device,
        dtype=adjacency.dtype
    )

    # Add self-loops.
    adjacency_with_self_loops = (
        adjacency + identity
    )

    degree = adjacency_with_self_loops.sum(
        dim=1
    )

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5
    )

    degree_inv_sqrt[
        torch.isinf(degree_inv_sqrt)
    ] = 0.0

    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(1)
        * adjacency_with_self_loops
        * degree_inv_sqrt.unsqueeze(0)
    )

    return normalized_adjacency


def build_population_graph(x):
    """
    Complete adaptive population graph pipeline.

    Parameters
    ----------
    x : torch.Tensor
        Subject-level feature matrix:
        [N, 128]

    Returns
    -------
    adjacency : torch.Tensor
        Continuous similarity-weighted adjacency matrix.

    normalized_adjacency : torch.Tensor
        Normalized adjacency matrix used by the GCN.
    """

    adjacency, _ = build_adaptive_adjacency(x)

    normalized_adjacency = normalize_adjacency(
        adjacency
    )

    return adjacency, normalized_adjacency


if __name__ == "__main__":

    # Simple shape test.
    dummy_features = torch.randn(
        20,
        128
    )

    adjacency, normalized_adjacency = (
        build_population_graph(dummy_features)
    )

    print(
        "Subject feature shape:",
        dummy_features.shape
    )

    print(
        "Adjacency shape:",
        adjacency.shape
    )

    print(
        "Normalized adjacency shape:",
        normalized_adjacency.shape
    )

    print(
        "Minimum adjacency value:",
        adjacency.min().item()
    )

    print(
        "Maximum adjacency value:",
        adjacency.max().item()
    )
