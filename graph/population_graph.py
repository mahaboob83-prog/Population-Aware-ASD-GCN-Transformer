"""
Population graph construction for subject-level MRI representations.

Each subject is represented by a subject-level embedding. Pairwise
cosine similarities between subjects are used to construct a weighted
population graph. Positive cosine similarities are retained as
continuous edge weights.

Self-loops are added and the adjacency matrix is symmetrically
normalized before being supplied to the population GCN.
"""

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(x):
    """
    Compute pairwise cosine similarity between subject embeddings.

    Parameters
    ----------
    x : torch.Tensor
        Subject-level embeddings of shape [N, D].

    Returns
    -------
    torch.Tensor
        Pairwise cosine similarity matrix of shape [N, N].
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")

    if x.ndim != 2:
        raise ValueError(
            f"Expected x with shape [N, D], got {tuple(x.shape)}."
        )

    # L2-normalize each subject embedding.
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
    Construct the weighted population adjacency matrix.

    Positive cosine similarities are retained as continuous edge
    weights. Non-positive similarities are set to zero.

    Parameters
    ----------
    x : torch.Tensor
        Subject-level embeddings of shape [N, D].

    Returns
    -------
    torch.Tensor
        Weighted adjacency matrix of shape [N, N].
    """

    similarity = cosine_similarity_matrix(x)

    # Retain only positive cosine similarities.
    adjacency = torch.clamp(
        similarity,
        min=0.0
    )

    # Enforce numerical symmetry.
    adjacency = 0.5 * (
        adjacency + adjacency.transpose(0, 1)
    )

    return adjacency


def add_self_loops(adjacency):
    """
    Add self-loops to the population graph.

    Parameters
    ----------
    adjacency : torch.Tensor
        Weighted adjacency matrix of shape [N, N].

    Returns
    -------
    torch.Tensor
        Adjacency matrix with self-loops.
    """

    if not isinstance(adjacency, torch.Tensor):
        raise TypeError(
            "adjacency must be a torch.Tensor."
        )

    if adjacency.ndim != 2:
        raise ValueError(
            "Adjacency matrix must be two-dimensional."
        )

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "Adjacency matrix must be square."
        )

    num_nodes = adjacency.shape[0]

    identity = torch.eye(
        num_nodes,
        dtype=adjacency.dtype,
        device=adjacency.device
    )

    return adjacency + identity


def normalize_adjacency(adjacency):
    """
    Apply symmetric adjacency normalization.

    The normalization is:

        A_hat = D_tilde^(-1/2)
                (A + I)
                D_tilde^(-1/2)

    Parameters
    ----------
    adjacency : torch.Tensor
        Weighted adjacency matrix of shape [N, N].

    Returns
    -------
    torch.Tensor
        Symmetrically normalized adjacency matrix.
    """

    adjacency_with_loops = add_self_loops(
        adjacency
    )

    degree = adjacency_with_loops.sum(
        dim=1
    )

    # Numerical stability for non-zero degree values.
    degree_inv_sqrt = torch.pow(
        degree.clamp_min(1e-12),
        -0.5
    )

    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(1)
        * adjacency_with_loops
        * degree_inv_sqrt.unsqueeze(0)
    )

    return normalized_adjacency


def build_population_graph(x):
    """
    Construct the complete weighted population graph.

    Parameters
    ----------
    x : torch.Tensor
        Subject-level embeddings of shape [N, D].

    Returns
    -------
    adjacency : torch.Tensor
        Weighted population adjacency matrix.

    normalized_adjacency : torch.Tensor
        Symmetrically normalized adjacency matrix for GCN propagation.
    """

    adjacency = build_adaptive_adjacency(x)

    normalized_adjacency = normalize_adjacency(
        adjacency
    )

    return adjacency, normalized_adjacency


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Basic graph construction test
    # ---------------------------------------------------------------

    num_subjects = 20
    embedding_dim = 128

    subject_embeddings = torch.randn(
        num_subjects,
        embedding_dim
    )

    adjacency, normalized_adjacency = build_population_graph(
        subject_embeddings
    )

    print(
        "Subject embedding shape:   ",
        tuple(subject_embeddings.shape)
    )

    print(
        "Adjacency shape:            ",
        tuple(adjacency.shape)
    )

    print(
        "Normalized adjacency shape: ",
        tuple(normalized_adjacency.shape)
    )

    # ---------------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------------

    assert adjacency.shape == (
        num_subjects,
        num_subjects
    )

    assert normalized_adjacency.shape == (
        num_subjects,
        num_subjects
    )

    # ---------------------------------------------------------------
    # Symmetry checks
    # ---------------------------------------------------------------

    assert torch.allclose(
        adjacency,
        adjacency.transpose(0, 1),
        atol=1e-6
    )

    assert torch.allclose(
        normalized_adjacency,
        normalized_adjacency.transpose(0, 1),
        atol=1e-6
    )

    # ---------------------------------------------------------------
    # Non-negative edge-weight check
    # ---------------------------------------------------------------

    assert torch.all(
        adjacency >= 0
    )

    # ---------------------------------------------------------------
    # Diagonal check
    # ---------------------------------------------------------------

    assert torch.all(
        adjacency.diagonal() > 0
    )

    print("Population graph construction test passed.")
