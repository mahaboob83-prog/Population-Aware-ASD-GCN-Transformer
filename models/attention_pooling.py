"""
Attention-based patch aggregation for subject-level representation.

The module first projects each 512-dimensional ResNet18 patch
feature into a 128-dimensional embedding space. An attention
mechanism then assigns a normalized weight to each patch, and
the weighted patch embeddings are aggregated into a single
128-dimensional subject-level representation.
"""

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    """
    Project ResNet18 patch features from 512 dimensions to 128 dimensions.

    Input
    -----
    x : torch.Tensor
        Shape [B, N, 512]

    Output
    ------
    torch.Tensor
        Shape [B, N, 128]
    """

    def __init__(
        self,
        input_dim=512,
        embedding_dim=128
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            embedding_dim
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, N, D], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[-1] != self.projection.in_features:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.projection.in_features}, "
                f"got {x.shape[-1]}."
            )

        return self.projection(x)


class AttentionPooling(nn.Module):
    """
    Attention-based aggregation of patch embeddings.

    For each patch embedding h_i, an attention score is computed
    and normalized across all patches using softmax.

    Input
    -----
    x : torch.Tensor
        Shape [B, N, 128]

    Output
    ------
    subject_embedding : torch.Tensor
        Shape [B, 128]

    attention_weights : torch.Tensor
        Shape [B, N]
    """

    def __init__(
        self,
        embedding_dim=128,
        attention_hidden_dim=128
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                embedding_dim,
                attention_hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                attention_hidden_dim,
                1
            )
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, N, D], "
                f"got {tuple(x.shape)}."
            )

        # Compute one scalar attention score per patch.
        scores = self.attention(x).squeeze(-1)

        # Normalize scores across the patches of each subject.
        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # Weighted sum of patch embeddings.
        subject_embedding = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        return subject_embedding, attention_weights


class SubjectRepresentation(nn.Module):
    """
    Complete subject-level representation module.

    ResNet18 patch features:
        512-D

    Projected patch embeddings:
        128-D

    Subject-level representation:
        128-D
    """

    def __init__(
        self,
        input_dim=512,
        embedding_dim=128,
        attention_hidden_dim=128
    ):
        super().__init__()

        self.projection = PatchProjection(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )

        self.pooling = AttentionPooling(
            embedding_dim=embedding_dim,
            attention_hidden_dim=attention_hidden_dim
        )

    def forward(self, patch_features):
        """
        Parameters
        ----------
        patch_features : torch.Tensor
            Shape [B, N, 512]

        Returns
        -------
        subject_embedding : torch.Tensor
            Shape [B, 128]

        attention_weights : torch.Tensor
            Shape [B, N]
        """

        projected_features = self.projection(
            patch_features
        )

        subject_embedding, attention_weights = self.pooling(
            projected_features
        )

        return subject_embedding, attention_weights


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Test the attention-based subject representation.
    # ---------------------------------------------------------------

    batch_size = 4
    num_patches = 49
    patch_feature_dim = 512
    embedding_dim = 128

    patch_features = torch.randn(
        batch_size,
        num_patches,
        patch_feature_dim
    )

    model = SubjectRepresentation(
        input_dim=patch_feature_dim,
        embedding_dim=embedding_dim,
        attention_hidden_dim=128
    )

    subject_embedding, attention_weights = model(
        patch_features
    )

    print(
        "Input patch feature shape:     ",
        tuple(patch_features.shape)
    )

    print(
        "Subject embedding shape:       ",
        tuple(subject_embedding.shape)
    )

    print(
        "Attention weight shape:         ",
        tuple(attention_weights.shape)
    )

    print(
        "Attention weight sums:          ",
        attention_weights.sum(dim=1)
    )

    assert subject_embedding.shape == (
        batch_size,
        embedding_dim
    )

    assert attention_weights.shape == (
        batch_size,
        num_patches
    )

    # Each subject's attention weights should sum to 1.
    assert torch.allclose(
        attention_weights.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-6
    )

    print("Attention pooling test passed.")
