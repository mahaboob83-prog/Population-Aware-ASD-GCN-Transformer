"""
Attention-based patch aggregation.

This module projects the 512-dimensional patch features to
128-dimensional embeddings and performs attention-based
aggregation across the 49 MRI patches to obtain a
subject-level representation.
"""

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    """
    Project patch-level ResNet18 features from 512 to 128 dimensions.
    """

    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            embedding_dim
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Patch features with shape:
            [batch_size, num_patches, 512]

        Returns
        -------
        torch.Tensor
            Projected patch embeddings:
            [batch_size, num_patches, 128]
        """

        return self.projection(x)


class AttentionPooling(nn.Module):
    """
    Attention-based aggregation of patch embeddings.

    The attention mechanism assigns a normalized importance
    weight to each patch and computes a weighted sum to obtain
    a subject-level representation.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Patch embeddings with shape:
            [batch_size, num_patches, embedding_dim]

        Returns
        -------
        subject_embedding : torch.Tensor
            Subject-level representation:
            [batch_size, embedding_dim]

        attention_weights : torch.Tensor
            Patch importance weights:
            [batch_size, num_patches]
        """

        scores = self.attention(x)

        scores = scores.squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        subject_embedding = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return subject_embedding, weights


class SubjectRepresentation(nn.Module):
    """
    Complete patch-to-subject representation module.

    Input:
        49 patches × 512 features

    Output:
        128-dimensional subject representation
    """

    def __init__(
        self,
        input_dim=512,
        embedding_dim=128
    ):
        super().__init__()

        self.projection = PatchProjection(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )

        self.pooling = AttentionPooling(
            embedding_dim=embedding_dim
        )

    def forward(self, patch_features):

        patch_embeddings = self.projection(
            patch_features
        )

        subject_embedding, attention_weights = (
            self.pooling(patch_embeddings)
        )

        return (
            subject_embedding,
            attention_weights
        )


if __name__ == "__main__":

    # Shape test
    dummy_features = torch.randn(
        4,      # batch size
        49,     # number of patches
        512     # ResNet18 feature dimension
    )

    model = SubjectRepresentation(
        input_dim=512,
        embedding_dim=128
    )

    subject_embedding, attention_weights = (
        model(dummy_features)
    )

    print(
        "Patch feature shape:",
        dummy_features.shape
    )

    print(
        "Subject embedding shape:",
        subject_embedding.shape
    )

    print(
        "Attention weight shape:",
        attention_weights.shape
    )
