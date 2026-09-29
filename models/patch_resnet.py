"""
Patch-wise ResNet18 feature extraction.

This module implements the patch-level feature extractor used
in the proposed structural MRI representation learning framework.

Input:
    32 x 32 MRI patch

Output:
    512-dimensional feature representation
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PatchResNet18(nn.Module):
    """
    ResNet18-based patch feature extractor.

    The final classification layer of ResNet18 is removed so that
    the network produces a 512-dimensional feature representation
    for each MRI patch.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(weights=weights)

        # Remove the original ImageNet classification layer.
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

    def forward(self, x):
        """
        Extract a feature vector for each input patch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape:
            [batch_size, 3, 32, 32]

        Returns
        -------
        torch.Tensor
            Feature tensor with shape:
            [batch_size, 512]
        """

        features = self.feature_extractor(x)

        features = torch.flatten(features, start_dim=1)

        return features


if __name__ == "__main__":
    # Simple shape test.
    model = PatchResNet18(pretrained=False)

    dummy_input = torch.randn(4, 3, 32, 32)

    output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
