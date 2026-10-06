"""
ResNet18-based patch feature extractor.

Each 32x32 MRI patch is processed independently using an
ImageNet-pretrained ResNet18. The final classification layer
is removed, and the 512-dimensional global average pooled
feature is used as the patch representation.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PatchResNet18(nn.Module):
    """
    ResNet18 feature extractor for MRI patches.

    Input
    -----
    x : torch.Tensor
        Shape [B, 3, 32, 32]

    Output
    ------
    torch.Tensor
        Shape [B, 512]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = resnet18(weights=weights)

        # Remove the final ImageNet classification layer.
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

    def forward(self, x):
        """
        Extract a 512-dimensional feature from each patch.
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, 3, H, W], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels, got {x.shape[1]}."
            )

        x = self.features(x)

        # Global average pooling output:
        # [B, 512, 1, 1] -> [B, 512]
        x = torch.flatten(x, start_dim=1)

        return x


def extract_patch_features(model, patches):
    """
    Extract features from a collection of MRI patches.

    Parameters
    ----------
    model : PatchResNet18
        ResNet18 patch feature extractor.

    patches : torch.Tensor
        Patch tensor of shape [N, 3, 32, 32].

    Returns
    -------
    torch.Tensor
        Feature tensor of shape [N, 512].
    """

    if patches.ndim != 4:
        raise ValueError(
            f"Expected patches with shape [N, 3, H, W], "
            f"got {tuple(patches.shape)}."
        )

    features = model(patches)

    return features


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Test the patch feature extractor
    # ---------------------------------------------------------------

    model = PatchResNet18(pretrained=True)

    # One subject with 49 patches.
    patches = torch.randn(
        49,
        3,
        32,
        32
    )

    features = extract_patch_features(
        model,
        patches
    )

    print("Input patch shape:    ", tuple(patches.shape))
    print("Output feature shape: ", tuple(features.shape))

    assert features.shape == (49, 512)

    print("Patch ResNet18 test passed.")
