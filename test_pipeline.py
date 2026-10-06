"""
End-to-end pipeline test.

Checks:
1. ABIDE-I dataset loading
2. MRI patch extraction
3. ResNet18 patch feature extraction
4. Attention-based subject representation
5. Population graph construction
6. GCN
7. Transformer
8. ASD/TD classifier

This script does NOT train the model.
"""

import torch

from dataset.abide_dataset import ABIDEDataset
from models.patch_resnet import ResNet18PatchEncoder
from models.attention_pooling import SubjectRepresentation
from graph.population_graph import build_population_graph
from models.gcn import PopulationGCN
from models.transformer import PopulationTransformer
from models.classifier import ASDClassifier
from utils.seed import set_seed


# ============================================================
# Configuration
# ============================================================

SEED = 42
DATASET_ROOT = "E:/ABIDE1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

set_seed(SEED)


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 70)
    print("TESTING POPULATION-AWARE ASD PIPELINE")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    dataset = ABIDEDataset(
        dataset_root=DATASET_ROOT
    )

    print(
        f"\nDataset subjects: {len(dataset)}"
    )

    labels = dataset.get_labels()

    print(
        f"ASD subjects: {sum(label == 1 for label in labels)}"
    )

    print(
        f"TD subjects: {sum(label == 0 for label in labels)}"
    )

    # --------------------------------------------------------
    # 2. Load a few subjects
    # --------------------------------------------------------

    num_test_subjects = min(
        10,
        len(dataset)
    )

    subject_indices = list(
        range(num_test_subjects)
    )

    batch_patches = []

    for idx in subject_indices:

        patches, label = dataset[idx]

        print(
            f"Subject {idx}: "
            f"patch shape = {tuple(patches.shape)}, "
            f"label = {label}"
        )

        batch_patches.append(
            patches
        )

    # Expected:
    # [10, 49, 3, 32, 32]

    patches = torch.stack(
        batch_patches,
        dim=0
    ).to(DEVICE)

    print(
        f"\nPatch batch shape: "
        f"{tuple(patches.shape)}"
    )

    assert patches.shape[1:] == (
        49,
        3,
        32,
        32
    ), (
        "Unexpected patch shape. "
        "Expected [N, 49, 3, 32, 32]."
    )

    # --------------------------------------------------------
    # 3. ResNet18 patch encoder
    # --------------------------------------------------------

    patch_encoder = ResNet18PatchEncoder(
        pretrained=True
    ).to(DEVICE)

    patch_encoder.eval()

    num_subjects = patches.shape[0]
    num_patches = patches.shape[1]

    patches_for_resnet = patches.view(
        num_subjects * num_patches,
        3,
        32,
        32
    )

    with torch.no_grad():

        patch_features = patch_encoder(
            patches_for_resnet
        )

    # Expected:
    # [N*49, 512]

    print(
        f"ResNet18 output: "
        f"{tuple(patch_features.shape)}"
    )

    assert patch_features.shape == (
        num_subjects * 49,
        512
    ), (
        "Unexpected ResNet18 output shape. "
        "Expected [N*49, 512]."
    )

    patch_features = patch_features.view(
        num_subjects,
        num_patches,
        512
    )

    print(
        f"Patch feature tensor: "
        f"{tuple(patch_features.shape)}"
    )

    # --------------------------------------------------------
    # 4. Attention-based subject representation
    # --------------------------------------------------------

    subject_representation = SubjectRepresentation(
        patch_feature_dim=512,
        embedding_dim=128,
        attention_hidden_dim=128
    ).to(DEVICE)

    subject_representation.eval()

    with torch.no_grad():

        subject_embeddings, attention_weights = (
            subject_representation(
                patch_features
            )
        )

    # Expected:
    # subject_embeddings = [N, 128]
    # attention_weights = [N, 49]

    print(
        f"Subject embeddings: "
        f"{tuple(subject_embeddings.shape)}"
    )

    print(
        f"Attention weights: "
        f"{tuple(attention_weights.shape)}"
    )

    assert subject_embeddings.shape == (
        num_subjects,
        128
    )

    assert attention_weights.shape == (
        num_subjects,
        49
    )

    # Check attention weights sum to 1

    attention_sums = (
        attention_weights.sum(dim=1)
    )

    assert torch.allclose(
        attention_sums,
        torch.ones_like(attention_sums),
        atol=1e-5
    )

    print(
        "Attention weights sum-to-one check: PASSED"
    )

    # --------------------------------------------------------
    # 5. Population graph
    # --------------------------------------------------------

    raw_adjacency, normalized_adjacency = (
        build_population_graph(
            subject_embeddings
        )
    )

    print(
        f"\nRaw adjacency: "
        f"{tuple(raw_adjacency.shape)}"
    )

    print(
        f"Normalized adjacency: "
        f"{tuple(normalized_adjacency.shape)}"
    )

    assert raw_adjacency.shape == (
        num_subjects,
        num_subjects
    )

    assert normalized_adjacency.shape == (
        num_subjects,
        num_subjects
    )

    # Symmetry check

    assert torch.allclose(
        raw_adjacency,
        raw_adjacency.T,
        atol=1e-5
    )

    print(
        "Graph symmetry check: PASSED"
    )

    # --------------------------------------------------------
    # 6. GCN
    # --------------------------------------------------------

    gcn = PopulationGCN(
        input_dim=128,
        hidden_dim=128,
        output_dim=128
    ).to(DEVICE)

    gcn.eval()

    with torch.no_grad():

        gcn_output = gcn(
            subject_embeddings,
            normalized_adjacency
        )

    print(
        f"GCN output: "
        f"{tuple(gcn_output.shape)}"
    )

    assert gcn_output.shape == (
        num_subjects,
        128
    )

    # --------------------------------------------------------
    # 7. Transformer
    # --------------------------------------------------------

    transformer = PopulationTransformer(
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_nodes=1112
    ).to(DEVICE)

    transformer.eval()

    with torch.no_grad():

        transformer_output = transformer(
            gcn_output
        )

    print(
        f"Transformer output: "
        f"{tuple(transformer_output.shape)}"
    )

    assert transformer_output.shape == (
        num_subjects,
        128
    )

    # --------------------------------------------------------
    # 8. Classifier
    # --------------------------------------------------------

    classifier = ASDClassifier(
        input_dim=128,
        hidden_dim=64,
        output_dim=1,
        dropout=0.3
    ).to(DEVICE)

    classifier.eval()

    with torch.no_grad():

        predictions = classifier(
            transformer_output
        )

    print(
        f"Classifier output: "
        f"{tuple(predictions.shape)}"
    )

    assert predictions.shape == (
        num_subjects,
        1
    )

    # --------------------------------------------------------
    # 9. Prediction range
    # --------------------------------------------------------

    assert torch.all(
        predictions >= 0
    )

    assert torch.all(
        predictions <= 1
    )

    print(
        "Classifier probability range check: PASSED"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 70)

    print(
        "\nPipeline:"
    )

    print(
        "MRI → 49 patches → ResNet18 → "
        "512-D → Attention → 128-D → "
        "Graph → GCN → Transformer → Classifier"
    )

    print(
        "\nThe pipeline is ready for training."
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
