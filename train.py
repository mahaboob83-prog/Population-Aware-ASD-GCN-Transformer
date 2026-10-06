"""
Population-Aware ASD Classification Training Pipeline

Pipeline:
1. Load ABIDE-I sMRI images from ASD/TD folders.
2. Perform subject-level stratified 10-fold cross-validation.
3. Split each development set into training and validation subsets.
4. Extract 49 non-overlapping 32x32 patches from each 224x224 image.
5. Extract 512-D patch features using ImageNet-pretrained ResNet18.
6. Project patch features to 128-D and perform attention-based pooling.
7. Construct the population graph using training subjects only.
8. Apply the two-layer GCN.
9. Apply the population Transformer.
10. Classify ASD vs TD using the MLP classifier.
11. Attach validation/test subjects to the training graph only during inference.
12. Save the best validation checkpoint for each fold.
13. Report fold-wise and mean +/- SD performance.

Labels:
    TD  = 0
    ASD = 1
"""

import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset.abide_dataset import ABIDEDataset
from models.patch_resnet import ResNet18PatchEncoder
from models.attention_pooling import SubjectRepresentation
from models.gcn import PopulationGCN
from models.transformer import PopulationTransformer
from models.classifier import ASDClassifier
from graph.population_graph import build_population_graph
from utils.seed import set_seed


# ============================================================
# Configuration
# ============================================================

SEED = 42

DATASET_ROOT = "E:/ABIDE1"
CHECKPOINT_DIR = "checkpoints"

NUM_FOLDS = 10

# IMPORTANT:
# Set this to the validation proportion actually used
# in your experiments/manuscript.
VALIDATION_RATIO = 0.10

BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

GRADIENT_CLIP_MAX_NORM = 5.0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

set_seed(SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================

dataset = ABIDEDataset(
    dataset_root=DATASET_ROOT
)

labels = np.asarray(dataset.get_labels())

num_subjects = len(labels)

print("=" * 70)
print("Population-Aware ASD Classification")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Subjects     : {num_subjects}")
print(f"ASD subjects : {np.sum(labels == 1)}")
print(f"TD subjects  : {np.sum(labels == 0)}")
print(f"CV folds     : {NUM_FOLDS}")
print(f"Seed         : {SEED}")
print("=" * 70)


# ============================================================
# Model creation
# ============================================================

def create_model():
    """
    Create all trainable components of the proposed framework.
    """

    patch_encoder = ResNet18PatchEncoder(
        pretrained=True
    )

    subject_representation = SubjectRepresentation(
        patch_feature_dim=512,
        embedding_dim=128,
        attention_hidden_dim=128
    )

    gcn = PopulationGCN(
        input_dim=128,
        hidden_dim=128,
        output_dim=128
    )

    transformer = PopulationTransformer(
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_nodes=1112
    )

    classifier = ASDClassifier(
        input_dim=128,
        hidden_dim=64,
        output_dim=1,
        dropout=0.3
    )

    return (
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier,
    )


# ============================================================
# Device helper
# ============================================================

def move_models_to_device(models):
    return tuple(model.to(DEVICE) for model in models)


# ============================================================
# Patch feature extraction
# ============================================================

def extract_subject_embeddings(
    indices,
    patch_encoder,
    subject_representation,
):
    """
    Extract one 128-D subject-level embedding per subject.

    Returns:
        Tensor of shape [N, 128]
    """

    embeddings = []

    patch_encoder.eval()
    subject_representation.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(indices),
            BATCH_SIZE
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            batch_patches = []

            for idx in batch_indices:

                patches, _ = dataset[idx]

                # Expected:
                # [49, 3, 32, 32]
                patches = patches.to(DEVICE)

                batch_patches.append(patches)

            patches = torch.stack(
                batch_patches,
                dim=0
            )

            # [B, 49, 3, 32, 32]
            batch_size = patches.size(0)
            num_patches = patches.size(1)

            patches = patches.view(
                batch_size * num_patches,
                3,
                32,
                32
            )

            # [B*49, 512]
            patch_features = patch_encoder(
                patches
            )

            patch_features = patch_features.view(
                batch_size,
                num_patches,
                512
            )

            # [B, 128]
            subject_features, _ = (
                subject_representation(
                    patch_features
                )
            )

            embeddings.append(
                subject_features.cpu()
            )

    return torch.cat(
        embeddings,
        dim=0
    ).to(DEVICE)


# ============================================================
# Trainable subject embedding extraction
# ============================================================

def extract_train_embeddings(
    indices,
    patch_encoder,
    subject_representation,
):
    """
    Extract subject-level embeddings while retaining
    gradients for model training.

    The population graph is constructed from these
    embeddings during each training epoch.
    """

    embeddings = []

    for start in range(
        0,
        len(indices),
        BATCH_SIZE
    ):

        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        batch_patches = []

        for idx in batch_indices:

            patches, _ = dataset[idx]

            patches = patches.to(DEVICE)

            batch_patches.append(patches)

        patches = torch.stack(
            batch_patches,
            dim=0
        )

        batch_size = patches.size(0)
        num_patches = patches.size(1)

        patches = patches.view(
            batch_size * num_patches,
            3,
            32,
            32
        )

        patch_features = patch_encoder(
            patches
        )

        patch_features = patch_features.view(
            batch_size,
            num_patches,
            512
        )

        subject_features, _ = (
            subject_representation(
                patch_features
            )
        )

        embeddings.append(subject_features)

    return torch.cat(
        embeddings,
        dim=0
    )


# ============================================================
# Attach one held-out subject to training graph
# ============================================================

def attach_subject_to_graph(
    training_embeddings,
    subject_embedding,
):
    """
    Attach one validation/test subject to the training
    population graph.

    The training population remains unchanged.

    No validation/test subject is used to construct the
    original training graph.

    Returns:
        augmented_normalized_adjacency
    """

    all_embeddings = torch.cat(
        [
            training_embeddings,
            subject_embedding.unsqueeze(0)
        ],
        dim=0
    )

    _, normalized_adjacency = (
        build_population_graph(
            all_embeddings
        )
    )

    return normalized_adjacency


# ============================================================
# Forward pass for population
# ============================================================

def population_forward(
    embeddings,
    normalized_adjacency,
    gcn,
    transformer,
    classifier,
):
    """
    Forward propagation through:

        Subject embedding
            ↓
        GCN
            ↓
        Transformer
            ↓
        Classifier
    """

    gcn_features = gcn(
        embeddings,
        normalized_adjacency
    )

    transformer_features = transformer(
        gcn_features
    )

    logits = classifier(
        transformer_features
    )

    return logits


# ============================================================
# Training one fold
# ============================================================

def train_one_fold(
    fold,
    train_indices,
    val_indices,
    test_indices,
):
    """
    Train one cross-validation fold.

    Graph construction:
        training subjects only

    Validation/test:
        attached to training graph only for inference
    """

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    (
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier,
    ) = create_model()

    (
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier,
    ) = move_models_to_device(
        (
            patch_encoder,
            subject_representation,
            gcn,
            transformer,
            classifier,
        )
    )

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.BCELoss()

    parameters = list(
        patch_encoder.parameters()
    )

    parameters += list(
        subject_representation.parameters()
    )

    parameters += list(
        gcn.parameters()
    )

    parameters += list(
        transformer.parameters()
    )

    parameters += list(
        classifier.parameters()
    )

    optimizer = Adam(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    train_labels = torch.tensor(
        labels[train_indices],
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Best checkpoint tracking
    # --------------------------------------------------------

    best_val_accuracy = -np.inf
    best_state = None
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        patch_encoder.train()
        subject_representation.train()
        gcn.train()
        transformer.train()
        classifier.train()

        optimizer.zero_grad()

        # ----------------------------------------------
        # Extract training subject representations
        # ----------------------------------------------

        train_embeddings = extract_train_embeddings(
            train_indices,
            patch_encoder,
            subject_representation
        )

        # ----------------------------------------------
        # Population graph
        # ----------------------------------------------

        _, normalized_adjacency = (
            build_population_graph(
                train_embeddings
            )
        )

        # ----------------------------------------------
        # GCN
        # ----------------------------------------------

        gcn_features = gcn(
            train_embeddings,
            normalized_adjacency
        )

        # ----------------------------------------------
        # Transformer
        # ----------------------------------------------

        transformer_features = transformer(
            gcn_features
        )

        # ----------------------------------------------
        # Classifier
        # ----------------------------------------------

        predictions = classifier(
            transformer_features
        ).view(-1)

        # ----------------------------------------------
        # Training loss
        # ----------------------------------------------

        loss = criterion(
            predictions,
            train_labels
        )

        loss.backward()

        # ----------------------------------------------
        # Gradient clipping
        # ----------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=GRADIENT_CLIP_MAX_NORM
        )

        optimizer.step()

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        val_accuracy = evaluate_validation(
            val_indices,
            train_embeddings.detach(),
            patch_encoder,
            subject_representation,
            gcn,
            transformer,
            classifier
        )

        scheduler.step(
            val_accuracy
        )

        # ----------------------------------------------
        # Best model
        # ----------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = {
                "patch_encoder":
                    copy.deepcopy(
                        patch_encoder.state_dict()
                    ),

                "subject_representation":
                    copy.deepcopy(
                        subject_representation.state_dict()
                    ),

                "gcn":
                    copy.deepcopy(
                        gcn.state_dict()
                    ),

                "transformer":
                    copy.deepcopy(
                        transformer.state_dict()
                    ),

                "classifier":
                    copy.deepcopy(
                        classifier.state_dict()
                    ),

                "epoch": epoch,

                "validation_accuracy":
                    val_accuracy,
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------
        # Progress
        # ----------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {fold:02d} | "
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"LR {current_lr:.2e}"
        )

        # ----------------------------------------------
        # Early stopping
        # ----------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    # ========================================================
    # Restore best checkpoint
    # ========================================================

    patch_encoder.load_state_dict(
        best_state["patch_encoder"]
    )

    subject_representation.load_state_dict(
        best_state["subject_representation"]
    )

    gcn.load_state_dict(
        best_state["gcn"]
    )

    transformer.load_state_dict(
        best_state["transformer"]
    )

    classifier.load_state_dict(
        best_state["classifier"]
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold}.pt"
    )

    torch.save(
        best_state,
        checkpoint_path
    )

    print(
        f"Best checkpoint saved: "
        f"{checkpoint_path}"
    )

    # ========================================================
    # Final test evaluation
    # ========================================================

    test_metrics = evaluate_test(
        train_indices,
        test_indices,
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier
    )

    return test_metrics


# ============================================================
# Validation
# ============================================================

def evaluate_validation(
    val_indices,
    training_embeddings,
    patch_encoder,
    subject_representation,
    gcn,
    transformer,
    classifier,
):
    """
    Evaluate validation subjects by attaching each validation
    subject to the training population graph.

    Validation subjects do NOT modify the training graph.
    """

    patch_encoder.eval()
    subject_representation.eval()
    gcn.eval()
    transformer.eval()
    classifier.eval()

    predictions = []

    with torch.no_grad():

        for idx in val_indices:

            patches, _ = dataset[idx]

            patches = patches.to(DEVICE)

            # [49, 3, 32, 32]
            patch_features = patch_encoder(
                patches
            )

            # [49, 512]
            patch_features = patch_features.unsqueeze(0)

            # [1, 128]
            subject_embedding, _ = (
                subject_representation(
                    patch_features
                )
            )

            subject_embedding = (
                subject_embedding.squeeze(0)
            )

            # ------------------------------------------
            # Attach validation subject to training graph
            # ------------------------------------------

            normalized_adjacency = (
                attach_subject_to_graph(
                    training_embeddings,
                    subject_embedding
                )
            )

            all_embeddings = torch.cat(
                [
                    training_embeddings,
                    subject_embedding.unsqueeze(0)
                ],
                dim=0
            )

            # ------------------------------------------
            # GCN
            # ------------------------------------------

            gcn_features = gcn(
                all_embeddings,
                normalized_adjacency
            )

            # ------------------------------------------
            # Transformer
            # ------------------------------------------

            transformer_features = transformer(
                gcn_features
            )

            # ------------------------------------------
            # Last node = validation subject
            # ------------------------------------------

            test_feature = (
                transformer_features[-1:]
            )

            probability = classifier(
                test_feature
            ).view(-1)

            predictions.append(
                probability.item()
            )

    predicted_labels = (
        np.asarray(predictions) >= 0.5
    ).astype(int)

    true_labels = labels[val_indices]

    return accuracy_score(
        true_labels,
        predicted_labels
    )


# ============================================================
# Test evaluation
# ============================================================

def evaluate_test(
    train_indices,
    test_indices,
    patch_encoder,
    subject_representation,
    gcn,
    transformer,
    classifier,
):
    """
    Evaluate held-out test subjects.

    The training population graph is constructed using
    training subjects only.

    Each test subject is temporarily attached to that
    training population during inference.
    """

    patch_encoder.eval()
    subject_representation.eval()
    gcn.eval()
    transformer.eval()
    classifier.eval()

    # --------------------------------------------------------
    # Training subject embeddings
    # --------------------------------------------------------

    training_embeddings = extract_subject_embeddings(
        train_indices,
        patch_encoder,
        subject_representation
    )

    probabilities = []

    # --------------------------------------------------------
    # Test subjects
    # --------------------------------------------------------

    with torch.no_grad():

        for idx in test_indices:

            patches, _ = dataset[idx]

            patches = patches.to(DEVICE)

            # [49, 512]
            patch_features = patch_encoder(
                patches
            )

            patch_features = (
                patch_features.unsqueeze(0)
            )

            # [1, 128]
            subject_embedding, _ = (
                subject_representation(
                    patch_features
                )
            )

            subject_embedding = (
                subject_embedding.squeeze(0)
            )

            # ------------------------------------------
            # Attach test subject to training graph
            # ------------------------------------------

            normalized_adjacency = (
                attach_subject_to_graph(
                    training_embeddings,
                    subject_embedding
                )
            )

            all_embeddings = torch.cat(
                [
                    training_embeddings,
                    subject_embedding.unsqueeze(0)
                ],
                dim=0
            )

            # ------------------------------------------
            # GCN
            # ------------------------------------------

            gcn_features = gcn(
                all_embeddings,
                normalized_adjacency
            )

            # ------------------------------------------
            # Transformer
            # ------------------------------------------

            transformer_features = transformer(
                gcn_features
            )

            # ------------------------------------------
            # Test subject is the final node
            # ------------------------------------------

            test_feature = (
                transformer_features[-1:]
            )

            probability = classifier(
                test_feature
            ).view(-1)

            probabilities.append(
                probability.item()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    probabilities = np.asarray(
        probabilities
    )

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    true_labels = labels[test_indices]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    if len(np.unique(true_labels)) == 2:

        auc = roc_auc_score(
            true_labels,
            probabilities
        )

    else:

        auc = np.nan

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    print(
        "\nTest Results:"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"AUC       : {auc:.4f}"
    )

    return metrics


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation():

    all_fold_metrics = []

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    for fold, (
        development_indices,
        test_indices
    ) in enumerate(
        skf.split(
            np.zeros(num_subjects),
            labels
        ),
        start=1
    ):

        # ----------------------------------------------------
        # Development -> training + validation
        # ----------------------------------------------------

        development_labels = labels[
            development_indices
        ]

        train_indices, val_indices = (
            train_test_split(
                development_indices,
                test_size=VALIDATION_RATIO,
                stratify=development_labels,
                random_state=SEED
            )
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        assert train_set.isdisjoint(
            val_set
        )

        assert train_set.isdisjoint(
            test_set
        )

        assert val_set.isdisjoint(
            test_set
        )

        # ----------------------------------------------------
        # Print split information
        # ----------------------------------------------------

        print("\n")
        print(
            f"Fold {fold} split:"
        )

        print(
            f"Training   : {len(train_indices)}"
        )

        print(
            f"Validation : {len(val_indices)}"
        )

        print(
            f"Testing    : {len(test_indices)}"
        )

        # ----------------------------------------------------
        # Train fold
        # ----------------------------------------------------

        fold_metrics = train_one_fold(
            fold,
            train_indices,
            val_indices,
            test_indices
        )

        all_fold_metrics.append(
            fold_metrics
        )

    return all_fold_metrics


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    results = run_cross_validation()

    print("\n")
    print("=" * 70)
    print("10-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    for metric in metric_names:

        values = np.asarray(
            [
                result[metric]
                for result in results
            ],
            dtype=float
        )

        mean_value = np.nanmean(
            values
        )

        std_value = np.nanstd(
            values
        )

        print(
            f"{metric.capitalize():10s}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    print("=" * 70)

    print(
        "\nTraining completed successfully."
    )
