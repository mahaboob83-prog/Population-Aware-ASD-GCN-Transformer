"""
Population-Aware ASD Classification Training

Training protocol:
- ABIDE-I folder-based sMRI dataset
- Subject-level stratified 10-fold cross-validation
- Training / validation / held-out test split
- ResNet18 patch feature extraction
- Attention-based subject representation
- Population graph construction using training subjects
- Two-layer GCN
- Two-layer population Transformer
- MLP classifier
- Adam optimizer
- BCE loss
- ReduceLROnPlateau scheduler
- Early stopping
- Mean +/- SD across 10 folds

Labels:
    TD  = 0
    ASD = 1
"""

import os
import json
import csv
import copy

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

NUM_FOLDS = 10

# IMPORTANT:
# Use the validation ratio actually used in your experiment.
VALIDATION_RATIO = 0.10

BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

GRADIENT_CLIP_MAX_NORM = 5.0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

EMBEDDING_DIM = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Directories and seed
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

set_seed(SEED)


# ============================================================
# Dataset
# ============================================================

dataset = ABIDEDataset(
    dataset_root=DATASET_ROOT
)

labels = np.asarray(
    dataset.get_labels()
)

NUM_SUBJECTS = len(labels)

print(f"Device: {DEVICE}")
print(f"Subjects: {NUM_SUBJECTS}")
print(f"ASD: {np.sum(labels == 1)}")
print(f"TD: {np.sum(labels == 0)}")
print(f"Folds: {NUM_FOLDS}")


# ============================================================
# Model creation
# ============================================================

def create_model():

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

    models = (
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier
    )

    return tuple(
        model.to(DEVICE)
        for model in models
    )


# ============================================================
# Extract subject embeddings
# ============================================================

def extract_subject_embeddings(
    indices,
    patch_encoder,
    subject_representation
):
    """
    Extract subject-level 128-D embeddings without gradients.
    """

    patch_encoder.eval()
    subject_representation.eval()

    embeddings = []

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

                batch_patches.append(
                    patches
                )

            patches = torch.stack(
                batch_patches,
                dim=0
            ).to(DEVICE)

            batch_size = patches.shape[0]
            num_patches = patches.shape[1]

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

            embeddings.append(
                subject_features
            )

    return torch.cat(
        embeddings,
        dim=0
    )


# ============================================================
# Extract train embeddings with gradients
# ============================================================

def extract_train_embeddings(
    indices,
    patch_encoder,
    subject_representation
):

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

            batch_patches.append(
                patches
            )

        patches = torch.stack(
            batch_patches,
            dim=0
        ).to(DEVICE)

        batch_size = patches.shape[0]
        num_patches = patches.shape[1]

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

        embeddings.append(
            subject_features
        )

    return torch.cat(
        embeddings,
        dim=0
    )


# ============================================================
# Attach held-out subject to training population
# ============================================================

def attach_subject_to_graph(
    training_embeddings,
    subject_embedding
):

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
# Validation
# ============================================================

def evaluate_validation(
    val_indices,
    training_embeddings,
    patch_encoder,
    subject_representation,
    gcn,
    transformer,
    classifier
):

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

            patch_features = patch_encoder(
                patches
            )

            patch_features = (
                patch_features.unsqueeze(0)
            )

            subject_embedding, _ = (
                subject_representation(
                    patch_features
                )
            )

            subject_embedding = (
                subject_embedding.squeeze(0)
            )

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

            gcn_features = gcn(
                all_embeddings,
                normalized_adjacency
            )

            transformer_features = transformer(
                gcn_features
            )

            probability = classifier(
                transformer_features[-1:]
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
    classifier
):

    patch_encoder.eval()
    subject_representation.eval()
    gcn.eval()
    transformer.eval()
    classifier.eval()

    training_embeddings = (
        extract_subject_embeddings(
            train_indices,
            patch_encoder,
            subject_representation
        )
    )

    probabilities = []

    with torch.no_grad():

        for idx in test_indices:

            patches, _ = dataset[idx]

            patches = patches.to(DEVICE)

            patch_features = patch_encoder(
                patches
            )

            patch_features = (
                patch_features.unsqueeze(0)
            )

            subject_embedding, _ = (
                subject_representation(
                    patch_features
                )
            )

            subject_embedding = (
                subject_embedding.squeeze(0)
            )

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

            gcn_features = gcn(
                all_embeddings,
                normalized_adjacency
            )

            transformer_features = transformer(
                gcn_features
            )

            probability = classifier(
                transformer_features[-1:]
            ).view(-1)

            probabilities.append(
                probability.item()
            )

    probabilities = np.asarray(
        probabilities
    )

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    true_labels = labels[test_indices]

    metrics = {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels
        ),

        "precision": precision_score(
            true_labels,
            predicted_labels,
            zero_division=0
        ),

        "recall": recall_score(
            true_labels,
            predicted_labels,
            zero_division=0
        ),

        "f1": f1_score(
            true_labels,
            predicted_labels,
            zero_division=0
        ),

        "auc": (
            roc_auc_score(
                true_labels,
                probabilities
            )
            if len(np.unique(true_labels)) == 2
            else np.nan
        )
    }

    return metrics


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(
    fold,
    train_indices,
    val_indices,
    test_indices
):

    (
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier
    ) = create_model()

    criterion = nn.BCELoss()

    parameters = (
        list(patch_encoder.parameters())
        + list(subject_representation.parameters())
        + list(gcn.parameters())
        + list(transformer.parameters())
        + list(classifier.parameters())
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

    train_labels = torch.tensor(
        labels[train_indices],
        dtype=torch.float32,
        device=DEVICE
    )

    best_val_accuracy = -np.inf
    best_state = None
    patience_counter = 0

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

        train_embeddings = (
            extract_train_embeddings(
                train_indices,
                patch_encoder,
                subject_representation
            )
        )

        _, normalized_adjacency = (
            build_population_graph(
                train_embeddings
            )
        )

        gcn_features = gcn(
            train_embeddings,
            normalized_adjacency
        )

        transformer_features = transformer(
            gcn_features
        )

        predictions = classifier(
            transformer_features
        ).view(-1)

        loss = criterion(
            predictions,
            train_labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=GRADIENT_CLIP_MAX_NORM
        )

        optimizer.step()

        validation_accuracy = (
            evaluate_validation(
                val_indices,
                train_embeddings.detach(),
                patch_encoder,
                subject_representation,
                gcn,
                transformer,
                classifier
            )
        )

        scheduler.step(
            validation_accuracy
        )

        if validation_accuracy > best_val_accuracy:

            best_val_accuracy = (
                validation_accuracy
            )

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
                    validation_accuracy
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # Minimal progress output
        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Fold {fold}/10 | "
                f"Epoch {epoch}/{MAX_EPOCHS} | "
                f"Loss {loss.item():.4f} | "
                f"Val Acc {validation_accuracy:.4f}"
            )

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold:02d}.pt"
    )

    torch.save(
        best_state,
        checkpoint_path
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    metrics = evaluate_test(
        train_indices,
        test_indices,
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier
    )

    metrics["fold"] = fold
    metrics["best_epoch"] = (
        best_state["epoch"]
    )
    metrics["best_validation_accuracy"] = (
        best_state["validation_accuracy"]
    )

    return metrics


# ============================================================
# Save fold result
# ============================================================

def save_fold_result(metrics):

    path = os.path.join(
        RESULTS_DIR,
        f"fold_{metrics['fold']:02d}.json"
    )

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


# ============================================================
# Save summary CSV
# ============================================================

def save_summary(results):

    csv_path = os.path.join(
        RESULTS_DIR,
        "cross_validation_results.csv"
    )

    fieldnames = [
        "fold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "best_epoch",
        "best_validation_accuracy"
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )

    summary = {}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        values = np.asarray(
            [
                result[metric]
                for result in results
            ],
            dtype=float
        )

        summary[metric] = {
            "mean": float(
                np.nanmean(values)
            ),
            "std": float(
                np.nanstd(values)
            )
        }

    summary_path = os.path.join(
        RESULTS_DIR,
        "cross_validation_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    return summary


# ============================================================
# 10-fold cross-validation
# ============================================================

def run_cross_validation():

    results = []

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
            np.zeros(NUM_SUBJECTS),
            labels
        ),
        start=1
    ):

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

        print(
            f"\nFold {fold}/10 | "
            f"Train: {len(train_indices)} | "
            f"Val: {len(val_indices)} | "
            f"Test: {len(test_indices)}"
        )

        metrics = train_one_fold(
            fold,
            train_indices,
            val_indices,
            test_indices
        )

        results.append(
            metrics
        )

        save_fold_result(
            metrics
        )

        print(
            f"Fold {fold}/10 | "
            f"Test Acc: {metrics['accuracy']:.4f} | "
            f"AUC: {metrics['auc']:.4f}"
        )

    summary = save_summary(
        results
    )

    return results, summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    results, summary = (
        run_cross_validation()
    )

    print("\nFinal 10-fold results")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        print(
            f"{metric}: "
            f"{summary[metric]['mean']:.4f} "
            f"+/- "
            f"{summary[metric]['std']:.4f}"
        )

    print(
        "\nResults saved to:",
        RESULTS_DIR
    )
