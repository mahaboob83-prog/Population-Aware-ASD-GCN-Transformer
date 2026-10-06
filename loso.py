"""
Leave-One-Site-Out (LOSO) Evaluation
for Population-Aware ASD Classification.

Protocol:
- One ABIDE-I acquisition site is held out for testing.
- All remaining sites are used for training.
- A validation subset is selected from the training population.
- The population graph is constructed using training subjects only.
- The held-out site is attached to the training population only during inference.
- Each site is evaluated independently.
- Results are saved for every site.

IMPORTANT:
A valid site_mapping.json file is required.

The mapping must associate each image filename with its
ABIDE-I acquisition site.

Example:

{
    "subject001.png": "NYU",
    "subject002.png": "UCLA",
    "subject003.png": "PITT"
}

Do not invent or infer site labels from filenames.
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

from sklearn.model_selection import train_test_split
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

SITE_MAPPING_FILE = "site_mapping.json"

RESULTS_DIR = "results/loso"
CHECKPOINT_DIR = "checkpoints/loso"

VALIDATION_RATIO = 0.10

BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

GRADIENT_CLIP_MAX_NORM = 5.0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

NUM_EXPECTED_SITES = 17

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================

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

image_paths = dataset.get_image_paths()

NUM_SUBJECTS = len(labels)


# ============================================================
# Load site mapping
# ============================================================

def load_site_mapping():

    if not os.path.exists(
        SITE_MAPPING_FILE
    ):

        raise FileNotFoundError(
            f"\nSite mapping file not found:\n"
            f"{SITE_MAPPING_FILE}\n\n"
            f"LOSO evaluation requires acquisition-site "
            f"labels for every subject.\n"
            f"Do not infer or invent site labels."
        )

    with open(
        SITE_MAPPING_FILE,
        "r"
    ) as file:

        mapping = json.load(
            file
        )

    if not isinstance(
        mapping,
        dict
    ):

        raise ValueError(
            "site_mapping.json must contain "
            "a JSON dictionary."
        )

    return mapping


# ============================================================
# Match site labels to dataset images
# ============================================================

def build_site_labels():

    site_mapping = load_site_mapping()

    site_labels = []

    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )

        if filename not in site_mapping:

            raise ValueError(
                f"\nNo site label found for:\n"
                f"{filename}\n\n"
                f"Every dataset image must have an "
                f"ABIDE-I acquisition-site label."
            )

        site_labels.append(
            site_mapping[filename]
        )

    return np.asarray(
        site_labels
    )


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
# Train embedding extraction
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
# Extract embeddings without gradients
# ============================================================

def extract_subject_embeddings(
    indices,
    patch_encoder,
    subject_representation
):

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
# Attach held-out subject
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

            subject_embedding, _ = (
                subject_representation(
                    patch_features.unsqueeze(0)
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

            subject_embedding, _ = (
                subject_representation(
                    patch_features.unsqueeze(0)
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
# Train one LOSO site
# ============================================================

def train_one_site(
    site,
    train_indices,
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

    # --------------------------------------------------------
    # Training -> validation split
    # --------------------------------------------------------

    train_labels = labels[
        train_indices
    ]

    training_indices, validation_indices = (
        train_test_split(
            train_indices,
            test_size=VALIDATION_RATIO,
            stratify=train_labels,
            random_state=SEED
        )
    )

    training_labels = torch.tensor(
        labels[training_indices],
        dtype=torch.float32,
        device=DEVICE
    )

    best_val_accuracy = -np.inf
    best_state = None
    patience_counter = 0

    # --------------------------------------------------------
    # Training
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

        training_embeddings = (
            extract_train_embeddings(
                training_indices,
                patch_encoder,
                subject_representation
            )
        )

        _, normalized_adjacency = (
            build_population_graph(
                training_embeddings
            )
        )

        gcn_features = gcn(
            training_embeddings,
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
            training_labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=GRADIENT_CLIP_MAX_NORM
        )

        optimizer.step()

        validation_accuracy = (
            evaluate_validation(
                validation_indices,
                training_embeddings.detach(),
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

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Site {site} | "
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
    # Save LOSO checkpoint
    # --------------------------------------------------------

    safe_site = str(site).replace(
        "/",
        "_"
    ).replace(
        "\\",
        "_"
    ).replace(
        " ",
        "_"
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"site_{safe_site}.pt"
    )

    torch.save(
        best_state,
        checkpoint_path
    )

    # --------------------------------------------------------
    # Test held-out site
    # --------------------------------------------------------

    metrics = evaluate_test(
        training_indices,
        test_indices,
        patch_encoder,
        subject_representation,
        gcn,
        transformer,
        classifier
    )

    metrics["site"] = str(site)
    metrics["test_subjects"] = len(
        test_indices
    )
    metrics["best_epoch"] = (
        best_state["epoch"]
    )

    metrics["best_validation_accuracy"] = (
        best_state["validation_accuracy"]
    )

    return metrics


# ============================================================
# Save LOSO results
# ============================================================

def save_results(results):

    json_path = os.path.join(
        RESULTS_DIR,
        "loso_results.json"
    )

    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    csv_path = os.path.join(
        RESULTS_DIR,
        "loso_results.csv"
    )

    fieldnames = [
        "site",
        "test_subjects",
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

    print(
        f"\nLOSO results saved to: {RESULTS_DIR}"
    )


# ============================================================
# Calculate LOSO summary
# ============================================================

def calculate_summary(results):

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]

    summary = {}

    for metric in metrics:

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
        "loso_summary.json"
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
# Main LOSO procedure
# ============================================================

def run_loso():

    site_labels = build_site_labels()

    unique_sites = sorted(
        np.unique(site_labels)
    )

    print(
        f"Subjects: {NUM_SUBJECTS}"
    )

    print(
        f"Acquisition sites: "
        f"{len(unique_sites)}"
    )

    if len(unique_sites) != NUM_EXPECTED_SITES:

        raise ValueError(
            f"Expected {NUM_EXPECTED_SITES} "
            f"ABIDE-I sites, but found "
            f"{len(unique_sites)}."
        )

    results = []

    for site in unique_sites:

        test_indices = np.where(
            site_labels == site
        )[0]

        train_indices = np.where(
            site_labels != site
        )[0]

        print(
            f"\nLOSO site: {site} | "
            f"Train: {len(train_indices)} | "
            f"Test: {len(test_indices)}"
        )

        metrics = train_one_site(
            site,
            train_indices,
            test_indices
        )

        results.append(
            metrics
        )

        print(
            f"Site {site} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"AUC={metrics['auc']:.4f}"
        )

        save_results(
            results
        )

    summary = calculate_summary(
        results
    )

    print("\nLOSO Mean +/- SD")

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


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    run_loso()
