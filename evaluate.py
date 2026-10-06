"""
Evaluation utility for the 10-fold cross-validation results.

Reads:
    results/fold_01.json
    ...
    results/fold_10.json

Produces:
    - Fold-wise performance
    - Mean +/- SD for Accuracy
    - Mean +/- SD for Precision
    - Mean +/- SD for Recall
    - Mean +/- SD for F1-score
    - Mean +/- SD for ROC-AUC
"""

import os
import json
import numpy as np


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"

NUM_FOLDS = 10


# ============================================================
# Load fold results
# ============================================================

def load_fold_results():

    results = []

    for fold in range(
        1,
        NUM_FOLDS + 1
    ):

        path = os.path.join(
            RESULTS_DIR,
            f"fold_{fold:02d}.json"
        )

        if not os.path.exists(path):

            print(
                f"Missing result: {path}"
            )

            continue

        with open(
            path,
            "r"
        ) as file:

            result = json.load(
                file
            )

        results.append(
            result
        )

    return results


# ============================================================
# Calculate mean and standard deviation
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

    return summary


# ============================================================
# Print results
# ============================================================

def print_results(
    results,
    summary
):

    print("\nFold-wise results")

    for result in results:

        print(
            f"Fold {result['fold']:02d}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"AUC={result['auc']:.4f}"
        )

    print("\nMean +/- SD")

    print(
        f"Accuracy : "
        f"{summary['accuracy']['mean']:.4f} "
        f"+/- "
        f"{summary['accuracy']['std']:.4f}"
    )

    print(
        f"Precision: "
        f"{summary['precision']['mean']:.4f} "
        f"+/- "
        f"{summary['precision']['std']:.4f}"
    )

    print(
        f"Recall   : "
        f"{summary['recall']['mean']:.4f} "
        f"+/- "
        f"{summary['recall']['std']:.4f}"
    )

    print(
        f"F1-score : "
        f"{summary['f1']['mean']:.4f} "
        f"+/- "
        f"{summary['f1']['std']:.4f}"
    )

    print(
        f"AUC      : "
        f"{summary['auc']['mean']:.4f} "
        f"+/- "
        f"{summary['auc']['std']:.4f}"
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(summary):

    path = os.path.join(
        RESULTS_DIR,
        "evaluation_summary.json"
    )

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"\nSummary saved to: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    if not os.path.exists(
        RESULTS_DIR
    ):

        print(
            "Results directory does not exist."
        )

        print(
            "Run train.py first."
        )

        return

    results = load_fold_results()

    if len(results) == 0:

        print(
            "No fold results were found."
        )

        print(
            "Run train.py first."
        )

        return

    summary = calculate_summary(
        results
    )

    print_results(
        results,
        summary
    )

    save_summary(
        summary
    )


if __name__ == "__main__":

    main()
