"""
Evaluation metrics for ASD versus TD classification.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):
    """
    Calculate classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.

    y_pred : array-like
        Predicted class labels.

    y_probability : array-like
        Predicted probability for the ASD class.

    Returns
    -------
    dict
        Dictionary containing Accuracy, Precision, Recall,
        F1-score, and ROC-AUC.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probability = np.asarray(y_probability)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_probability
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def summarize_metrics(fold_results):
    """
    Calculate mean and standard deviation across folds.

    Parameters
    ----------
    fold_results : list of dict
        Metrics from individual cross-validation folds.

    Returns
    -------
    dict
        Mean and standard deviation for each metric.
    """

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    summary = {}

    for metric in metric_names:

        values = np.asarray([
            result[metric]
            for result in fold_results
        ])

        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(
                values,
                ddof=1
            )),
        }

    return summary


if __name__ == "__main__":

    # Example only.
    # These values are for testing the metric functions and
    # are not experimental results from the manuscript.

    y_true = np.array([
        0, 0, 1, 1, 1
    ])

    y_pred = np.array([
        0, 1, 1, 1, 0
    ])

    y_probability = np.array([
        0.10, 0.60, 0.80, 0.90, 0.30
    ])

    results = calculate_metrics(
        y_true,
        y_pred,
        y_probability
    )

    print("Example metric calculation:")

    for name, value in results.items():
        print(
            f"{name}: {value:.4f}"
        )
