"""
Classification metrics for ASD versus TD prediction.

Provides functions for computing accuracy, precision, recall,
F1-score, and ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.

    y_pred : array-like
        Predicted binary labels.

    y_prob : array-like
        Predicted probability for the positive class.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall,
        F1-score, and ROC-AUC.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if y_prob.ndim != 1:
        y_prob = y_prob.reshape(-1)

    if not (
        len(y_true) == len(y_pred) == len(y_prob)
    ):
        raise ValueError(
            "y_true, y_pred, and y_prob must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Input arrays must not be empty."
        )

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

    # ROC-AUC requires both classes to be present.
    if np.unique(y_true).size < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(
            y_true,
            y_prob
        )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    """

    return float(
        accuracy_score(
            y_true,
            y_pred
        )
    )


def calculate_precision(y_true, y_pred):
    """
    Calculate classification precision.
    """

    return float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


def calculate_recall(y_true, y_pred):
    """
    Calculate classification recall.
    """

    return float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


def calculate_f1(y_true, y_pred):
    """
    Calculate F1-score.
    """

    return float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


def calculate_auc(y_true, y_prob):
    """
    Calculate ROC-AUC.

    Returns NaN if only one class is present in y_true.
    """

    if np.unique(y_true).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            y_prob
        )
    )


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Metric test
    # ---------------------------------------------------------------

    y_true = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1
    ])

    y_pred = np.array([
        0, 0, 1, 0,
        1, 1, 0, 1
    ])

    y_prob = np.array([
        0.10, 0.20, 0.70, 0.30,
        0.90, 0.80, 0.40, 0.85
    ])

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob
    )

    print("Classification metrics:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['auc']:.4f}")

    assert all(
        key in metrics
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        ]
    )

    print("Metrics test passed.")
