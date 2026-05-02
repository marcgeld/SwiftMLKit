from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def round_float(value: float, precision: int = 6) -> float:
    return float(f"{float(value):.{precision}f}")


def normalize_numbers(value: Any, precision: int = 6) -> Any:
    if isinstance(value, dict):
        return {key: normalize_numbers(item, precision=precision) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item, precision=precision) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return round_float(float(value), precision=precision)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return metrics, confusion

