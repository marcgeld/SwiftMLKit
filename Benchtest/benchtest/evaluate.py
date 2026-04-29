import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)


def _score_from_model(model, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)


def evaluate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, test_indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict] = []
    predictions = pd.DataFrame({"test_row_index": test_indices, "y_true": y_test})

    for name, model in models.items():
        try:
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_sec = time.perf_counter() - train_start

            infer_start = time.perf_counter()
            y_pred = model.predict(X_test)
            y_score = _score_from_model(model, X_test)
            infer_time_sec = time.perf_counter() - infer_start

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr, tpr, _ = roc_curve(y_test, y_score)
            specificity = tn / (tn + fp) if (tn + fp) else 0.0

            results.append(
                {
                    "Model": name,
                    "Status": "ok",
                    "Error": "",
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "BalancedAccuracy": balanced_accuracy_score(y_test, y_pred),
                    "Specificity": specificity,
                    "MCC": matthews_corrcoef(y_test, y_pred),
                    "AUC": auc(fpr, tpr),
                    "TrainTimeSec": train_time_sec,
                    "InferenceTimeSec": infer_time_sec,
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                }
            )
            predictions[f"pred::{name}"] = y_pred
            predictions[f"score::{name}"] = y_score
        except Exception as exc:
            results.append(
                {
                    "Model": name,
                    "Status": "error",
                    "Error": str(exc),
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1": np.nan,
                    "BalancedAccuracy": np.nan,
                    "Specificity": np.nan,
                    "MCC": np.nan,
                    "AUC": np.nan,
                    "TrainTimeSec": np.nan,
                    "InferenceTimeSec": np.nan,
                    "TN": np.nan,
                    "FP": np.nan,
                    "FN": np.nan,
                    "TP": np.nan,
                }
            )
            predictions[f"pred::{name}"] = np.nan
            predictions[f"score::{name}"] = np.nan

    metrics_df = pd.DataFrame(results).sort_values(by=["Status", "Recall", "AUC"], ascending=[True, False, False], na_position="last")
    return metrics_df, predictions

