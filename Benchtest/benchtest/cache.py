"""Generate parity-test cache: one dataset file + one file per algorithm."""

import json
import time
from pathlib import Path

import numpy as np
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

from .config import BenchmarkConfig
from .data import load_tumor_data, make_split
from .models import build_model_registry


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_").strip("_")


def _extract_params(model) -> dict:
    params = model.get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}


def _score_from_model(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def _compute_metrics(y_true, y_pred, y_score=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    auc_value = None
    if y_score is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = float(auc(fpr, tpr))
        except Exception:
            pass

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Specificity": specificity,
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "AUC": auc_value,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def generate_cache(config: BenchmarkConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_tumor_data(config.data_path)
    split = make_split(df, config.test_size, config.random_state, config.scale_features)

    dataset = {
        "name": "breast_cancer",
        "random_state": config.random_state,
        "test_size": config.test_size,
        "scale_features": config.scale_features,
        "n_train": int(split.X_train.shape[0]),
        "n_test": int(split.X_test.shape[0]),
        "n_features": int(split.X_train.shape[1]),
        "feature_names": split.feature_names,
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
        "X_train": np.round(split.X_train, 8).tolist(),
        "X_test": np.round(split.X_test, 8).tolist(),
        "y_train": split.y_train.tolist(),
        "y_test": split.y_test.tolist(),
    }

    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    print(f"  Dataset: {dataset_path} ({dataset['n_train']} train, {dataset['n_test']} test)")

    artifacts = {"dataset": str(dataset_path)}
    models = build_model_registry(config.random_state)

    for name, model in models.items():
        filename = _safe_filename(name) + ".json"
        filepath = output_dir / filename

        try:
            t0 = time.perf_counter()
            model.fit(split.X_train, split.y_train)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(split.X_test)
            infer_time = time.perf_counter() - t0

            y_score = _score_from_model(model, split.X_test)
            metrics = _compute_metrics(
                split.y_test, y_pred,
                y_score=y_score if y_score is not None else y_pred,
            )

            result = {
                "algorithm": name,
                "dataset": "breast_cancer",
                "status": "ok",
                "hyperparameters": _extract_params(model),
                "predictions": y_pred.tolist(),
                "probabilities": y_score.tolist() if y_score is not None else None,
                "metrics": metrics,
                "train_time_sec": train_time,
                "inference_time_sec": infer_time,
            }
            print(f"  {name}: Accuracy={metrics['Accuracy']:.4f}  F1={metrics['F1']:.4f}")

        except Exception as exc:
            result = {
                "algorithm": name,
                "dataset": "breast_cancer",
                "status": "error",
                "error": str(exc),
                "hyperparameters": {},
                "predictions": [],
                "probabilities": None,
                "metrics": {},
                "train_time_sec": None,
                "inference_time_sec": None,
            }
            print(f"  {name}: ERROR — {exc}")

        filepath.write_text(json.dumps(result, indent=2), encoding="utf-8")
        artifacts[name] = str(filepath)

    return artifacts
