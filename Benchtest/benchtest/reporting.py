import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "BalancedAccuracy",
    "Specificity",
    "MCC",
    "AUC",
    "TrainTimeSec",
    "InferenceTimeSec",
    "TN",
    "FP",
    "FN",
    "TP",
]


def create_run_dir(output_root: Path, run_id: str | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_id = run_id or datetime.now(tz=timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_outputs(run_dir: Path, metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, metadata: dict) -> dict:
    metrics_csv = run_dir / "metrics.csv"
    metrics_json = run_dir / "metrics.json"
    predictions_csv = run_dir / "predictions_test.csv"
    metadata_json = run_dir / "metadata.json"

    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_json(metrics_json, orient="records", indent=2)
    predictions_df.to_csv(predictions_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "predictions_csv": str(predictions_csv),
        "metadata_json": str(metadata_json),
    }


def load_metrics(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        loaded = pd.read_csv(path)
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(f"Expected DataFrame from CSV: {path}")
        return loaded
    if path.suffix.lower() == ".json":
        loaded = pd.read_json(path, typ="frame")
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(f"Expected DataFrame from JSON: {path}")
        return loaded
    raise ValueError(f"Unsupported file type for metrics: {path}")


def compare_metrics(python_metrics: pd.DataFrame, swift_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = python_metrics.merge(
        swift_metrics,
        on="Model",
        how="outer",
        suffixes=("_python", "_swift"),
        indicator=True,
    )

    for metric in METRIC_COLUMNS:
        py_col = f"{metric}_python"
        swift_col = f"{metric}_swift"
        if py_col in merged.columns and swift_col in merged.columns:
            merged[f"Delta_{metric}"] = merged[py_col] - merged[swift_col]

    return merged.sort_values(by=["_merge", "Model"], ascending=[True, True])

