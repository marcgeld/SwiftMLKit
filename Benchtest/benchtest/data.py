from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_tumor_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Class" in df.columns:
        target_column = "Class"
    elif "diagnosis" in df.columns:
        target_column = "diagnosis"
        df = df.rename(columns={"diagnosis": "Class"})
    else:
        raise ValueError("Expected either a 'Class' column or a 'diagnosis' column in the dataset.")

    if target_column == "diagnosis":
        class_values = df["Class"].astype(str).str.upper().str.strip()
        df["Class"] = np.where(class_values == "M", 4, 2)

    numeric_columns = [col for col in df.columns if col != "Class"]
    df_clean = df.copy()
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    nan_counts = df_clean[numeric_columns].isna().sum()
    if nan_counts.sum() > 0:
        raise ValueError(
            "NaN values found after numeric conversion. Fix input data before training.\n"
            f"Columns with NaN:\n{nan_counts[nan_counts > 0].to_string()}"
        )

    return df_clean


def make_split(df: pd.DataFrame, test_size: float, random_state: int, scale_features: bool) -> DataSplit:
    working_df = df.copy()
    if "Sample code number" in working_df.columns:
        working_df = working_df.drop(columns=["Sample code number"])
    if "id" in working_df.columns:
        working_df = working_df.drop(columns=["id"])

    X = working_df.drop(columns=["Class"]).to_numpy()
    y_raw = working_df["Class"].to_numpy()
    y = np.where(y_raw == 4, 1, 0)
    feature_names = list(working_df.drop(columns=["Class"]).columns)

    indices = np.arange(len(working_df))
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        train_indices=idx_train,
        test_indices=idx_test,
    )

