from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .parity_metrics import classification_metrics, normalize_numbers, round_float

SCHEMA_VERSION = "1.0"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def _make_split(seed: int = DEFAULT_SEED, test_size: float = DEFAULT_TEST_SIZE) -> SplitData:
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return SplitData(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)


def _build_payload(
    algorithm: str,
    estimator,
    hyperparameters: dict,
    split: SplitData,
    seed: int,
) -> dict:
    estimator.fit(split.x_train, split.y_train)
    y_pred = estimator.predict(split.x_test)

    metrics, confusion = classification_metrics(split.y_test, y_pred)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "algorithm": algorithm,
            "library": "scikit-learn",
            "task": "classification",
            "dataset": "breast_cancer",
            "n_samples": int(split.y_test.shape[0]),
            "n_features": int(split.x_test.shape[1]),
            "seed": int(seed),
        },
        "hyperparameters": hyperparameters,
        "data": {
            "y_true": [int(v) for v in split.y_test.tolist()],
            "y_pred": [int(v) for v in y_pred.tolist()],
        },
        "metrics": metrics,
        "confusion_matrix": confusion,
        "timing": {
            "train_seconds": 0.0,
            "inference_seconds": 0.0,
        },
    }
    return normalize_numbers(payload, precision=6)


def run_decision_tree(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "ccp_alpha": 0.0,
        "random_state": seed,
    }
    estimator = DecisionTreeClassifier(
        criterion=hyperparameters["criterion"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        ccp_alpha=hyperparameters["ccp_alpha"],
        random_state=hyperparameters["random_state"],
    )
    return _build_payload("decision_tree", estimator, hyperparameters, split, seed)


def run_logistic_regression(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": seed,
    }
    estimator = LogisticRegression(
        solver=hyperparameters["solver"],
        max_iter=hyperparameters["max_iter"],
        random_state=hyperparameters["random_state"],
    )
    return _build_payload("logistic_regression", estimator, hyperparameters, split, seed)


def run_svm_linear(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": seed,
    }
    estimator = SVC(
        kernel=hyperparameters["kernel"],
        C=hyperparameters["C"],
        gamma=hyperparameters["gamma"],
        probability=hyperparameters["probability"],
        random_state=hyperparameters["random_state"],
    )
    return _build_payload("svm_linear", estimator, hyperparameters, split, seed)


def run_svm_rbf(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": seed,
    }
    estimator = SVC(
        kernel=hyperparameters["kernel"],
        C=hyperparameters["C"],
        gamma=hyperparameters["gamma"],
        probability=hyperparameters["probability"],
        random_state=hyperparameters["random_state"],
    )
    return _build_payload("svm_rbf", estimator, hyperparameters, split, seed)


def run_knn(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
    }
    estimator = KNeighborsClassifier(
        n_neighbors=hyperparameters["n_neighbors"],
        weights=hyperparameters["weights"],
        algorithm=hyperparameters["algorithm"],
    )
    return _build_payload("knn", estimator, hyperparameters, split, seed)


def run_gradient_boosting(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "random_state": seed,
    }
    estimator = GradientBoostingClassifier(
        n_estimators=hyperparameters["n_estimators"],
        learning_rate=hyperparameters["learning_rate"],
        max_depth=hyperparameters["max_depth"],
        subsample=hyperparameters["subsample"],
        random_state=hyperparameters["random_state"],
    )
    return _build_payload("gradient_boosting", estimator, hyperparameters, split, seed)


def run_random_forest(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": seed,
        "n_jobs": 1,
    }
    estimator = RandomForestClassifier(
        n_estimators=hyperparameters["n_estimators"],
        criterion=hyperparameters["criterion"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        max_features=hyperparameters["max_features"],
        bootstrap=hyperparameters["bootstrap"],
        random_state=hyperparameters["random_state"],
        n_jobs=hyperparameters["n_jobs"],
    )
    return _build_payload("random_forest", estimator, hyperparameters, split, seed)


def run_extra_trees(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": seed,
        "n_jobs": 1,
    }
    estimator = ExtraTreesClassifier(
        n_estimators=hyperparameters["n_estimators"],
        criterion=hyperparameters["criterion"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        max_features=hyperparameters["max_features"],
        bootstrap=hyperparameters["bootstrap"],
        random_state=hyperparameters["random_state"],
        n_jobs=hyperparameters["n_jobs"],
    )
    return _build_payload("extra_trees", estimator, hyperparameters, split, seed)


def run_naive_bayes(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "var_smoothing": 1e-09,
    }
    estimator = GaussianNB(var_smoothing=hyperparameters["var_smoothing"])
    return _build_payload("naive_bayes", estimator, hyperparameters, split, seed)


def run_lda(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "solver": "svd",
    }
    estimator = LinearDiscriminantAnalysis(solver=hyperparameters["solver"])
    return _build_payload("lda", estimator, hyperparameters, split, seed)


def run_qda(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "reg_param": 0.01,
    }
    estimator = QuadraticDiscriminantAnalysis(reg_param=hyperparameters["reg_param"])
    return _build_payload("qda", estimator, hyperparameters, split, seed)


def run_xgboost(split: SplitData, seed: int) -> dict:
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": 1,
    }
    estimator = XGBClassifier(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"],
        subsample=hyperparameters["subsample"],
        colsample_bytree=hyperparameters["colsample_bytree"],
        eval_metric=hyperparameters["eval_metric"],
        random_state=hyperparameters["random_state"],
        n_jobs=hyperparameters["n_jobs"],
    )
    return _build_payload("xgboost", estimator, hyperparameters, split, seed)


ALGORITHM_REGISTRY = {
    "logistic_regression": run_logistic_regression,
    "svm_linear": run_svm_linear,
    "svm_rbf": run_svm_rbf,
    "knn": run_knn,
    "decision_tree": run_decision_tree,
    "gradient_boosting": run_gradient_boosting,
    "random_forest": run_random_forest,
    "extra_trees": run_extra_trees,
    "naive_bayes": run_naive_bayes,
    "lda": run_lda,
    "qda": run_qda,
    "xgboost": run_xgboost,
}


def generate_cache_files(output_dir: Path, seed: int = DEFAULT_SEED) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _make_split(seed=seed)
    written: list[Path] = []

    for algorithm_name, runner in ALGORITHM_REGISTRY.items():
        payload = runner(split, seed)
        payload["timing"]["train_seconds"] = round_float(payload["timing"]["train_seconds"])
        payload["timing"]["inference_seconds"] = round_float(payload["timing"]["inference_seconds"])

        output_path = output_dir / f"{algorithm_name}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(output_path)

    return written

