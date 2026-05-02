# Benchtest: Python vs Swift ML Benchmark

Benchmarks 12 classification algorithms on the breast cancer dataset and produces structured output (CSV/JSON) that can be directly compared with a Swift implementation using the same data split.

---

## Project layout

```
Benchtest/
├── main.py                  # Thin entrypoint — forwards to benchtest CLI
├── pyproject.toml           # Build metadata and script entrypoint
├── benchtest/
│   ├── cli.py               # CLI: run-python, compare-swift
│   ├── config.py            # BenchmarkConfig dataclass
│   ├── data.py              # Dataset loading, schema normalisation, split, scaling
│   ├── evaluate.py          # Training, inference, and per-model metrics with timing
│   ├── models.py            # Algorithm registry (12 models)
│   └── reporting.py        # Output files and Swift/Python metric comparison
└── outputs/                 # Created at runtime — one folder per run
```

---

## Default dataset

The benchmark defaults to the same CSV file used by the Swift target:

```
../Sources/SwiftMLKitExample/Resources/breast_cancer.csv
```

This path is resolved **relative to the installed package location**, so it works correctly regardless of the current working directory.

The data loader supports two CSV schemas automatically:

| Schema        | Target column | ID column            | Class encoding |
|---------------|---------------|----------------------|----------------|
| Legacy (UCI)  | `Class`       | `Sample code number` | 2 = benign, 4 = malignant |
| Swift (WDBC)  | `diagnosis`   | `id`                 | B = benign, M = malignant |

Both are normalised to binary labels (0 = benign, 1 = malignant) before training.

---

## Quick start

```bash
uv run bench
```

That's it. Trains and evaluates all 12 models using the Swift dataset and writes artifacts to `outputs/<run_id>/`.

---

## All CLI options

### Generate parity cache (ground truth)

```bash
python main.py generate-cache
```

or:

```bash
uv run bench generate-cache
```

This command creates deterministic JSON cache files used as canonical references for Swift parity tests:

- `cache/logistic_regression.json`
- `cache/svm_linear.json`
- `cache/svm_rbf.json`
- `cache/knn.json`
- `cache/decision_tree.json`
- `cache/random_forest.json`
- `cache/gradient_boosting.json`
- `cache/extra_trees.json`
- `cache/naive_bayes.json`
- `cache/lda.json`
- `cache/qda.json`
- `cache/xgboost.json`

Contract guarantees for cache files:

- Fixed dataset: `sklearn.datasets.load_breast_cancer`
- Fixed split: `train_test_split(..., random_state=42, stratify=y)`
- Fixed estimators/hyperparameters per algorithm
- `y_true` and `y_pred` are always integer arrays
- All float values are rounded to 6 decimals
- JSON output uses sorted keys for deterministic ordering
- Timing fields are stable placeholders (`0.0`) so cache files are byte-for-byte reproducible

### Run Python benchmark

```bash
uv run bench run-python [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data PATH` | Swift resource CSV | Path to input CSV file |
| `--output-dir PATH` | `outputs/` | Directory for run artifacts |
| `--run-id STRING` | timestamp | Optional override for run folder name |
| `--test-size FLOAT` | `0.2` | Fraction of data used for testing |
| `--random-state INT` | `42` | Seed for reproducible split |
| `--no-scale` | off | Disable StandardScaler preprocessing |

If `--data` points to a file that does not exist, the run fails immediately with a clear error message and exit code `2` — no partial training is started.

### Compare with Swift results

```bash
uv run bench compare-swift \
  --python-metrics outputs/<run_id>/metrics.csv \
  --swift-metrics swift_metrics.csv \
  --output outputs/swift_comparison.csv
```

| Option | Required | Description |
|--------|----------|-------------|
| `--python-metrics PATH` | yes | Python run metrics (`.csv` or `.json`) |
| `--swift-metrics PATH` | yes | Swift run metrics (`.csv` or `.json`) |
| `--output PATH` | no | Output path (default: `outputs/swift_comparison.csv`) |

---

## Models benchmarked

| Model | Notes |
|-------|-------|
| Logistic Regression | `max_iter=1000` |
| SVM (Linear) | linear kernel, probability calibrated |
| SVM (RBF) | RBF kernel, probability calibrated |
| KNN | k-nearest neighbours |
| Decision Tree | — |
| Random Forest | — |
| Gradient Boosting | — |
| Extra Trees | — |
| Naive Bayes | Gaussian |
| LDA | Linear Discriminant Analysis |
| QDA | Quadratic Discriminant Analysis (`reg_param=0.01`) |
| XGBoost | `eval_metric=logloss` |

QDA uses `reg_param=0.01` to regularise the covariance matrix for high-dimensional/collinear data such as the WDBC breast cancer dataset.

### Algorithms included in parity cache

- `logistic_regression` (`LogisticRegression`)
- `svm_linear` (`SVC(kernel="linear")`)
- `svm_rbf` (`SVC(kernel="rbf")`)
- `knn` (`KNeighborsClassifier`)
- `decision_tree` (`DecisionTreeClassifier`)
- `random_forest` (`RandomForestClassifier`)
- `gradient_boosting` (`GradientBoostingClassifier`)
- `extra_trees` (`ExtraTreesClassifier`)
- `naive_bayes` (`GaussianNB`)
- `lda` (`LinearDiscriminantAnalysis`)
- `qda` (`QuadraticDiscriminantAnalysis`)
- `xgboost` (`XGBClassifier`)

---

## Run artifacts

Each run writes to `outputs/<run_id>/`:

| File | Description |
|------|-------------|
| `metrics.csv` | Per-model metrics table |
| `metrics.json` | Same data in JSON format |
| `predictions_test.csv` | Per-row predictions and probability scores for every model |
| `metadata.json` | Config, feature names, train/test split indices |

### Metrics reported per model

- **Classification:** Accuracy, Precision, Recall, F1, Balanced Accuracy, Specificity, MCC
- **Ranking:** AUC (ROC)
- **Timing:** TrainTimeSec, InferenceTimeSec
- **Confusion matrix:** TN, FP, FN, TP
- **Status:** `ok` or `error` with error message if a model fails

---

## Swift comparison output

The `compare-swift` command merges Python and Swift metrics side-by-side and adds `Delta_*` columns for every shared metric so you can inspect numerical parity:

```
Model | Accuracy_python | Accuracy_swift | Delta_Accuracy | ...
```

The Swift metrics file must contain at minimum: `Model`, and any subset of the metric columns above.
