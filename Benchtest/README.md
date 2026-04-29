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
