import argparse
import sys
from pathlib import Path

from .cache_generator import DEFAULT_SEED, generate_cache_files
from .config import BenchmarkConfig
from .data import load_tumor_data, make_split
from .evaluate import evaluate_models
from .models import build_model_registry
from .reporting import compare_metrics, create_run_dir, load_metrics, write_run_outputs

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "Sources" / "SwiftMLKitExample" / "Resources" / "breast_cancer.csv"
)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Python ML models and compare with Swift results.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run-python", help="Train and evaluate Python models.")
    run_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV data.")
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for run artifacts.")
    run_parser.add_argument("--run-id", type=str, default=None, help="Optional run folder name.")
    run_parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for test split.")
    run_parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible split.")
    run_parser.add_argument("--no-scale", action="store_true", help="Disable StandardScaler preprocessing.")

    compare_parser = subparsers.add_parser("compare-swift", help="Compare Python benchmark metrics with Swift metrics.")
    compare_parser.add_argument("--python-metrics", type=Path, required=True, help="Path to Python metrics (.csv or .json).")
    compare_parser.add_argument("--swift-metrics", type=Path, required=True, help="Path to Swift metrics (.csv or .json).")
    compare_parser.add_argument("--output", type=Path, default=Path("outputs/swift_comparison.csv"), help="Output path for merged comparison CSV.")

    cache_parser = subparsers.add_parser(
        "generate-cache",
        help="Generate deterministic scikit-learn cache files for Swift parity testing.",
    )
    cache_parser.add_argument("--output-dir", type=Path, default=Path("cache"), help="Directory for generated cache JSON files.")
    cache_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed seed for deterministic train/test split and estimators.")

    return parser


def _run_python(args: argparse.Namespace) -> int:
    data_path = getattr(args, "data", DEFAULT_DATA_PATH)
    output_dir = getattr(args, "output_dir", Path("outputs"))
    test_size = getattr(args, "test_size", 0.2)
    random_state = getattr(args, "random_state", 42)
    no_scale = getattr(args, "no_scale", False)
    run_id = getattr(args, "run_id", None)

    resolved_data_path = Path(data_path).expanduser().resolve()
    using_default_dataset = resolved_data_path == DEFAULT_DATA_PATH.resolve()
    if not resolved_data_path.exists():
        print(
            f"error: Dataset file not found: {resolved_data_path}\n"
            f"  Provide a valid path with --data, or place the Swift dataset at:\n"
            f"  {DEFAULT_DATA_PATH.resolve()}",
            file=sys.stderr,
        )
        sys.exit(2)

    dataset_label = "default" if using_default_dataset else "custom"
    print(f"Dataset in use ({dataset_label}): {resolved_data_path}")

    config = BenchmarkConfig(
        data_path=resolved_data_path,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        scale_features=not no_scale,
    )

    df = load_tumor_data(config.data_path)
    split = make_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        scale_features=config.scale_features,
    )

    models = build_model_registry(config.random_state)
    metrics_df, predictions_df = evaluate_models(
        models,
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        test_indices=split.test_indices,
    )

    run_dir = create_run_dir(config.output_dir, run_id)
    metadata = {
        **config.to_metadata(),
        "feature_names": split.feature_names,
        "train_size": int(len(split.train_indices)),
        "test_size_rows": int(len(split.test_indices)),
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
    }
    output_paths = write_run_outputs(run_dir, metrics_df, predictions_df, metadata)

    print("Python benchmark complete.")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nArtifacts:")
    for key, value in output_paths.items():
        print(f"  - {key}: {value}")

    return 0


def _compare_swift(args: argparse.Namespace) -> int:
    python_metrics = load_metrics(args.python_metrics)
    swift_metrics = load_metrics(args.swift_metrics)

    comparison_df = compare_metrics(python_metrics, swift_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output, index=False)

    print("Swift/Python comparison complete.")
    print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nComparison artifact: {args.output}")

    return 0


def _generate_cache(args: argparse.Namespace) -> int:
    written = generate_cache_files(output_dir=args.output_dir, seed=args.seed)
    print("Deterministic cache generation complete.")
    for path in written:
        print(f"  - {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "run-python"):
        return _run_python(args)
    if args.command == "compare-swift":
        return _compare_swift(args)
    if args.command == "generate-cache":
        return _generate_cache(args)

    parser.error(f"Unknown command: {args.command}")

