# SwiftMLKit

A scikit-learn-inspired machine learning framework for Swift, built on top of [MLX Swift](https://github.com/ml-explore/mlx-swift) for GPU-accelerated array operations on Apple Silicon.

## Features

- Familiar **fit/predict** API inspired by scikit-learn
- **Pipeline** support for chaining preprocessing and models
- Built on **MLX** for hardware-accelerated computation on Apple Silicon
- Pure Swift with value semantics
- Swift 6 concurrency-safe

## Models

### Classifiers

| Model | Key Parameters |
|---|---|
| Logistic Regression | `inputSize`, `epochs`, `learningRate` |
| SVM | `inputSize`, `epochs`, `learningRate` |
| K-Nearest Neighbors | `k` |
| Gaussian Naive Bayes | -- |
| LDA | -- |
| QDA | -- |
| Decision Tree | `maxDepth` |
| Random Forest | `nTrees`, `maxDepth` |
| Extra Trees | `nTrees`, `maxDepth` |
| Gradient Boosting | `nEstimators`, `learningRate` |

### Preprocessing

- **StandardScaler** -- zero mean, unit variance normalization

### Metrics

- **Accuracy** -- classification accuracy score
- **ConfusionMatrix** -- TP, TN, FP, FN with derived precision, recall, and F1

### Losses

- **BinaryCrossEntropy** -- with logits (numerically stable) and with probabilities

## Requirements

- Swift 6.3+
- macOS 26+ / iOS 26+

## Installation

Add SwiftMLKit as a dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/marcgeld/swiftmlkit.git", from: "0.1.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "SwiftMLKit", package: "swiftmlkit")
    ]
)
```

## Quick Start

```swift
import SwiftMLKit
import MLX

// Load your data as MLXArrays
let X: MLXArray = ...  // [N, features]
let y: MLXArray = ...  // [N, 1]

// Create a pipeline with preprocessing + model
var pipeline = Pipeline(steps: [
    .transformer(StandardScaler()),
    .model(LogisticRegression(inputSize: 30, epochs: 500, learningRate: 0.01))
])

// Train
pipeline.fit(X: xTrain, y: yTrain)

// Predict
let predictions = pipeline.predict(X: xTest)

// Evaluate
let cm = ConfusionMatrix().compute(yTest, predictions)
print("Accuracy:  \(cm.accuracy)")
print("Precision: \(cm.precision)")
print("Recall:    \(cm.recall)")
print("F1:        \(cm.f1)")
```

## Example

The included `SwiftMLKitExample` target trains all 10 models on the Breast Cancer Wisconsin dataset and prints evaluation metrics for each:

```bash
swift run SwiftMLKitExample
```

## License

MIT
