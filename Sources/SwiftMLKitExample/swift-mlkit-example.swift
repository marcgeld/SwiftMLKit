// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import SwiftMLKit
import MLX
import MLXNN
import MLXOptimizers
import TabularData

// Verify build without Metal: export MLX_METAL_ENABLED=0
// MLX.setDefaultDevice(.cpu)
@main
struct swiftmlx {
    
    static func loadData() throws -> (MLXArray, MLXArray) {
        let url = Bundle.module.url(forResource: "breast_cancer", withExtension: "csv")!

        //let url = URL(fileURLWithPath: "Data/breast_cancer.csv")
        let df = try DataFrame(contentsOfCSVFile: url)
        print(df)
        //print(df.prefix(5))
        //print(df.columns.map(\.name))
        print("Rows:", df.rows.count)
        print("Columns:", df.columns.count)

        // Labels: Tensor av rang 1, vektor
        let labels: [Float32] = df["diagnosis"].map {
            ($0 as! String == "M") ? 1.0 : 0.0
        }
        let y = MLXArray(labels)
        
        // Features (droppa id + diagnosis = börja på index 2)
        let featureColumns =
            df.columns.filter { $0.name != "id" && $0.name != "diagnosis" }

        // Features: Tensor av rang 2, matris
        var features: [Float32] = []
        features.reserveCapacity(df.rows.count * featureColumns.count)

        for row in df.rows {
            for col in featureColumns {
                features.append(Float32(row[col.name] as! Double))
            }
        }
       
        let X = MLXArray(features).reshaped([df.rows.count, featureColumns.count])

        // Check shape
        precondition(X.shape == [569, 30])
        precondition(y.shape == [569])
        
        // Normalisera Data
        let mean = X.mean(axis: 0)
        let variance = ((X - mean) * (X - mean)).mean(axis: 0)
        // Ett väldigt litet tal, men inte noll
        let std = sqrt(variance + 1e-8)
        let Xnorm = (X - mean) / std

        return (Xnorm, y)
    }
    
    static func main() throws {

        // MARK: - Load data

        let (X, y): (MLXArray, MLXArray) = try loadData()

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        // Ensure y has shape [N, 1]
        let y2 = y.reshaped([y.shape[0], 1])

        // MARK: - Train/Test split (80/20)

        let n = X.shape[0]
        let split = Int(Double(n) * 0.8)

        let Xtrain = X[0..<split]
        let ytrain = y2[0..<split]

        let Xtest  = X[split..<n]
        let ytest  = y2[split..<n]

        // MARK: - SVM requires {-1, +1}

        let ytrainSVM = 2 * ytrain - 1
        let ytestSVM  = 2 * ytest - 1

        // MARK: - Models
        let models: [(String, any Model, MLXArray, MLXArray)] = [

            ("Logistic Regression",
             LogisticRegression(inputSize: X.shape[1], epochs: 500, learningRate: 0.01),
             ytrain, ytest),

            ("SVM",
             SVM(inputSize: X.shape[1], epochs: 500, learningRate: 0.01),
             ytrainSVM, ytestSVM),

            ("KNN",
             KNN(k: 5),
             ytrain, ytest),

            ("Naive Bayes",
             GaussianNaiveBayes(),
             ytrain, ytest),

            ("LDA",
             LDA(),
             ytrain, ytest),

            ("QDA",
             QDA(),
             ytrain, ytest),

            // Tree-based models 🌲
            ("Decision Tree",
             DecisionTree(maxDepth: 5),
             ytrain, ytest),

            ("Random Forest",
             RandomForest(nTrees: 20, maxDepth: 5),
             ytrain, ytest),

            ("Extra Trees",
             ExtraTrees(nTrees: 20, maxDepth: 5),
             ytrain, ytest),

            ("Gradient Boosting",
             GradientBoosting(nEstimators: 20, learningRate: 0.1),
             ytrain, ytest)
        ]

        // MARK: - Loop models

        for (name, baseModel, ytr, yte) in models {

            print("\n--- \(name) ---")

            // Important: copy model (value semantics)
            let model = baseModel

            var pipeline = Pipeline(steps: [
                .transformer(StandardScaler()),
                .model(model)
            ])

            // Train
            pipeline.fit(X: Xtrain, y: ytr)

            // Predict
            let predsRaw = pipeline.predict(X: Xtest)

            // Convert predictions for metrics (SVM → 0/1)
            let preds: MLXArray = (name == "SVM")
                ? `where`(predsRaw .> 0, MLXArray(1), MLXArray(0))
                : predsRaw

            // Metrics ALWAYS use 0/1 labels
            let yEval: MLXArray = (name == "SVM")
                ? `where`(yte .> 0, MLXArray(1), MLXArray(0))
                : yte

            let cm = ConfusionMatrix().compute(yEval, preds)

            print("""
              Accuracy:  \(cm.accuracy)
              Precision: \(cm.precision)
              Recall:    \(cm.recall)
              F1:        \(cm.f1)

              TP: \(cm.TP)  TN: \(cm.TN)
              FP: \(cm.FP)  FN: \(cm.FN)
            """)
        }
    }
}
