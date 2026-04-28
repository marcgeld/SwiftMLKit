//
//  KNN.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - K-Nearest Neighbors (Vectorized)
//
// Efficient implementation using matrix operations.
// Computes full distance matrix in one pass.
//
// Works for binary classification (0/1 labels)

public struct KNN: Classifier {

    private var Xtrain: MLXArray?
    private var ytrain: MLXArray?

    public let k: Int

    public init(k: Int = 3) {
        self.k = k
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) {
        self.Xtrain = X
        self.ytrain = y
    }

    // MARK: - Predict (vectorized)
    public func predict(X: MLXArray) -> MLXArray {

        guard let Xtrain, let ytrain else {
            fatalError("KNN not fitted")
        }

        let nTest = X.shape[0]

        // MARK: - Compute squared distances (vectorized)

        // ||X||^2  → shape [nTest, 1]
        let Xnorm = (X * X).sum(axis: 1).reshaped([nTest, 1])

        // ||Xtrain||^2 → shape [1, nTrain]
        let XtrainNorm = (Xtrain * Xtrain).sum(axis: 1).reshaped([1, Xtrain.shape[0]])

        // Cross term
        let cross = matmul(X, Xtrain.T)

        // Full distance matrix: [nTest, nTrain]
        let dist = Xnorm + XtrainNorm - 2 * cross

        // MARK: - Find k nearest neighbors
        // argsort per row → indices of nearest neighbors
        let sortedIdx = argSort(dist, axis: 1)

        // Take first k columns
        let kIdx = sortedIdx[0..<nTest, 0..<k]

        // Gather labels
        let neighbors = ytrain[kIdx]

        // MARK: - Majority vote (binary)
        let mean = neighbors.mean(axis: 1)

        let preds = `where`(mean .> 0.5,
                            MLXArray(1),
                            MLXArray(0))

        return preds.reshaped([nTest, 1])
    }
}
