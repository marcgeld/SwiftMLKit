//
//  RandomForest.swift
//  SwiftMLKit
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

public struct RandomForest: Classifier {

    private var trees: [DecisionTree] = []
    public let nTrees: Int
    public let maxDepth: Int

    public init(nTrees: Int = 10, maxDepth: Int = 5) {
        self.nTrees = nTrees
        self.maxDepth = maxDepth
    }

    public mutating func fit(X: MLXArray, y: MLXArray) {

        trees = []

        for _ in 0..<nTrees {

            let bootIdx = MLXArray((0..<X.shape[0]).map { _ in Int32.random(in: 0..<Int32(X.shape[0])) })
            let Xb = X[bootIdx]
            let yb = y[bootIdx]

            var tree = DecisionTree(maxDepth: maxDepth)
            tree.fit(X: Xb, y: yb)

            trees.append(tree)
        }
    }

    public func predict(X: MLXArray) -> MLXArray {

        let preds = trees.map { $0.predict(X: X) }
        let stacked = MLX.stacked(preds, axis: 1)

        let mean = stacked.mean(axis: 1)

        return `where`(mean .> 0.5, MLXArray(1), MLXArray(0))
            .reshaped([X.shape[0], 1])
    }
}
