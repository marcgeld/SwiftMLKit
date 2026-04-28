//
//  ExtraTrees.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//


import MLX

public struct ExtraTrees: Classifier {

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

            var tree = DecisionTree(maxDepth: maxDepth)
            tree.fit(X: X, y: y) // no bootstrap

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