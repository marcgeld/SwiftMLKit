//
//  DecisionTree.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//


import MLX

// MARK: - Decision Tree (CART, classification)

public struct DecisionTree: Classifier {

    final class Node {
        var feature: Int?
        var threshold: Float?
        var left: Node?
        var right: Node?
        var value: Float?

        init(value: Float) {
            self.value = value
        }

        init(feature: Int, threshold: Float, left: Node, right: Node) {
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    private var root: Node?
    public let maxDepth: Int
    public let minSamplesSplit: Int

    public init(maxDepth: Int = 5, minSamplesSplit: Int = 2) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
    }

    // MARK: - Fit

    public mutating func fit(X: MLXArray, y: MLXArray) {
        root = buildTree(X: X, y: y, depth: 0)
    }

    // MARK: - Predict
    public func predict(X: MLXArray) -> MLXArray {
        let preds = (0..<X.shape[0]).map { i -> Float in
            predictSample(x: X[i], node: root!)
        }
        return MLXArray(preds).reshaped([X.shape[0], 1])
    }

    // MARK: - Tree building
    private func buildTree(X: MLXArray, y: MLXArray, depth: Int) -> Node {

        let nSamples = X.shape[0]

        let yArr = y.flattened().asArray(Float.self)
        let majority = yArr.reduce(into: [:]) { $0[$1, default: 0] += 1 }
            .max { $0.value < $1.value }!.key

        // stopping conditions
        if depth >= maxDepth || nSamples < minSamplesSplit || Set(yArr).count == 1 {
            return Node(value: majority)
        }

        let (bestFeature, bestThreshold) = bestSplit(X: X, y: y)

        guard let f = bestFeature else {
            return Node(value: majority)
        }

        let leftMask = X[0..<nSamples, f] .<= bestThreshold!
        let rightMask = X[0..<nSamples, f] .> bestThreshold!

        let Xleft = X[leftMask]
        let yleft = y[leftMask]

        let Xright = X[rightMask]
        let yright = y[rightMask]

        if Xleft.shape[0] == 0 || Xright.shape[0] == 0 {
            return Node(value: majority)
        }

        let leftNode = buildTree(X: Xleft, y: yleft, depth: depth + 1)
        let rightNode = buildTree(X: Xright, y: yright, depth: depth + 1)

        return Node(feature: f, threshold: bestThreshold!, left: leftNode, right: rightNode)
    }

    private func bestSplit(X: MLXArray, y: MLXArray) -> (Int?, Float?) {

        let nFeatures = X.shape[1]
        var bestGini: Float = Float.infinity
        var bestFeature: Int?
        var bestThreshold: Float?

        for f in 0..<nFeatures {
            let values = X[0..<X.shape[0], f].asArray(Float.self)

            for t in values {

                let leftMask = X[0..<X.shape[0], f] .<= t
                let rightMask = X[0..<X.shape[0], f] .> t

                let yLeft = y[leftMask]
                let yRight = y[rightMask]

                let g = gini(yLeft) * Float(yLeft.shape[0]) +
                        gini(yRight) * Float(yRight.shape[0])

                if g < bestGini {
                    bestGini = g
                    bestFeature = f
                    bestThreshold = t
                }
            }
        }

        return (bestFeature, bestThreshold)
    }

    private func gini(_ y: MLXArray) -> Float {
        let arr = y.flattened().asArray(Float.self)
        let total = Float(arr.count)
        guard total > 0 else { return 0 }

        let counts = arr.reduce(into: [Float: Int]()) { $0[$1, default: 0] += 1 }

        var sumSq: Float = 0
        for count in counts.values {
            let p = Float(count) / total
            sumSq += p * p
        }
        return 1.0 - sumSq
    }

    private func predictSample(x: MLXArray, node: Node) -> Float {
        if let value = node.value {
            return value
        }

        if x[node.feature!].item(Float.self) <= node.threshold! {
            return predictSample(x: x, node: node.left!)
        } else {
            return predictSample(x: x, node: node.right!)
        }
    }
}
