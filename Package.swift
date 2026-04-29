// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftMLKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "SwiftMLKit",
            targets: ["SwiftMLKit"]
        ),
        .executable(
            name: "SwiftMLKitExample",
            targets: ["SwiftMLKitExample"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3")
    ],
    targets: [
        // MARK: - SwiftMLKit (ML framework)
        .target(
            name: "SwiftMLKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),

        // MARK: - Example
        .executableTarget(
            name: "SwiftMLKitExample",
            dependencies: [
                "SwiftMLKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            exclude: [
                "README.md"
            ],
            resources: [
                .process("Resources")
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "SwiftMLKitTests",
            dependencies: [
                "SwiftMLKit"
            ],
            path: "Tests/SwiftMLKitTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
