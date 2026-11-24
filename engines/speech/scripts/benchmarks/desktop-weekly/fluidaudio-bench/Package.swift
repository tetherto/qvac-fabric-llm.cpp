// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "fluidaudio-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio", revision: "ee78635d23d52379117554ba72358b65a0612447")
    ],
    targets: [
        .executableTarget(
            name: "fluidaudio-bench",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/fluidaudio-bench"
        )
    ]
)
