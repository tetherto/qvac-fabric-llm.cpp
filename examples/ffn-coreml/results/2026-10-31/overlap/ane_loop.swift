// Neural Engine load: runs the two FFN partitions of one layer on two threads for a fixed time and reports ms per call.
// usage: ane_loop <layer dir> <seconds>
import CoreML
import Foundation

let dir = CommandLine.arguments[1]
let seconds = Double(CommandLine.arguments[2])!
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
let names = ["ffn-512.mlmodelc", "ffn-512-1.mlmodelc"]
let models = try names.map { try MLModel(contentsOf: URL(fileURLWithPath: dir + "/" + $0), configuration: config) }
let x = try MLMultiArray(shape: [512, 5120], dataType: .float32)
for i in 0..<x.count { x[i] = NSNumber(value: Float((i % 97) - 48) * 0.01) }
let input = try MLDictionaryFeatureProvider(dictionary: ["x": x])
for m in models { _ = try m.prediction(from: input) }

var results = [(Int, Double)](repeating: (0, 0), count: models.count)
let group = DispatchGroup()
for (k, m) in models.enumerated() {
    group.enter()
    Thread.detachNewThread {
        let start = Date()
        var n = 0
        while Date().timeIntervalSince(start) < seconds {
            _ = try! m.prediction(from: input)
            n += 1
        }
        results[k] = (n, Date().timeIntervalSince(start) * 1000 / Double(n))
        group.leave()
    }
}
group.wait()
for (k, r) in results.enumerated() { print(String(format: "ane partition=%d calls=%d ms_per_call=%.3f", k, r.0, r.1)) }
