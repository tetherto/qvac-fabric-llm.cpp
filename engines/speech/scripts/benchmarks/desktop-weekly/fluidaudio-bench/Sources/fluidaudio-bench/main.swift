import CoreML
import FluidAudio
import Foundation

let modelTdtV3 = "tdt-v3"
let modelUnified = "unified"
let backendName = "CoreML"
let nanosecondsPerMillisecond = 1_000_000.0
let millisecondsPerSecond = 1000.0
let sampleRate = 16000.0

struct Options {
    var model = modelTdtV3
    var wav = ""
    var modelsDir: URL?
    var runs = 5
    var warmup = 1
    var jsonOut = ""
    var encoder = "ane"
    var revision = ""
}

func usage() -> Never {
    FileHandle.standardError.write(Data("usage: fluidaudio-bench --model tdt-v3|unified --wav FILE --bench-json OUT [--models-dir DIR] [--bench-runs N] [--bench-warmup N] [--encoder ane|gpu|cpu] [--revision SHA]\n".utf8))
    exit(2)
}

func parseOptions(_ arguments: [String]) -> Options {
    var options = Options()
    var index = 0
    while index + 1 < arguments.count {
        let value = arguments[index + 1]
        switch arguments[index] {
        case "--model": options.model = value
        case "--wav": options.wav = value
        case "--models-dir": options.modelsDir = URL(fileURLWithPath: value)
        case "--bench-runs": options.runs = Int(value) ?? options.runs
        case "--bench-warmup": options.warmup = Int(value) ?? options.warmup
        case "--bench-json": options.jsonOut = value
        case "--encoder": options.encoder = value
        case "--revision": options.revision = value
        default: usage()
        }
        index += 2
    }
    if options.wav.isEmpty || options.jsonOut.isEmpty { usage() }
    return options
}

func computeUnits(_ name: String) -> MLComputeUnits {
    switch name {
    case "gpu": return .cpuAndGPU
    case "cpu": return .cpuOnly
    default: return .cpuAndNeuralEngine
    }
}

protocol Transcriber {
    func transcribe(_ samples: [Float]) async throws -> String
}

final class TdtTranscriber: Transcriber {
    let manager: AsrManager
    let decoderLayers: Int

    init(options: Options) async throws {
        let models = try await AsrModels.downloadAndLoad(
            to: options.modelsDir, version: .v3, encoderComputeUnits: computeUnits(options.encoder))
        manager = AsrManager()
        try await manager.loadModels(models)
        decoderLayers = models.version.decoderLayers
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        var state = try TdtDecoderState(decoderLayers: decoderLayers)
        return try await manager.transcribe(samples, decoderState: &state).text
    }
}

final class UnifiedTranscriber: Transcriber {
    let manager = UnifiedAsrManager()

    init(options: Options) async throws {
        try await manager.loadModels(to: options.modelsDir, configuration: nil, progressHandler: nil)
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        try await manager.transcribe(samples)
    }
}

func makeTranscriber(_ options: Options) async throws -> Transcriber {
    if options.model == modelUnified { return try await UnifiedTranscriber(options: options) }
    if options.model == modelTdtV3 { return try await TdtTranscriber(options: options) }
    usage()
}

func nowMilliseconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / nanosecondsPerMillisecond
}

func timed(_ transcriber: Transcriber, _ samples: [Float]) async throws -> (Double, String) {
    let start = nowMilliseconds()
    let text = try await transcriber.transcribe(samples)
    return (nowMilliseconds() - start, text)
}

func warmUp(_ transcriber: Transcriber, _ samples: [Float], _ count: Int) async throws {
    for _ in 0..<count { _ = try await transcriber.transcribe(samples) }
}

func timedRuns(_ transcriber: Transcriber, _ samples: [Float], _ count: Int) async throws -> ([Double], String) {
    var times: [Double] = []
    var text = ""
    for _ in 0..<count {
        let (ms, out) = try await timed(transcriber, samples)
        times.append(ms)
        text = out
    }
    return (times, text)
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

func round3(_ value: Double) -> Double { (value * 1000).rounded() / 1000 }

func benchJson(_ options: Options, loadMs: Double, times: [Double], text: String, audioSeconds: Double) -> [String: Any] {
    let medianMs = median(times)
    return [
        "engine": "FluidAudio", "backend": backendName, "encoder_backend": "coreml-\(options.encoder)", "model": options.model,
        "engine_commit": options.revision, "load_ms": round3(loadMs),
        "inference_ms": ["median": round3(medianMs), "min": round3(times.min() ?? 0), "max": round3(times.max() ?? 0), "all": times.map(round3)],
        "rtf_median": medianMs / millisecondsPerSecond / audioSeconds, "audio_seconds": audioSeconds,
        "runs": times.count, "warmup": options.warmup, "transcript": text.trimmingCharacters(in: .whitespacesAndNewlines),
    ]
}

func writeJson(_ payload: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path))
}

func summaryLine(_ payload: [String: Any]) -> String {
    let inference = payload["inference_ms"] as? [String: Any] ?? [:]
    return "backend: \(backendName)\ninference_ms median \(inference["median"] ?? 0) rtf \(payload["rtf_median"] ?? 0): \(payload["transcript"] ?? "")"
}

func run() async throws {
    let options = parseOptions(Array(CommandLine.arguments.dropFirst()))
    let samples = try AudioConverter().resampleAudioFile(path: options.wav)
    let loadStart = nowMilliseconds()
    let transcriber = try await makeTranscriber(options)
    let loadMs = nowMilliseconds() - loadStart
    try await warmUp(transcriber, samples, options.warmup)
    let (times, text) = try await timedRuns(transcriber, samples, options.runs)
    let payload = benchJson(options, loadMs: loadMs, times: times, text: text, audioSeconds: Double(samples.count) / sampleRate)
    try writeJson(payload, to: options.jsonOut)
    print(summaryLine(payload))
}

let group = DispatchGroup()
group.enter()
Task {
    do { try await run() } catch {
        FileHandle.standardError.write(Data("fluidaudio-bench failed: \(error)\n".utf8))
        exit(1)
    }
    group.leave()
}
group.wait()
