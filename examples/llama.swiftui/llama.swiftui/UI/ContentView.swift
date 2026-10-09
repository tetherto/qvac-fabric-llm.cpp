import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var multiLineText = ""
    @State private var showingHelp = false    // To track if Help Sheet should be shown

    var body: some View {
        NavigationView {
            VStack {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }

                TextEditor(text: $multiLineText)
                    .frame(height: 80)
                    .padding()
                    .border(Color.gray, width: 0.5)

                HStack {
                    Button("Send") {
                        sendText()
                    }

                    Button("Bench") {
                        bench()
                    }

                    Button(llamaState.isFinetuning ? "Finetuning..." : "Finetune") {
                        finetune()
                    }
                    .disabled(llamaState.isFinetuning)

                    Button("Clear") {
                        clear()
                    }

                    Button("Copy") {
                        UIPasteboard.general.string = llamaState.messageLog
                    }
                }
                .buttonStyle(.bordered)
                .padding()

                NavigationLink(destination: SettingsView(llamaState: llamaState)) {
                    Text("Settings")
                }
                .padding()

            }
            .padding()
            .navigationBarTitle("Settings", displayMode: .inline)

        }
    }

    func sendText() {
        Task {
            await llamaState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await llamaState.bench()
        }
    }

    func finetune() {
        Task {
            await llamaState.finetune()
        }
    }

    func clear() {
        Task {
            await llamaState.clear()
        }
    }
    struct SettingsView: View {

        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false
        var body: some View {
            List {
                Section(header: Text("Dataset for Finetuning")) {
                    if let selectedFilename = llamaState.selectedDatasetFilename,
                       let dataset = llamaState.downloadedDatasets.first(where: { $0.filename == selectedFilename }) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(dataset.name)
                                .font(.body)
                            Text(dataset.filename)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("No dataset selected. Download or choose one below.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }

                if !llamaState.downloadedDatasets.isEmpty {
                    Section(header: Text("Downloaded Datasets")) {
                        ForEach(llamaState.downloadedDatasets) { dataset in
                            DatasetDownloadedRow(llamaState: llamaState, dataset: dataset)
                        }
                        .onDelete { offsets in
                            llamaState.deleteDownloadedDatasets(at: offsets)
                        }
                    }
                }

                if !llamaState.availableDatasets.isEmpty {
                    Section(header: Text("Available Datasets")) {
                        ForEach(llamaState.availableDatasets) { dataset in
                            DatasetAvailableRow(llamaState: llamaState, dataset: dataset)
                        }
                    }
                }

                Section(header: Text("Download Dataset From URL")) {
                    DatasetURLInputRow(llamaState: llamaState)
                }
                Section(header: Text("Model for Inference")) {
                    if let selectedFilename = llamaState.selectedModelFilename,
                       let model = llamaState.downloadedModels.first(where: { $0.filename == selectedFilename }) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.name)
                                .font(.body)
                            Text(model.filename)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else if let selectedFilename = llamaState.selectedModelFilename {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(selectedFilename)
                                .font(.body)
                            Text("Bundle resource")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("No model loaded. Download or choose one below.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }

                if !llamaState.downloadedModels.isEmpty {
                    Section(header: Text("Downloaded Models")) {
                        ForEach(llamaState.downloadedModels) { model in
                            ModelDownloadedRow(llamaState: llamaState, model: model)
                        }
                        .onDelete { offsets in
                            llamaState.deleteDownloadedModels(at: offsets)
                        }
                    }
                }

                if !llamaState.undownloadedModels.isEmpty {
                    Section(header: Text("Available Models")) {
                        ForEach(llamaState.undownloadedModels) { model in
                            ModelAvailableRow(llamaState: llamaState, model: model)
                        }
                    }
                }

                Section(header: Text("Download Model From URL")) {
                    ModelURLInputRow(llamaState: llamaState)
                }

                Section(header: Text("Finetuning & Runtime Options")) {
                    Stepper(value: $llamaState.optionNGpuLayers, in: -1...256, step: 1) {
                        HStack {
                            Text("n-gpu-layers (ngl)")
                            Spacer()
                            if llamaState.optionNGpuLayers >= 0 {
                                Text("\(llamaState.optionNGpuLayers)")
                                    .foregroundColor(.secondary)
                            } else {
                                Text("Auto")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    Stepper(value: $llamaState.optionContextLength, in: 32...8192, step: 64) {
                        HStack {
                            Text("Context size (c)")
                            Spacer()
                            Text("\(llamaState.optionContextLength)")
                                .foregroundColor(.secondary)
                        }
                    }

                    Stepper(value: $llamaState.optionSeed, in: 0...1_000_000, step: 1) {
                        HStack {
                            Text("Seed (s)")
                            Spacer()
                            Text("\(llamaState.optionSeed)")
                                .foregroundColor(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Temperature (temp)")
                            Spacer()
                            Text(String(format: "%.2f", llamaState.optionTemperature))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $llamaState.optionTemperature, in: 0...0)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Top-p")
                            Spacer()
                            Text(String(format: "%.2f", llamaState.optionTopP))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $llamaState.optionTopP, in: 0.01...1.0, step: 0.01)
                    }

                    Stepper(value: $llamaState.optionTopK, in: 1...500, step: 1) {
                        HStack {
                            Text("Top-k")
                            Spacer()
                            Text("\(llamaState.optionTopK)")
                                .foregroundColor(.secondary)
                        }
                    }

                    Toggle(isOn: $llamaState.optionFlashAttention) {
                        Text("Flash Attention")
                    }

                    Toggle(isOn: $llamaState.optionSingleTurn) {
                        Text("Single Turn (st)")
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Settings", displayMode: .inline).toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }.sheet(isPresented: $showingHelp) {    // Sheet for help modal
                NavigationView {
                    VStack(alignment: .leading) {
                        VStack(alignment: .leading) {
                            Text("1. Make sure the model is in GGUF Format")
                                    .padding()
                            Text("2. Copy the download link of the quantized model")
                                    .padding()
                        }
                        Spacer()
                    }
                    .navigationTitle("Help")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") {
                                showingHelp = false
                            }
                        }
                    }
                }
            }
        }
    }
}

struct DatasetDownloadedRow: View {
    @ObservedObject var llamaState: LlamaState
    let dataset: Dataset

    private var isSelected: Bool {
        llamaState.selectedDatasetFilename == dataset.filename
    }

    var body: some View {
        Button {
            llamaState.selectDataset(dataset)
        } label: {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(dataset.name)
                        .foregroundColor(.primary)
                    Text(dataset.filename)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.accentColor)
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
    }
}

struct DatasetAvailableRow: View {
    @ObservedObject var llamaState: LlamaState
    let dataset: Dataset
    @State private var status: Status = .idle

    private enum Status: Equatable {
        case idle
        case downloading
        case failed(String)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(dataset.name)
                .font(.body)
            Text(dataset.url)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)

            switch status {
            case .idle:
                Button("Download") {
                    startDownload()
                }
                .buttonStyle(.borderedProminent)
            case .downloading:
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Downloading…")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            case .failed(let message):
                VStack(alignment: .leading, spacing: 4) {
                    Text("Download failed: \(message)")
                        .font(.caption)
                        .foregroundColor(.red)
                    Button("Retry") {
                        status = .idle
                        startDownload()
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func startDownload() {
        guard status != .downloading else { return }
        guard let downloadURL = URL(string: dataset.url) else {
            status = .failed("Invalid URL")
            return
        }

        status = .downloading

        Task {
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: downloadURL)
                guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                    await MainActor.run {
                        status = .failed("Server error")
                    }
                    return
                }

                let destination = await MainActor.run {
                    llamaState.datasetFileURL(for: dataset.filename)
                }

                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: temporaryURL, to: destination)

                await MainActor.run {
                    llamaState.registerDownloadedDataset(dataset)
                    status = .idle
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    status = .failed(error.localizedDescription)
                }
            }
        }
    }
}

struct DatasetURLInputRow: View {
    @ObservedObject var llamaState: LlamaState
    @State private var urlText: String = ""
    @State private var status: Status = .idle

    private enum Status: Equatable {
        case idle
        case downloading
        case failed(String)
        case success
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("https://example.com/dataset.jsonl", text: $urlText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocorrectionDisabled(true)
                .textInputAutocapitalization(.never)

            switch status {
            case .failed(let message):
                Text(message)
                    .font(.caption)
                    .foregroundColor(.red)
            case .success:
                Text("Download complete!")
                    .font(.caption)
                    .foregroundColor(.green)
            default:
                EmptyView()
            }

            Button(action: startDownload) {
                if status == .downloading {
                    HStack {
                        ProgressView()
                        Text("Downloading…")
                    }
                } else {
                    Text("Download Dataset")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(status == .downloading || urlText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.vertical, 4)
    }

    private func startDownload() {
        guard status != .downloading else { return }
        let trimmed = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let downloadURL = URL(string: trimmed), !trimmed.isEmpty else {
            status = .failed("Enter a valid dataset URL")
            return
        }

        let filename = deriveFilename(from: downloadURL)
        let displayName = deriveDisplayName(from: filename)
        let dataset = Dataset(name: displayName, url: trimmed, filename: filename, status: "download")

        status = .downloading

        Task {
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: downloadURL)
                guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                    await MainActor.run {
                        status = .failed("Server error")
                    }
                    return
                }

                let destination = await MainActor.run {
                    llamaState.datasetFileURL(for: dataset.filename)
                }

                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: temporaryURL, to: destination)

                await MainActor.run {
                    llamaState.registerDownloadedDataset(dataset)
                    status = .success
                    urlText = ""
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    status = .failed(error.localizedDescription)
                }
            }
        }
    }

    private func deriveFilename(from url: URL) -> String {
        let candidate = url.lastPathComponent
        if candidate.isEmpty {
            return "dataset-\(UUID().uuidString).txt"
        }
        return candidate
    }

    private func deriveDisplayName(from filename: String) -> String {
        let decoded = filename.removingPercentEncoding ?? filename
        let baseName = (decoded as NSString).deletingPathExtension
        let cleaned = baseName.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Custom Dataset"
        }
        return trimmed.capitalized
    }
}

struct ModelDownloadedRow: View {
    @ObservedObject var llamaState: LlamaState
    let model: Model

    private var isSelected: Bool {
        llamaState.selectedModelFilename == model.filename
    }

    var body: some View {
        Button {
            llamaState.selectDownloadedModel(model)
        } label: {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name)
                        .foregroundColor(.primary)
                    Text(model.filename)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.accentColor)
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
    }
}

struct ModelAvailableRow: View {
    @ObservedObject var llamaState: LlamaState
    let model: Model
    @State private var status: Status = .idle

    private enum Status: Equatable {
        case idle
        case downloading
        case failed(String)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(model.name)
                .font(.body)
            Text(model.url)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)

            switch status {
            case .idle:
                Button("Download") {
                    startDownload()
                }
                .buttonStyle(.borderedProminent)
            case .downloading:
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Downloading…")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            case .failed(let message):
                VStack(alignment: .leading, spacing: 4) {
                    Text("Download failed: \(message)")
                        .font(.caption)
                        .foregroundColor(.red)
                    Button("Retry") {
                        status = .idle
                        startDownload()
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func startDownload() {
        guard status != .downloading else { return }
        guard let downloadURL = URL(string: model.url) else {
            status = .failed("Invalid URL")
            return
        }

        status = .downloading

        Task {
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: downloadURL)
                guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                    await MainActor.run {
                        status = .failed("Server error")
                    }
                    return
                }

                let destination = await MainActor.run {
                    llamaState.modelFileURL(for: model.filename)
                }

                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: temporaryURL, to: destination)

                await MainActor.run {
                    llamaState.registerDownloadedModel(model)
                    status = .idle
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    status = .failed(error.localizedDescription)
                }
            }
        }
    }
}

struct ModelURLInputRow: View {
    @ObservedObject var llamaState: LlamaState
    @State private var urlText: String = ""
    @State private var status: Status = .idle

    private enum Status: Equatable {
        case idle
        case downloading
        case failed(String)
        case success
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("https://example.com/model.gguf", text: $urlText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocorrectionDisabled(true)
                .textInputAutocapitalization(.never)

            switch status {
            case .failed(let message):
                Text(message)
                    .font(.caption)
                    .foregroundColor(.red)
            case .success:
                Text("Download complete!")
                    .font(.caption)
                    .foregroundColor(.green)
            default:
                EmptyView()
            }

            Button(action: startDownload) {
                if status == .downloading {
                    HStack {
                        ProgressView()
                        Text("Downloading…")
                    }
                } else {
                    Text("Download Model")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(status == .downloading || urlText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.vertical, 4)
    }

    private func startDownload() {
        guard status != .downloading else { return }
        let trimmed = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let downloadURL = URL(string: trimmed), !trimmed.isEmpty else {
            status = .failed("Enter a valid model URL")
            return
        }

        let filename = deriveFilename(from: downloadURL)
        let displayName = deriveDisplayName(from: filename)
        let model = Model(name: displayName, url: trimmed, filename: filename, status: "download")

        status = .downloading

        Task {
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: downloadURL)
                guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                    await MainActor.run {
                        status = .failed("Server error")
                    }
                    return
                }

                let destination = await MainActor.run {
                    llamaState.modelFileURL(for: model.filename)
                }

                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: temporaryURL, to: destination)

                await MainActor.run {
                    llamaState.registerDownloadedModel(model)
                    status = .success
                    urlText = ""
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    status = .failed(error.localizedDescription)
                }
            }
        }
    }

    private func deriveFilename(from url: URL) -> String {
        let candidate = url.lastPathComponent
        if candidate.isEmpty {
            return "model-\(UUID().uuidString).gguf"
        }
        return candidate
    }

    private func deriveDisplayName(from filename: String) -> String {
        let decoded = filename.removingPercentEncoding ?? filename
        let baseName = (decoded as NSString).deletingPathExtension
        let cleaned = baseName.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Custom Model"
        }
        return trimmed.capitalized
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
