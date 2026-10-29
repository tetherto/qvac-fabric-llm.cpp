#include "ffn-coreml-common.h"

int main(int argc, char ** argv) {
    bool ane_only = false, gguf_reference = false, split = false, cpu_only = false, plan_requested = false;
    int nargs = argc;
    while (nargs > 1) {
        if (strcmp(argv[nargs - 1], "--ane-only") == 0) { ane_only = true; }
        else if (strcmp(argv[nargs - 1], "--gguf-reference") == 0) { gguf_reference = true; }
        else if (strcmp(argv[nargs - 1], "--split") == 0) { split = true; }
        else if (strcmp(argv[nargs - 1], "--cpu-only") == 0) { cpu_only = true; }
        else if (strcmp(argv[nargs - 1], "--plan") == 0) { plan_requested = true; }
        else { break; }
        --nargs;
    }
    if ((nargs != 6 && nargs != 8) || (split && nargs != 8)) {
        fprintf(stderr, "usage: %s model.gguf assets tokens repetitions layer [input.f32 reference.f32] [--ane-only] [--gguf-reference] [--split] [--cpu-only] [--plan]\n", argv[0]);
        return 1;
    }
    if (split) { ane_only = true; gguf_reference = true; }
    @autoreleasepool {
        try {
            const std::string assets = argv[2];
            int tokens = std::stoi(argv[3]), reps = std::stoi(argv[4]), layer = std::stoi(argv[5]);
            require(tokens > 0 && tokens <= 512 && reps > 0 && reps <= 1000 && layer >= 0 && layer < 256, "invalid arguments");
            ggml_backend_load_all();
            printf("reference=%s\n", gguf_reference ? "metal-gguf" : "metal-f16");
            std::vector<float> x, reference, actual;
            int64_t source_channels = 0;
            for (bool f16 : {true, false}) {
                if (f16 && gguf_reference) { continue; }
                metal_ffn metal;
                metal.init(argv[1], assets, layer, tokens, f16);
                source_channels = metal.channels;
                if (x.empty()) {
                    x.resize(ggml_nelements(metal.input));
                    reference.resize(x.size());
                    actual.resize(x.size());
                    if (nargs == 8) {
                        read_at(argv[6], 0, x.data(), x.size() * sizeof(float));
                    } else {
                        std::mt19937 random(11259);
                        std::normal_distribution<float> normal(0.0f, 1.0f);
                        for (auto & v : x) { v = normal(random); }
                    }
                    for (float value : x) { require(std::isfinite(value), "non-finite input"); }
                }
                measure(f16 ? "ggml-metal-f16" : "ggml-metal-gguf", tokens, reps, [&]() { metal.predict(x, actual); });
                if (f16 || gguf_reference) { reference = actual; }
                if (!f16) {
                    if (!gguf_reference) { require(compare("gguf-vs-f16", reference, actual), "Metal precision control failed"); }
                    if (nargs == 8) {
                        std::vector<float> captured(actual.size());
                        read_at(argv[7], 0, captured.data(), captured.size() * sizeof(float));
                        require(compare("metal-vs-captured-decoder", captured, actual), "FFN replay differs from decoder capture");
                    }
                }
            }
            metal_ffn bridge;
            metal_boundary boundary;
            if (nargs == 8) {
                bridge.init(argv[1], assets, layer, tokens, false);
                ggml_backend_tensor_set(bridge.input, x.data(), 0, x.size() * sizeof(float));
                measure("ggml-metal-gguf-resident", tokens, reps, [&]() {
                    require(ggml_backend_graph_compute(bridge.backend, bridge.graph) == GGML_STATUS_SUCCESS, "resident Metal compute failed");
                    ggml_backend_synchronize(bridge.backend);
                });
                ggml_backend_tensor_get(bridge.output, actual.data(), 0, actual.size() * sizeof(float));
                require(compare(gguf_reference ? "resident-metal-vs-gguf" : "resident-metal-vs-f16", reference, actual), "resident Metal replay failed");
                boundary.init(bridge.backend, x.size() / tokens, tokens, x);
            }
            NSError * error = nil;
            NSString * path = [NSString stringWithFormat:@"%s/ffn-%d.mlpackage", argv[2], tokens];
            temporary_model compiled{[MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&error]};
            require(compiled.url, error ? error.localizedDescription.UTF8String : "Core ML compilation failed");
            bool accuracy_ok = true;
            for (auto units : {MLComputeUnitsCPUAndGPU, MLComputeUnitsCPUAndNeuralEngine, MLComputeUnitsAll, MLComputeUnitsCPUOnly}) {
                if (cpu_only) {
                    if (units != MLComputeUnitsCPUOnly) { continue; }
                } else if (units == MLComputeUnitsCPUOnly || (ane_only && units != MLComputeUnitsCPUAndNeuralEngine)) { continue; }
                MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
                config.computeUnits = units;
                const char * name = units == MLComputeUnitsCPUOnly ? "coreml-cpu-only" : units == MLComputeUnitsCPUAndGPU ? "coreml-cpu-gpu" :
                                   units == MLComputeUnitsCPUAndNeuralEngine ? "coreml-cpu-ane" : "coreml-all";
                printf("compute_configuration=%s\n", name);
                print_memory(name, "before_load");
                MLModel * model = [MLModel modelWithContentsOfURL:compiled.url configuration:config error:&error];
                require(model, error ? error.localizedDescription.UTF8String : "Core ML load failed");
                NSDictionary * metadata = model.modelDescription.metadata[MLModelCreatorDefinedKey];
                require(!metadata[@"source_layer"] || [metadata[@"source_layer"] intValue] == layer, "Core ML package is for a different layer");
                int channel_start = [metadata[@"channel_start"] intValue];
                int channel_count = [metadata[@"channel_count"] intValue];
                if (split) {
                    require(channel_start > 0 && channel_start < bridge.channels && channel_count > 0 && channel_count < bridge.channels && channel_start + channel_count == bridge.channels, "invalid Core ML suffix channel range");
                    printf("split_channels gpu=%d ane=%d\n", channel_start, channel_count);
                } else { require(channel_start == 0 && (channel_count == 0 || channel_count == source_channels), "partial package requires --split"); }
                double output_scale = metadata[@"output_scale"] ? [metadata[@"output_scale"] doubleValue] : 1.0;
                require(std::isfinite(output_scale) && output_scale >= 1 && output_scale <= 256, "invalid output scale");
                const float inverse_scale = 1.0f / output_scale;
                int hidden = int(x.size() / tokens);
                MLMultiArray * input = [[MLMultiArray alloc] initWithShape:@[@(tokens), @(hidden)] dataType:MLMultiArrayDataTypeFloat32 error:&error];
                require(input, "Core ML input allocation failed");
                MLDictionaryFeatureProvider * features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": input} error:&error];
                require(features, "Core ML feature provider failed");
                auto predict = [&]() {
                    @autoreleasepool {
                        memcpy(input.dataPointer, x.data(), x.size() * sizeof(float));
                        NSError * prediction_error = nil;
                        id<MLFeatureProvider> result = [model predictionFromFeatures:features error:&prediction_error];
                        require(result, prediction_error ? prediction_error.localizedDescription.UTF8String : "prediction failed");
                        MLMultiArray * output = [result featureValueForName:@"y"].multiArrayValue;
                        require(output && output.dataType == MLMultiArrayDataTypeFloat32 && output.shape.count == 2 &&
                                output.shape[0].intValue == tokens && output.shape[1].intValue == hidden, "unexpected Core ML output");
                        size_t row_stride = output.strides[0].unsignedLongLongValue;
                        size_t col_stride = output.strides[1].unsignedLongLongValue;
                        const float * data = static_cast<const float *>(output.dataPointer);
                        for (int t = 0; t < tokens; ++t) {
                            for (int h = 0; h < hidden; ++h) { actual[t * hidden + h] = data[t * row_stride + h * col_stride] * inverse_scale; }
                        }
                    }
                };
                if (split) {
                    if (!measure_split(argv[1], assets, layer, tokens, reps, channel_start, bridge, x, reference, actual, predict)) { accuracy_ok = false; }
                    if (plan_requested) { report_placement(compiled.url, config, name); }
                    continue;
                }
                measure(name, tokens, reps, predict);
                print_memory(name, "after_predictions");
                if (!compare(name, reference, actual)) { accuracy_ok = false; }
                if (nargs == 8 && units == MLComputeUnitsCPUAndNeuralEngine) {
                    measure("coreml-cpu-ane-metal-roundtrip", tokens, reps, [&]() {
                        ggml_backend_synchronize(bridge.backend);
                        ggml_backend_tensor_get(boundary.input, x.data(), 0, x.size() * sizeof(float));
                        predict();
                        ggml_backend_tensor_set(boundary.output, actual.data(), 0, actual.size() * sizeof(float));
                        ggml_backend_synchronize(bridge.backend);
                    });
                    ggml_backend_tensor_get(boundary.output, actual.data(), 0, actual.size() * sizeof(float));
                    if (!compare("coreml-metal-roundtrip", reference, actual)) { accuracy_ok = false; }
                }
                if (ane_only && !plan_requested) {
                    printf("placement=%s skipped=ane-only\n", name);
                    continue;
                }
                report_placement(compiled.url, config, name);
            }
            if (!accuracy_ok) { return 2; }
        } catch (const std::exception & error) {
            fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
    return 0;
}
