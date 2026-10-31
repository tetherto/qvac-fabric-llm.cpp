// Times the DeltaNet input projection split: Metal rows [0, R) plus two Core ML row slices in parallel.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

static const int T = 512, H = 5120;

static double median_ms(int reps, const std::function<void()> & f) {
    for (int i = 0; i < 3; ++i) f();
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        auto s = std::chrono::steady_clock::now(); f();
        v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static MLModel * load(const char * dir) {
    NSError * e = nil;
    NSURL * c = [MLModel compileModelAtURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s/ffn-512.mlpackage", dir]] error:&e];
    MLModelConfiguration * cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel * m = [MLModel modelWithContentsOfURL:c configuration:cfg error:&e];
    if (!m) { fprintf(stderr, "load failed %s\n", dir); exit(1); }
    return m;
}

struct slice { MLModel * model; MLDictionaryFeatureProvider * features; int start, rows; };

static void predict(const slice & s, float * out, int out_rows) {
    @autoreleasepool {
        NSError * e = nil;
        id<MLFeatureProvider> r = [s.model predictionFromFeatures:s.features error:&e];
        MLMultiArray * y = [r featureValueForName:@"y"].multiArrayValue;
        const float * d = (const float *) y.dataPointer;
        size_t rs = y.strides[0].unsignedLongValue, cs = y.strides[1].unsignedLongValue;
        for (int t = 0; t < T; ++t) for (int c = 0; c < s.rows; ++c) out[(size_t) t * out_rows + s.start + c] = d[t * rs + c * cs];
    }
}

static double rel_l2(const std::vector<float> & ref, const std::vector<float> & got, int total_rows, int lo, int hi) {
    double err = 0, norm = 0;
    for (int t = 0; t < T; ++t) for (int c = lo; c < hi; ++c) {
        double r = ref[(size_t) t * total_rows + c], d = r - got[(size_t) t * total_rows + c];
        err += d * d; norm += r * r;
    }
    return std::sqrt(err / norm);
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        const std::string path = argv[1];
        const int layer = atoi(argv[2]), gpu_rows = atoi(argv[3]);
        ggml_backend_load_all();
        ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_reg_by_name("MTL"), 0), nullptr);
        ggml_context * meta = nullptr;
        gguf_context * header = gguf_init_from_file(path.c_str(), {true, &meta});
        ggml_context * wctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
        std::string names[2] = {"blk." + std::to_string(layer) + ".attn_qkv.weight", "blk." + std::to_string(layer) + ".attn_gate.weight"};
        ggml_tensor * w[2];
        for (int i = 0; i < 2; ++i) { auto * s = ggml_get_tensor(meta, names[i].c_str()); w[i] = ggml_new_tensor_2d(wctx, s->type, s->ne[0], s->ne[1]); }
        auto * wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        for (int i = 0; i < 2; ++i) {
            std::vector<char> b(ggml_nbytes(w[i]));
            std::ifstream in(path, std::ios::binary);
            in.seekg(gguf_get_data_offset(header) + gguf_get_tensor_offset(header, gguf_find_tensor(header, names[i].c_str())));
            in.read(b.data(), b.size());
            ggml_backend_tensor_set(w[i], b.data(), 0, b.size());
        }
        const int qkv_rows = w[0]->ne[1], total = qkv_rows + w[1]->ne[1];
        ggml_context * g = ggml_init({ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead(), nullptr, true});
        auto * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, H, T); ggml_set_input(x);
        auto * ref_qkv = ggml_mul_mat(g, w[0], x), * ref_z = ggml_mul_mat(g, w[1], x);
        auto * full = ggml_new_graph(g);
        ggml_build_forward_expand(full, ref_qkv); ggml_build_forward_expand(full, ref_z);
        auto * part_w = ggml_view_2d(g, w[0], H, gpu_rows, w[0]->nb[1], 0);
        auto * part = ggml_mul_mat(g, part_w, x);
        auto * split = ggml_new_graph(g);
        ggml_build_forward_expand(split, part);
        auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(alloc, full);
        ggml_gallocr_alloc_graph(alloc, full);
        auto * alloc2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc2, split);

        MLMultiArray * xin = [[MLMultiArray alloc] initWithShape:@[@(T), @(H)] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        float * xd = (float *) xin.dataPointer;
        for (int i = 0; i < T * H; ++i) xd[i] = float((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
        ggml_backend_tensor_set(x, xd, 0, (size_t) T * H * sizeof(float));
        NSDictionary * dict = @{@"x": xin};
        slice a{load(argv[4]), [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:nil], atoi(argv[5]), atoi(argv[6])};
        slice b{load(argv[7]), [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:nil], atoi(argv[8]), atoi(argv[9])};
        if (a.start != gpu_rows || b.start != a.start + a.rows || b.start + b.rows != total) { fprintf(stderr, "slices must tile rows\n"); return 1; }

        std::vector<float> ref((size_t) T * total), got((size_t) T * total), tmp((size_t) T * qkv_rows);
        ggml_backend_graph_compute(backend, full);
        ggml_backend_tensor_get(ref_qkv, tmp.data(), 0, tmp.size() * sizeof(float));
        std::vector<float> ztmp((size_t) T * (total - qkv_rows));
        ggml_backend_tensor_get(ref_z, ztmp.data(), 0, ztmp.size() * sizeof(float));
        for (int t = 0; t < T; ++t) {
            std::copy_n(&tmp[(size_t) t * qkv_rows], qkv_rows, &ref[(size_t) t * total]);
            std::copy_n(&ztmp[(size_t) t * (total - qkv_rows)], total - qkv_rows, &ref[(size_t) t * total + qkv_rows]);
        }
        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        auto run_split = [&](bool gpu, bool ane_a, bool ane_b) {
            if (gpu) ggml_backend_graph_compute_async(backend, split);
            dispatch_group_t grp = dispatch_group_create();
            if (ane_b) dispatch_group_async(grp, q, ^{ predict(b, got.data(), total); });
            if (ane_a) predict(a, got.data(), total);
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
            if (gpu) ggml_backend_synchronize(backend);
        };
        printf("layer=%d types=%s/%s gpu_rows=%d ane_rows=%d+%d\n", layer, ggml_type_name(w[0]->type), ggml_type_name(w[1]->type), gpu_rows, a.rows, b.rows);
        printf("metal_full_ms=%.3f\n", median_ms(30, [&] { ggml_backend_graph_compute(backend, full); ggml_backend_synchronize(backend); }));
        printf("metal_part_ms=%.3f\n", median_ms(30, [&] { run_split(true, false, false); }));
        printf("ane_a_ms=%.3f\n", median_ms(30, [&] { run_split(false, true, false); }));
        printf("ane_b_ms=%.3f\n", median_ms(30, [&] { run_split(false, false, true); }));
        printf("ane_both_ms=%.3f\n", median_ms(30, [&] { run_split(false, true, true); }));
        printf("split_all_ms=%.3f\n", median_ms(30, [&] { run_split(true, true, true); }));
        printf("ane_rel_l2 a=%.5f b=%.5f\n", rel_l2(ref, got, total, a.start, a.start + a.rows), rel_l2(ref, got, total, b.start, b.start + b.rows));
    }
    return 0;
}
