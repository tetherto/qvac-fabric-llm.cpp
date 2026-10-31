// Times each projection matmul of every layer on Metal for one 512-token chunk.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static ggml_backend_t backend;

static double time_mm(const std::string & path, gguf_context * header, ggml_context * meta, const std::vector<std::string> & names, int tokens) {
    ggml_context * wctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    std::vector<ggml_tensor *> w;
    for (auto & n : names) {
        auto * src = ggml_get_tensor(meta, n.c_str());
        if (!src) { fprintf(stderr, "missing %s\n", n.c_str()); exit(1); }
        w.push_back(ggml_new_tensor_2d(wctx, src->type, src->ne[0], src->ne[1]));
    }
    auto * buf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    for (size_t i = 0; i < names.size(); ++i) {
        std::vector<char> bytes(ggml_nbytes(w[i]));
        std::ifstream in(path, std::ios::binary);
        in.seekg(gguf_get_data_offset(header) + gguf_get_tensor_offset(header, gguf_find_tensor(header, names[i].c_str())));
        in.read(bytes.data(), bytes.size());
        ggml_backend_tensor_set(w[i], bytes.data(), 0, bytes.size());
    }
    ggml_context * gctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
    auto * graph = ggml_new_graph(gctx);
    std::map<int64_t, ggml_tensor *> inputs;
    for (auto * t : w) {
        auto & x = inputs[t->ne[0]];
        if (!x) { x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, t->ne[0], tokens); ggml_set_input(x); }
        auto * y = ggml_mul_mat(gctx, t, x);
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
    }
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    for (auto & [k, x] : inputs) {
        std::vector<float> v(ggml_nelements(x));
        for (size_t i = 0; i < v.size(); ++i) { v[i] = float((i * 2654435761u) % 1000) / 1000.0f - 0.5f; }
        ggml_backend_tensor_set(x, v.data(), 0, v.size() * sizeof(float));
    }
    std::vector<double> times;
    for (int i = 0; i < 23; ++i) {
        auto start = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (i >= 3) { times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()); }
    }
    std::sort(times.begin(), times.end());
    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(buf);
    ggml_free(wctx);
    return times[times.size() / 2];
}

int main(int argc, char ** argv) {
    const std::string path = argv[1];
    ggml_backend_load_all();
    backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_reg_by_name("MTL"), 0), nullptr);
    ggml_context * meta = nullptr;
    gguf_context * header = gguf_init_from_file(path.c_str(), {true, &meta});
    std::map<std::string, double> total;
    for (int l = 0; l < 64; ++l) {
        auto b = "blk." + std::to_string(l) + ".";
        const bool attention = l % 4 == 3;
        std::map<std::string, std::vector<std::string>> groups;
        groups["ffn"] = {b + "ffn_gate.weight", b + "ffn_up.weight", b + "ffn_down.weight"};
        if (attention) {
            groups["attn_qkv_proj"] = {b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"};
            groups["attn_out_proj"] = {b + "attn_output.weight"};
        } else {
            groups["gdn_in_proj"] = {b + "attn_qkv.weight", b + "attn_gate.weight"};
            groups["gdn_out_proj"] = {b + "ssm_out.weight"};
            groups["gdn_ab_proj"] = {b + "ssm_alpha.weight", b + "ssm_beta.weight"};
        }
        printf("layer=%d", l);
        for (auto & [g, names] : groups) {
            double ms = time_mm(path, header, meta, names, 512);
            total[g] += ms;
            printf(" %s=%.3f", g.c_str(), ms);
        }
        printf("\n");
        fflush(stdout);
    }
    for (auto & [g, ms] : total) { printf("total %s_ms=%.2f\n", g.c_str(), ms); }
    return 0;
}
