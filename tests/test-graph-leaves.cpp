#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>

static bool expect(bool condition, const char * name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
    return condition;
}

static bool copy_graph_leaves(ggml_cgraph * destination, ggml_cgraph * source) {
    bool ok = true;
    const int count = ggml_graph_n_leafs(source);
    for (int index = 0; index < count; ++index) {
        ggml_tensor * leaf = ggml_graph_leaf(source, index);
        ok &= expect(leaf == ggml_graph_leaf(source, index - count), "negative leaf index");
        ok &= expect(leaf == ggml_graph_leafs(source)[index], "leaf array");
        ggml_graph_add_leaf(destination, leaf);
        ok &= expect(ggml_graph_leaf(destination, index) == leaf, "appended leaf order");
        ok &= expect(ggml_graph_get_tensor(destination, ggml_get_name(leaf)) == leaf, "cut graph leaf lookup");
    }
    return ok;
}

static bool test_graph_leaves() {
    constexpr size_t context_bytes = 1024 * 1024;
    constexpr size_t leaf_count = 2;
    constexpr float input_a = 2.0f;
    constexpr float input_b = 3.0f;
    ggml_context * context = ggml_init({context_bytes, nullptr, false});
    if (!expect(context != nullptr, "graph context")) {
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    ggml_tensor * b = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    ggml_set_name(a, "cut_input_a");
    ggml_set_name(b, "cut_input_b");
    *static_cast<float *>(a->data) = input_a;
    *static_cast<float *>(b->data) = input_b;
    ggml_tensor * sum = ggml_add(context, a, b);
    ggml_tensor * result = ggml_add(context, sum, a);
    ggml_cgraph * source = ggml_new_graph(context);
    ggml_build_forward_expand(source, result);
    ggml_cgraph * cut = ggml_new_graph_custom(context, leaf_count, false);

    bool ok = expect(ggml_graph_n_leafs(source) == leaf_count, "shared input appears once");
    ok &= expect(ggml_graph_n_leafs(cut) == 0, "empty cut graph");
    ok &= copy_graph_leaves(cut, source);
    ok &= expect(ggml_graph_n_leafs(cut) == leaf_count, "copied leaf count at capacity");
    ggml_graph_add_node(cut, sum);
    ggml_graph_add_node(cut, result);
    ok &= expect(ggml_graph_compute_with_ctx(context, cut, 1) == GGML_STATUS_SUCCESS, "cut graph execution");
    ok &= expect(*static_cast<float *>(result->data) == input_a + input_b + input_a, "cut graph result");
    ggml_graph_clear(cut);
    ok &= expect(ggml_graph_n_leafs(cut) == 0, "cleared graph");
    ggml_graph_add_leaf(cut, b);
    ok &= expect(ggml_graph_n_leafs(cut) == 1 && ggml_graph_leaf(cut, -1) == b, "append after clear");
    ggml_free(context);
    return ok;
}

int main() {
    return test_graph_leaves() ? 0 : 1;
}
