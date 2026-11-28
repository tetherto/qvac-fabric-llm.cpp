#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

static const size_t LONG_NAME_LENGTH = 100;
static_assert(LONG_NAME_LENGTH < GGML_MAX_NAME, "the long name must fit the tensor name buffer");

static const char * const ROUND_TRIP_FILE = "test-tensor-name-length.gguf";

static bool check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
    }
    return ok;
}

static bool name_is_kept_on_tensor(ggml_tensor * tensor, const std::string & name) {
    ggml_set_name(tensor, name.c_str());
    return check(strcmp(ggml_get_name(tensor), name.c_str()) == 0, "ggml_set_name keeps a name longer than 64 chars");
}

static bool write_tensor(const ggml_tensor * tensor) {
    gguf_context * ctx = gguf_init_empty();
    gguf_add_tensor(ctx, tensor);
    const bool ok = gguf_write_to_file(ctx, ROUND_TRIP_FILE, false);
    gguf_free(ctx);
    return check(ok, "gguf writes a tensor with a long name");
}

static bool read_tensor_name(const std::string & name) {
    gguf_init_params params = { /*no_alloc =*/true, /*ctx =*/nullptr };
    gguf_context *   ctx    = gguf_init_from_file(ROUND_TRIP_FILE, params);
    const bool       ok     = ctx != nullptr && gguf_find_tensor(ctx, name.c_str()) >= 0;
    gguf_free(ctx);
    remove(ROUND_TRIP_FILE);
    return check(ok, "gguf reads back a tensor with a long name");
}

int main() {
    const std::string name(LONG_NAME_LENGTH, 'n');

    ggml_init_params params = { /*mem_size =*/2 * ggml_tensor_overhead() + 1024, /*mem_buffer =*/nullptr,
                                /*no_alloc =*/false };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    bool ok = name_is_kept_on_tensor(tensor, name);
    ok      = ok && write_tensor(tensor);
    ok      = ok && read_tensor_name(name);

    ggml_free(ctx);
    return ok ? 0 : 1;
}
