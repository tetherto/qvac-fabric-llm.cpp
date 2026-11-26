#include "moss/gguf_metadata.h"

#include "gguf.h"

using namespace tts_cpp::moss::detail;

namespace {

constexpr const char * ARRAY_KEY = "moss.test.ids";
constexpr size_t ARRAY_LIMIT = 8;

const std::vector<int32_t> & expected_ids() {
    static const std::vector<int32_t> ids = {3, 1, 4, 1, 5};
    return ids;
}

std::vector<int32_t> read_ids(gguf_context * file) {
    gguf_set_arr_data(file, ARRAY_KEY, GGUF_TYPE_INT32, expected_ids().data(), expected_ids().size());
    return GgufMetadata(file, "test-moss-header-includes").int_array(ARRAY_KEY, ARRAY_LIMIT);
}

}

bool gguf_metadata_header_reads_int_array() {
    gguf_context * file = gguf_init_empty();
    const bool ok = read_ids(file) == expected_ids();
    gguf_free(file);
    return ok;
}
