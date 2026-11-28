#include "minimax/mm3-tokenizer.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr const char * kShortSpecial = "<|x|>";
constexpr const char * kLongSpecial = "<|x|>long|>";

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Vocabulary {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("minimax-tokenizer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<std::string> tokens;
    std::vector<std::string> merges = {"h e", "l l", "he ll", "hell o", "a b"};

    Vocabulary() {
        std::filesystem::create_directories(directory);
        std::string bytes[256];
        build_byte_encoder(bytes);
        tokens.assign(bytes, bytes + 256);
        for (const char * token : {"he", "ll", "hello", kShortSpecial, kLongSpecial}) tokens.emplace_back(token);
    }

    ~Vocabulary() { std::filesystem::remove_all(directory); }

    std::string write(const char * name, bool include_merges = true) const {
        const auto path = directory / name;
        gguf_context * metadata = gguf_init_empty();
        std::vector<const char *> strings;
        for (const auto & token : tokens) strings.push_back(token.c_str());
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", strings.data(), strings.size());
        strings.clear();
        for (const auto & merge : merges) strings.push_back(merge.c_str());
        if (include_merges) gguf_set_arr_str(metadata, "tokenizer.ggml.merges", strings.data(), strings.size());
        const bool written = gguf_write_to_file(metadata, path.string().c_str(), true);
        gguf_free(metadata);
        require(written, "Cannot write tokenizer fixture");
        return path.string();
    }
};

void check_encoding(const MM3Tokenizer & tokenizer, const std::string & text,
                    const std::vector<int32_t> & expected) {
    std::vector<int32_t> actual = {-1};
    mm3_tokenizer_encode(tokenizer, text, &actual);
    require(actual == expected, "MiniMax tokenizer returned incorrect token IDs");
}

void test_tokenizer() {
    const std::unordered_map<std::string, int> ranks = {{"a b", 1}, {"b c", 0}};
    require(bpe_merge(ranks, {"a", "b", "c"}) == std::vector<std::string>({"a", "bc"}),
            "BPE ignored merge priority");
    Vocabulary vocabulary;
    MM3Model model;
    model.lm_file.found = true;
    model.lm_file.path = vocabulary.write("tokenizer.gguf");
    model.lm_cfg.semantic_vocab_offset = static_cast<uint32_t>(vocabulary.tokens.size());
    MM3Tokenizer tokenizer;
    std::string error;
    require(mm3_tokenizer_load(model, &tokenizer, &error), error.c_str());
    require(tokenizer.bpe.n_vocab == static_cast<int>(vocabulary.tokens.size()), "GGUF vocabulary size differs");
    check_encoding(tokenizer, "hello", {258});
    check_encoding(tokenizer, "ab", {'a', 'b'});
    check_encoding(tokenizer, " \xC3\xA9\n", {' ', 0xC3, 0xA9, '\n'});
    check_encoding(tokenizer, "hello<|x|>long|>hello<|x|>", {258, 260, 258, 259});
    check_encoding(tokenizer, "", {});
    model.lm_file.path = vocabulary.write("missing-merges.gguf", false);
    require(!mm3_tokenizer_load(model, &tokenizer, &error), "Accepted missing GGUF merges");
    require(!tokenizer.loaded, "Failed tokenizer load retained previous vocabulary");
    model.lm_file.found = false;
    require(!mm3_tokenizer_load(model, &tokenizer, &error), "Accepted missing LM");
}

}  // namespace

int main() {
    try {
        test_tokenizer();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
