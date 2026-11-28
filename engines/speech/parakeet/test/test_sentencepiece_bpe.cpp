#include "sentencepiece_bpe.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr const char * k_marker = "\xE2\x96\x81";

std::string marked(const char * piece) {
    return std::string(k_marker) + piece;
}

parakeet::BpeVocab make_vocab() {
    parakeet::BpeVocab vocab;
    vocab.pieces = {
        marked("see"),      // 0
        marked("if"),       // 1
        "ctuation",         // 2
        marked("pun"),      // 3
        ",",                // 4
        "<blank>",          // 5
        "<s>",              // 6
        "</s>",             // 7
        "<pad>",            // 8
        marked(""),         // 9: bare word-boundary marker
        "a" + marked("b"),  // 10: marker mid-piece
        "..",               // 11: two bytes, shorter than the marker
    };
    vocab.blank_id = 5;
    vocab.bos_id   = 6;
    vocab.eos_id   = 7;
    vocab.pad_id   = 8;
    return vocab;
}

constexpr int32_t k_en_us_tag_id = 12;
constexpr int32_t k_hi_in_tag_id = 14;
constexpr int32_t k_period_id    = 15;
constexpr int32_t k_marker_id    = 9;

parakeet::BpeVocab make_locale_vocab() {
    parakeet::BpeVocab vocab = make_vocab();
    vocab.pieces.push_back("<en-US>");   // 12
    vocab.pieces.push_back("<unk>");     // 13
    vocab.pieces.push_back("<hi-IN>");   // 14
    vocab.pieces.push_back(".");         // 15
    vocab.locale_tag_ids = parakeet::find_locale_tag_ids(vocab);
    return vocab;
}

std::string transcript_text(const parakeet::BpeVocab & vocab,
                            const std::vector<int32_t> & token_ids) {
    std::string pieces;
    parakeet::append_token_pieces(vocab, token_ids, pieces);
    return parakeet::pieces_to_text(vocab, pieces);
}

bool expect_text(const parakeet::BpeVocab & vocab,
                 const std::vector<int32_t> & token_ids,
                 const std::string & expected, const char * scenario) {
    const std::string got = parakeet::detokenize(vocab, token_ids);
    if (got == expected) return true;
    std::fprintf(stderr, "%s: got \"%s\", expected \"%s\"\n", scenario,
                 got.c_str(), expected.c_str());
    return false;
}

bool check_words_join_with_spaces() {
    return expect_text(make_vocab(), {0, 1}, "see if", "words join");
}

bool check_continuation_glues_onto_previous_word() {
    return expect_text(make_vocab(), {3, 2}, "punctuation",
                       "continuation glues");
}

bool check_punctuation_glues_without_space() {
    return expect_text(make_vocab(), {0, 4, 1}, "see, if",
                       "punctuation glues");
}

bool check_leading_space_is_stripped() {
    return expect_text(make_vocab(), {0}, "see", "leading space stripped");
}

bool check_specials_are_skipped() {
    return expect_text(make_vocab(), {6, 5, 0, 8, 1, 7}, "see if",
                       "specials skipped");
}

bool check_out_of_range_ids_are_skipped() {
    const auto vocab = make_vocab();
    return expect_text(vocab, {-1, 0, (int32_t) vocab.pieces.size(), 1},
                       "see if", "out-of-range skipped");
}

bool check_marker_mid_piece_becomes_space() {
    return expect_text(make_vocab(), {0, 10}, "seea b", "mid-piece marker");
}

bool check_empty_input_yields_empty_text() {
    return expect_text(make_vocab(), {}, "", "empty input");
}

bool check_word_start_for_marked_pieces() {
    const auto vocab = make_vocab();
    return parakeet::token_is_word_start(vocab, 0) &&
           parakeet::token_is_word_start(vocab, 1) &&
           parakeet::token_is_word_start(vocab, 3) &&
           parakeet::token_is_word_start(vocab, 9);
}

bool check_word_start_false_for_continuations_and_punctuation() {
    const auto vocab = make_vocab();
    return !parakeet::token_is_word_start(vocab, 2) &&
           !parakeet::token_is_word_start(vocab, 4) &&
           !parakeet::token_is_word_start(vocab, 10) &&
           !parakeet::token_is_word_start(vocab, 11);
}

bool check_word_start_false_for_specials_and_out_of_range() {
    const auto vocab = make_vocab();
    return !parakeet::token_is_word_start(vocab, vocab.blank_id) &&
           !parakeet::token_is_word_start(vocab, vocab.bos_id) &&
           !parakeet::token_is_word_start(vocab, vocab.eos_id) &&
           !parakeet::token_is_word_start(vocab, vocab.pad_id) &&
           !parakeet::token_is_word_start(vocab, -1) &&
           !parakeet::token_is_word_start(vocab,
                                          (int32_t) vocab.pieces.size());
}


bool check_locale_tag_pieces_are_recognised() {
    return parakeet::piece_is_locale_tag("<en-US>") &&
           parakeet::piece_is_locale_tag("<zh-CN>") &&
           parakeet::piece_is_locale_tag("<haw-US>");
}

bool check_non_locale_pieces_are_not_tags() {
    return !parakeet::piece_is_locale_tag("<unk>") &&
           !parakeet::piece_is_locale_tag("<blank>") &&
           !parakeet::piece_is_locale_tag("<s>") &&
           !parakeet::piece_is_locale_tag("en-US") &&
           !parakeet::piece_is_locale_tag("<e-US>") &&
           !parakeet::piece_is_locale_tag("<en-USABC>") &&
           !parakeet::piece_is_locale_tag(marked("<en-US>").c_str()) &&
           !parakeet::piece_is_locale_tag("<>");
}

bool check_locale_tag_ids_are_found_in_order() {
    const std::vector<int32_t> expected = {k_en_us_tag_id, k_hi_in_tag_id};
    return make_locale_vocab().locale_tag_ids == expected;
}

bool check_locale_tags_are_removed_from_text() {
    const std::string got = transcript_text(
        make_locale_vocab(),
        {0, k_period_id, k_marker_id, k_en_us_tag_id, 1, k_period_id,
         k_marker_id, k_hi_in_tag_id});
    if (got == "see. if.") return true;
    std::fprintf(stderr, "locale tags removed: got \"%s\"\n", got.c_str());
    return false;
}

bool check_locale_tags_stay_in_text_without_tag_ids() {
    parakeet::BpeVocab vocab = make_locale_vocab();
    vocab.locale_tag_ids.clear();
    return transcript_text(vocab, {0, k_period_id, k_marker_id, k_en_us_tag_id}) ==
           "see. <en-US>";
}

bool check_detokenize_drops_locale_tags_and_their_spaces() {
    return expect_text(make_locale_vocab(),
                       {0, k_period_id, k_marker_id, k_en_us_tag_id, 1,
                        k_period_id, k_marker_id, k_en_us_tag_id},
                       "see. if.", "detokenize drops locale tags");
}

bool check_detokenize_keeps_spaces_without_locale_tags() {
    return expect_text(make_vocab(), {0, k_marker_id}, "see ",
                       "spaces kept without locale tags");
}

bool check_collapse_spaces_trims_and_merges_runs() {
    return parakeet::collapse_spaces("  see.   if  ") == "see. if" &&
           parakeet::collapse_spaces("   ").empty() &&
           parakeet::collapse_spaces("").empty();
}

bool check_streamed_text_extends_previous_text() {
    const auto vocab = make_locale_vocab();
    const std::vector<std::vector<int32_t>> chunks = {
        {0, k_period_id}, {k_marker_id}, {k_en_us_tag_id, 1}, {k_period_id}};
    std::vector<int32_t> seen;
    std::string streamed;
    std::string previous;
    for (const auto & chunk : chunks) {
        seen.insert(seen.end(), chunk.begin(), chunk.end());
        const std::string current = transcript_text(vocab, seen);
        if (current.compare(0, previous.size(), previous) != 0) return false;
        streamed += current.substr(previous.size());
        previous = current;
    }
    return streamed == "see. if.";
}

bool check_tokens_start_word_skips_leading_locale_tag() {
    const auto vocab = make_locale_vocab();
    return parakeet::tokens_start_word(vocab, {k_en_us_tag_id, 1}) &&
           !parakeet::tokens_start_word(vocab, {k_en_us_tag_id, k_period_id}) &&
           parakeet::tokens_start_word(vocab, {k_en_us_tag_id}) &&
           parakeet::tokens_start_word(vocab, {});
}

bool check_word_start_false_for_locale_tags() {
    return !parakeet::token_is_word_start(make_locale_vocab(), k_en_us_tag_id);
}

}

int main() {
    int failures = 0;

    failures += check_words_join_with_spaces() ? 0 : 1;
    failures += check_continuation_glues_onto_previous_word() ? 0 : 1;
    failures += check_punctuation_glues_without_space() ? 0 : 1;
    failures += check_leading_space_is_stripped() ? 0 : 1;
    failures += check_specials_are_skipped() ? 0 : 1;
    failures += check_out_of_range_ids_are_skipped() ? 0 : 1;
    failures += check_marker_mid_piece_becomes_space() ? 0 : 1;
    failures += check_empty_input_yields_empty_text() ? 0 : 1;
    failures += check_word_start_for_marked_pieces() ? 0 : 1;
    failures += check_word_start_false_for_continuations_and_punctuation()
                    ? 0 : 1;
    failures += check_word_start_false_for_specials_and_out_of_range() ? 0 : 1;
    failures += check_locale_tag_pieces_are_recognised() ? 0 : 1;
    failures += check_non_locale_pieces_are_not_tags() ? 0 : 1;
    failures += check_locale_tag_ids_are_found_in_order() ? 0 : 1;
    failures += check_locale_tags_are_removed_from_text() ? 0 : 1;
    failures += check_locale_tags_stay_in_text_without_tag_ids() ? 0 : 1;
    failures += check_detokenize_drops_locale_tags_and_their_spaces() ? 0 : 1;
    failures += check_detokenize_keeps_spaces_without_locale_tags() ? 0 : 1;
    failures += check_collapse_spaces_trims_and_merges_runs() ? 0 : 1;
    failures += check_streamed_text_extends_previous_text() ? 0 : 1;
    failures += check_tokens_start_word_skips_leading_locale_tag() ? 0 : 1;
    failures += check_word_start_false_for_locale_tags() ? 0 : 1;

    if (failures > 0) {
        std::fprintf(stderr, "test_sentencepiece_bpe: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_sentencepiece_bpe: all checks passed\n");
    return 0;
}
