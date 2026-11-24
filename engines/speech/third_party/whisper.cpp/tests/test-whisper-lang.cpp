// QVAC test: pins the whisper_lang_* lookup API — the multilingual surface
// every language option (-l, --detect-language) resolves through. Model-free.

#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_failures = 0;

static void expect(bool ok, const char * what) {
    if (ok) {
        return;
    }
    fprintf(stderr, "FAIL: %s\n", what);
    g_failures++;
}

static void check_max_id_and_full_table() {
    const int max_id = whisper_lang_max_id();
    expect(max_id == 99, "whisper_lang_max_id() == 99");

    for (int id = 0; id <= max_id; ++id) {
        const char * code = whisper_lang_str(id);
        const char * full = whisper_lang_str_full(id);
        expect(code != nullptr, "every id has a code");
        expect(full != nullptr, "every id has a full name");
        if (code == nullptr || full == nullptr) {
            return;
        }
        expect(whisper_lang_id(code) == id, "code round-trips to its id");
        expect(whisper_lang_id(full) == id, "full name round-trips to its id");
    }
}

static void check_known_anchors() {
    expect(whisper_lang_id("en") == 0, "en == 0");
    expect(whisper_lang_id("yue") == 99, "yue == 99");
    expect(strcmp(whisper_lang_str(0), "en") == 0, "id 0 == en");
    expect(strcmp(whisper_lang_str_full(0), "english") == 0,
           "id 0 full == english");
    expect(whisper_lang_id("french") == whisper_lang_id("fr"),
           "full-name lookup matches code lookup");
}

static void check_unknown_inputs() {
    expect(whisper_lang_id("klingon") == -1, "unknown code == -1");
    expect(whisper_lang_id("") == -1, "empty string == -1");
    expect(whisper_lang_str(-1) == nullptr, "negative id == nullptr");
    expect(whisper_lang_str(whisper_lang_max_id() + 1) == nullptr,
           "out-of-range id == nullptr");
    expect(whisper_lang_str_full(whisper_lang_max_id() + 1) == nullptr,
           "out-of-range id full == nullptr");
}

int main() {
    check_max_id_and_full_table();
    check_known_anchors();
    check_unknown_inputs();

    if (g_failures > 0) {
        fprintf(stderr, "test-whisper-lang: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-whisper-lang: all checks passed\n");
    return 0;
}
