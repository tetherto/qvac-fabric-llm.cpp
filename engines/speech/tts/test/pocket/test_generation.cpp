#include "pocket/generation.h"
#include <cstdio>
#include <stdexcept>
using namespace tts_cpp::pocket::detail;

void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
void check_tail(int eos_frame, int tail) {
    FrameProgress progress({34, tail});
    int advances = 0, emitted = 0;
    bool completed = false;
    while (progress.has_next()) {
        const bool eos = advances++ == eos_frame;
        if (progress.finish_frame(eos)) { completed = true; break; }
        ++emitted;
    }
    check(completed, "EOS tail did not complete within its budget");
    check(emitted == eos_frame+tail, "EOS tail emitted an incorrect frame count");
    check(advances == emitted+1, "completion advance was not accounted for");
    check(!progress.has_next(), "generation continued after completion");
}
int main() {
    try {
        // First and final possible EOS, including zero and both default tails.
        for (int tail : {0, 1, 3, 5, 100}) for (int eos : {0, 33}) check_tail(eos, tail);
        FrameProgress missing({34, 100});
        int advances = 0;
        while (missing.has_next()) {
            check(!missing.finish_frame(false), "missing EOS marked complete");
            ++advances;
        }
        check(advances == 34, "missing EOS consumed the tail reservation");
        check(frame_budget(2, 100).max_frames() == 134, "short prompt omitted tail budget");
        std::puts("Pocket EOS boundary tests passed");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
