#include "sortformer_finalize.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Harness {
    bool finalized = false;
    bool cancelled = false;
    bool has_tail = false;
    bool cancel_while_flushing = false;
    bool cancel_while_draining = false;
    std::vector<std::string> events;
};

void finalize(Harness & harness) {
    parakeet::detail::finalize_sortformer_stream(
        harness.finalized,
        harness.cancelled,
        [&] {
            harness.events.emplace_back("flush");
            harness.cancelled = harness.cancel_while_flushing;
        },
        [&] {
            harness.events.emplace_back("check-tail");
            return harness.has_tail;
        },
        [&] {
            harness.events.emplace_back("drain-tail");
            harness.cancelled = harness.cancel_while_draining;
        },
        [&] {
            harness.events.emplace_back("final");
        });
}

bool expect_events(const Harness & harness,
                   const std::vector<std::string> & expected,
                   const char * scenario) {
    if (harness.events == expected) return true;

    std::fprintf(stderr, "%s: unexpected event sequence:", scenario);
    for (const auto & event : harness.events) {
        std::fprintf(stderr, " %s", event.c_str());
    }
    std::fprintf(stderr, "\n");
    return false;
}

bool check_no_tail() {
    Harness harness;
    finalize(harness);
    finalize(harness);

    return harness.finalized
        && expect_events(
            harness, {"flush", "check-tail", "final"}, "no tail");
}

bool check_tail() {
    Harness harness;
    harness.has_tail = true;
    finalize(harness);

    return harness.finalized
        && expect_events(
            harness,
            {"flush", "check-tail", "drain-tail", "final"},
            "tail");
}

bool check_already_cancelled() {
    Harness harness;
    harness.cancelled = true;
    finalize(harness);

    return harness.finalized
        && expect_events(harness, {}, "already cancelled");
}

bool check_cancel_while_flushing() {
    Harness harness;
    harness.has_tail = true;
    harness.cancel_while_flushing = true;
    finalize(harness);

    return harness.finalized
        && expect_events(harness, {"flush"}, "cancel while flushing");
}

bool check_cancel_while_draining() {
    Harness harness;
    harness.has_tail = true;
    harness.cancel_while_draining = true;
    finalize(harness);

    return harness.finalized
        && expect_events(
            harness,
            {"flush", "check-tail", "drain-tail"},
            "cancel while draining");
}

} // namespace

int main() {
    if (!check_no_tail()
            || !check_tail()
            || !check_already_cancelled()
            || !check_cancel_while_flushing()
            || !check_cancel_while_draining()) {
        return 1;
    }

    std::puts("Sortformer finalization state tests passed.");
    return 0;
}
