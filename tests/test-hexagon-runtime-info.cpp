#include "../ggml/src/ggml-hexagon/htp-runtime-info.h"

#include "../ggml/src/ggml-hexagon/htp/conv-taps-validate.h"

#include <atomic>
#include <cstdio>
#include <future>
#include <thread>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

static int check_arithmetic() {
    htp_memory_count count;
    count.pages(4097, 4096);
    CHECK(count.valid && count.bytes == 8192);
    count.array(3, 128);
    CHECK(count.bytes == 8576);
    count.add(UINT64_MAX);
    CHECK(!count.valid);
    htp_memory_count product;
    product.array(UINT64_MAX, 2);
    CHECK(!product.valid);
    htp_memory_count page;
    page.pages(UINT64_MAX, 4096);
    CHECK(!page.valid);
    htp_memory_count zero_page;
    zero_page.pages(1, 0);
    CHECK(!zero_page.valid);
    CHECK(htp_runtime_max_buffer_payload(UINT64_MAX, 8192, 4096) == 4096);
    CHECK(htp_runtime_max_buffer_payload(2048, 8192, 4096) == 2048);
    CHECK(htp_runtime_max_buffer_payload(2048, 4095, 4096) == 0);
    htp_memory_count empty, small, large, overflow;
    htp_runtime_future_metadata(empty, 0, 64, 4, 128, 256);
    htp_runtime_future_metadata(small, 64, 64, 4, 128, 256);
    htp_runtime_future_metadata(large, 128, 64, 4, 128, 256);
    htp_runtime_future_metadata(overflow, UINT64_MAX, 64, 4, 128, 256);
    CHECK(empty.valid && empty.bytes == 0);
    CHECK(small.valid && small.bytes > 0 && large.valid && large.bytes > small.bytes);
    CHECK(!overflow.valid);
    return 0;
}

static int check_snapshot_exclusion() {
    std::promise<void> entered, release;
    auto released = release.get_future();
    auto worker = std::async(std::launch::async, [&] {
        htp_runtime_activity outer;
        htp_runtime_activity inner;
        entered.set_value();
        released.wait();
        return htp_runtime_activity::active();
    });
    entered.get_future().wait();
    bool independent = false;
    {
        htp_runtime_activity activity;
        independent = htp_runtime_activity::active();
    }
    auto snapshot = std::async(std::launch::async, [] {
        auto lock = htp_runtime_activity::snapshot();
        return !htp_runtime_activity::active();
    });
    const bool blocked = snapshot.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
    release.set_value();
    CHECK(worker.get());
    CHECK(snapshot.get());
    CHECK(independent && blocked && !htp_runtime_activity::active());
    return 0;
}

static int check_conv_descriptors() {
    htp_tensor w = {}, x = {}, d = {};
    w.type = HTP_TYPE_F16; x.type = d.type = HTP_TYPE_F32;
    w.ne[0] = x.ne[0] = 64; w.ne[1] = d.ne[0] = 4;
    w.ne[2] = 2; w.ne[3] = x.ne[2] = x.ne[3] = d.ne[2] = d.ne[3] = 1;
    x.ne[1] = 66; d.ne[1] = 65;
    w.nb[0] = 2; w.nb[1] = 128; w.nb[2] = 512;
    x.nb[0] = d.nb[0] = 4; x.nb[1] = 256; d.nb[1] = 16;
    w.data = 4096; x.data = 8192; d.data = 65536;
    w.size = 1024; x.size = 66 * 256; d.size = 65 * 16;
    htp_conv_taps_params p = {1, 0, 64, 1024, 1024};
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_OK);
    w.ne[0] = x.ne[0] = 0x80000040;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_NO_SUPPORT);
    w.ne[0] = x.ne[0] = 64;
    w.data += uint64_t(1) << 32;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_NO_SUPPORT);
    w.data = UINT32_MAX - 511;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_NO_SUPPORT);
    w.data = d.data;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_NO_SUPPORT);
    w.data = 4096;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 2, 1024) == HTP_STATUS_VTCM_TOO_SMALL);
    x.ne[1] = 65;
    CHECK(htp_conv_taps_validate(&w, &x, &d, &p, 1, 1024) == HTP_STATUS_NO_SUPPORT);
    return 0;
}

int main() {
    return check_arithmetic() || check_snapshot_exclusion() || check_conv_descriptors();
}
