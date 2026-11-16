// Program-memory spike (probes/pfx_s1.py): the merged design's header. Words 0-7 are pgemm's; 8 and 9 gate the
// attention and row-op modes, 10-15 are the row op's.
#include <stdint.h>

extern "C" {

void ggml_xdna_pg_hdr_p(const void * a, int32_t * cnt, int32_t parity);

void pfx_hdr(const int32_t * a, int32_t * cnt, int32_t parity) {
    constexpr int PG_WORDS  = 8;
    constexpr int PFX_WORDS = 16;
    ggml_xdna_pg_hdr_p(a, cnt, parity);
    for (int i = PG_WORDS; i < PFX_WORDS; i++) {
        cnt[i] = a[i];
    }
}

}  // extern "C"
