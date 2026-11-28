#pragma once
#include <stdint.h>

#define HTP_CONV_TAPS_TILE_ROWS 64
#define HTP_CONV_TAPS_MAX_TAPS 16
#define HTP_CONV_TAPS_MAX_STEP 64
#define HTP_CONV_TAPS_ALIGNMENT 128

struct htp_conv_taps_params {
    int32_t tap_step;
    uint32_t first_row;
    uint32_t tile_rows;
    uint32_t worker_bytes;
    uint32_t vtcm_size;
};
