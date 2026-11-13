#pragma once

// internal mtmd-helper entry points that test-mtmd-impl calls directly
// kept free of LOG_* macros so it can be included next to mtmd-internal.h
// NOT part of the public mtmd-helper.h API

#include "mtmd-helper.h"

// mtmd_helper_bitmap_init_from_buf() with the model's audio sample rate passed in (< 0: no audio encoder), so that test-mtmd-impl can run the decoders without loading a model
mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf_impl(const mtmd_context * ctx, int audio_sample_rate,
                                                                 const unsigned char * buf, size_t len, bool placeholder,
                                                                 mtmd_helper_init_opt opt);
