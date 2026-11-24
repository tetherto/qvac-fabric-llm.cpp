#pragma once

// Top-level entry point for the multi-engine tts-cpp library.
//
// Persistent synthesis engines are exposed by:
//   <tts-cpp/chatterbox/engine.h>
//   <tts-cpp/supertonic/engine.h>
//   <tts-cpp/parler/engine.h>
//   <tts-cpp/cosyvoice/engine.h>
//   <tts-cpp/audio8/engine.h>
//   <tts-cpp/pocket/engine.h>
//   <tts-cpp/moss/engine.h>
// Sound-effect generation is exposed by <tts-cpp/moss/sound_effect.h>.
// Speech enhancement is exposed by <tts-cpp/lavasr/denoiser.h> and
// <tts-cpp/lavasr/enhancer.h>.
//
// Two layers of API are exposed today:
//
//   1. Persistent struct-based Engine APIs in the per-engine headers above.
//
//   2. The tts-cli argv dispatcher below and lower-level per-engine APIs such
//      as <tts-cpp/chatterbox/s3gen_pipeline.h>.

#include "tts-cpp/export.h"

#ifdef __cplusplus
extern "C" {
#endif

TTS_CPP_API int tts_cpp_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
