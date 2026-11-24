#pragma once
// Audio8 alias of the shared fixed-width window plan (../coreml_windows.h);
// see docs/audio8.md ("Core ML codec sidecar") for why dropping `context`
// leading frames and zero-padding on the right are exact.

#include "../coreml_windows.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {

using ::tts_cpp::detail::coreml_window;
using ::tts_cpp::detail::plan_coreml_windows;

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
