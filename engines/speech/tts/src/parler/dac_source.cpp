// The identity capture parler_load_gguf calls with the descriptor it reads
// through. Built with gguf.cpp wherever it is compiled; without Core ML it
// does nothing.

#include "dac_coreml.h"
#include "internal.h"

#ifdef TTS_CPP_USE_COREML
#include "dac_source.h"

#include "qvac_coreml/generation.h"

#include <memory>
#endif

namespace tts_cpp {
namespace parler {
namespace detail {

#ifdef TTS_CPP_USE_COREML
void parler_dac_source_record(parler_model & model, int fd, const std::string & path) {
    auto s = std::make_shared<parler_dac_source>();
    s->ok = QVAC_COREML_NS::generation::source_stat_fd(fd, path, s->stat);
    model.dac_source = std::move(s);
}
#else
void parler_dac_source_record(parler_model &, int, const std::string &) {}
#endif

} // namespace detail
} // namespace parler
} // namespace tts_cpp
