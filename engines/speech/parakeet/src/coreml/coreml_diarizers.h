#pragma once

#include <atomic>

#include "coreml/parakeet-diarizer.h"
#include "coreml/parakeet_coreml_sidecars.h"
#include "parakeet_coreml_path.h"
#include "parakeet_model_type.h"

#include <memory>
#include <string>
#include <vector>

namespace parakeet {

class CoremlSidecars;

constexpr const char * kCoremlDiarizerUnitsEnv = "PARAKEET_COREML_COMPUTE_UNITS";
constexpr const char * kCoremlDiarizerDisableEnv = "PARAKEET_COREML_DIARIZER_DISABLE";
constexpr const char * kCoremlDisableEnv = "PARAKEET_COREML_DISABLE";

// Per-call tally of diarizer passes, for the result labels.
struct CoremlDiarizerCallStats {
    int coreml = 0;      // passes that ran on a diarizer sidecar
    int not_ready = 0;   // passes that ran on ggml because the capacity was loading
    std::string label;   // Core ML label of the passes that ran on a sidecar
    int capacity = 0;    // the capacity (frames) that served the last such pass
};

// The diarizer sidecars of one model, one runtime slot per capacity
// (DiarizerSlots): each pass uses the smallest covering capacity that has
// not failed; while that capacity is still loading, run() returns
// kNoDiarizerSidecar and the pass runs on ggml. A load or prediction failure
// retires the capacity for the process (quarantine), and later passes use the
// next covering capacity or ggml. A template capacity that is not generated
// keeps a status-only slot and never serves a pass. Under use = require a
// pass that covering capacities cannot serve (every covering capacity failed
// or is not generated, a timeout, or a failed prediction, which still retires
// the capacity) throws pcm::CoremlError instead, and so does a pass longer
// than every capacity (under use = automatic it runs on ggml).
class CoremlDiarizers {
public:
    // Registers one slot per capacity with `sidecars` (DiarizerSlots).
    // `source`: the identity of the GGUF the model loaded its weights from
    // (fstat of the descriptor it read through); generated capacities
    // prepare only from it.
    CoremlDiarizers(const std::string & gguf_path, std::vector<DiarizerCapacity> capacities,
                    CoremlSidecars & sidecars, const pcm::SourceStat & source);
    ~CoremlDiarizers();
    CoremlDiarizers(const CoremlDiarizers &) = delete;
    CoremlDiarizers & operator=(const CoremlDiarizers &) = delete;

    // A capacity that has not failed covers `frames` (independent of
    // readiness). Under use = require a failed capacity still covers, so
    // run() refuses the pass rather than the caller falling back to ggml.
    bool covers(int frames);
    // Requests the covering capacity now; true when it is ready.
    bool prepare(int frames);
    int run(int valid_frames,
            const parakeet_coreml_rows * inputs, int n_inputs,
            const parakeet_coreml_rows_out * outputs, int n_outputs);
    // The owning engine's cancel flag: a pass waiting for a capacity stops
    // waiting when it is set.
    void set_cancel(const std::atomic<bool> * cancel);
    std::string label();
    // The capacities that can serve a pass: prebuilt, or generated.
    std::vector<int> capacities() const;

    // Engine calls are not concurrent (engine.h), so one tally per model.
    void reset_call_stats();
    CoremlDiarizerCallStats call_stats();

    struct State;

private:
    std::unique_ptr<State> state_;
};

bool coreml_diarizers_disabled();

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path);

// A diarizer capacity the library can generate from an embedded template.
struct CoremlDiarizerTemplate {
    int frames = 0;
    const char * template_id = nullptr;
};

// The Nemotron 3 Diarization templates (capacities 358, 745 and 1125).
const std::vector<CoremlDiarizerTemplate> & nemotron_diarizer_templates();

// The Sortformer v2.1 diarizer template (capacity 410: the default AOSC step
// and batch inputs up to 32.8 s).
const std::vector<CoremlDiarizerTemplate> & sortformer_diarizer_templates();

// The diarizer templates of a model: Nemotron 3 Diarization's, or the
// Sortformer v2.1 AOSC variant's; none for every other model. Discovery and
// fit both choose the templates here.
const std::vector<CoremlDiarizerTemplate> & diarizer_templates_for(ParakeetModelType type,
                                                                   const std::string & variant);

// The diarizer capacities of a model: every adjacent prebuilt bundle, plus
// every template in `templates` with no adjacent bundle (diarizer_capacities):
// generated when its route is generate (it matches the GGUF and this instance
// can generate), a status-only slot otherwise. Null when there is none.
std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path,
                                                           CoremlSidecars & sidecars,
                                                           const std::vector<CoremlDiarizerTemplate> & templates,
                                                           const pcm::SourceStat & source);

}
