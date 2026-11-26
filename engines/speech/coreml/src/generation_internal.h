#pragma once

// Internal to the generation glue (src/generation.cpp); the cache tests drive
// the scheduler through it.

#include "cache.h"
#include "qvac_coreml/slot.h"

namespace QVAC_COREML_NS {
namespace generation {
namespace internal {

// The PrepareResult of route `generate` for one run of the cache pipeline. An
// unsupported cache volume becomes PrepareResult::unsupported, not an error.
PrepareResult result_of(cache::PrepareOutcome & o);

}  // namespace internal
}  // namespace generation
}  // namespace QVAC_COREML_NS
