#pragma once

// Why a Core ML sidecar load failed, filled by the loaders that take one.
// The runtime classes the failure from it: a framework error (the model did
// not load) is transient and keeps its error domain and code; a model that
// loaded but is not one Parakeet can run (an unexpected interface, or a
// capacity other than its file name) is deterministic.

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_load_error {
    char      domain[96];    // NSError domain of a framework failure; "" when none
    long long code;          // NSError code of a framework failure
    int       incompatible;  // nonzero: the model loaded but its interface is unusable
    char      message[256];  // the framework's description, or what did not match
};

#if defined(__cplusplus)
}
#endif

#if defined(__OBJC__)
@class NSError;
// Fills `error` (may be null) from a framework load failure.
void parakeet_coreml_record_load_error(NSError * err, struct parakeet_coreml_load_error * error);
#endif
