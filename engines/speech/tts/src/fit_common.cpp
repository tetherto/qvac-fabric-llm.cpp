// Shared pieces of the memory-fit preflight surface (include/tts-cpp/fit.h).

#include "tts-cpp/fit.h"

namespace tts_cpp {

const char * fit_status_name(FitStatus status) {
    switch (status) {
        case FitStatus::Success: return "success";
        case FitStatus::Failure: return "failure";
        case FitStatus::Error:   return "error";
    }
    return "error";
}

}  // namespace tts_cpp
