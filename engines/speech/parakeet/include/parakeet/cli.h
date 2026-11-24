#pragma once

// C entry parakeet_cli_main(argc, argv): same flags as the parakeet binary (--help lists them).
// C entry parakeet_fit_cli_main(argc, argv): the parakeet-fit-params memory-fit
// preflight tool (see include/parakeet/fit.h; --help lists the flags).
// C entry parakeet_moss_transcribe_cli_main(argc, argv): the moss-transcribe
// speech-to-text tool (see include/parakeet/moss_transcribe.h).

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

PARAKEET_API int parakeet_cli_main(int argc, char ** argv);
PARAKEET_API int parakeet_fit_cli_main(int argc, char ** argv);
PARAKEET_API int parakeet_moss_transcribe_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
