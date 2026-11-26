// Executable shim for the supertonic-fit-params tool; the implementation
// lives in the library (src/supertonic_fit_main.cpp) so hosts can link it
// directly.

#include "tts-cpp/supertonic/fit.h"

int main(int argc, char ** argv) {
    return supertonic_fit_cli_main(argc, argv);
}
