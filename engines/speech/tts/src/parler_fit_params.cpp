// Executable shim for the parler-fit-params tool; the implementation lives in
// the library (src/parler/fit_main.cpp) so hosts can link it directly.

#include "tts-cpp/parler/fit.h"

int main(int argc, char ** argv) {
    return parler_fit_cli_main(argc, argv);
}
