// Executable shim for the cosyvoice-fit-params tool; the implementation lives
// in the library (src/cosyvoice_fit_main.cpp) so hosts can link it directly.

#include "tts-cpp/cosyvoice/fit.h"

int main(int argc, char ** argv) {
    return cosyvoice_fit_cli_main(argc, argv);
}
