// Thin shim for the audio8-fit-params executable; the implementation lives in
// the library (src/audio8/fit_main.cpp) so hosts can link audio8_fit_cli_main
// directly, same shape as audio8-cli / src/audio8_cli.cpp.

#include "tts-cpp/audio8/fit.h"

int main(int argc, char ** argv) {
    return audio8_fit_cli_main(argc, argv);
}
