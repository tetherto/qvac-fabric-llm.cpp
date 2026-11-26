// Thin shim for the chatterbox-fit-params executable; the implementation
// lives in the library (src/chatterbox_fit_main.cpp) so hosts can link
// chatterbox_fit_cli_main directly, same shape as tts-cli / src/cli_main.cpp.

#include "tts-cpp/chatterbox/fit.h"

int main(int argc, char ** argv) {
    return chatterbox_fit_cli_main(argc, argv);
}
