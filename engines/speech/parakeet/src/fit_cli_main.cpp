// Thin shim for the parakeet-fit-params executable; the implementation lives
// in the library (src/fit_main.cpp) so hosts can link parakeet_fit_cli_main
// directly, same shape as parakeet-cli / src/cli_main.cpp.

#include "parakeet/cli.h"

int main(int argc, char ** argv) {
    return parakeet_fit_cli_main(argc, argv);
}
