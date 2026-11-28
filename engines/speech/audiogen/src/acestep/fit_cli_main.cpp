// Thin shim for the acestep-fit-params executable; the implementation lives in
// the library (src/acestep/fit_cli.cpp) so hosts can link acestep_fit_cli_main
// directly and test/test_fit_cli.cpp can drive it in-process.

extern "C" int acestep_fit_cli_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    return acestep_fit_cli_main(argc, argv);
}
