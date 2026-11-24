#include "moss/cli.h"

#include <cstdio>
#include <stdexcept>

int main(int argc, char ** argv) {
    tts_cpp::moss::cli::CliArgs args;
    try {
        if (!tts_cpp::moss::cli::parse_args(argc, argv, args)) {
            tts_cpp::moss::cli::print_usage();
            return 1;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[moss-cli] %s\n", error.what());
        return 1;
    }
    return tts_cpp::moss::cli::run(args);
}
