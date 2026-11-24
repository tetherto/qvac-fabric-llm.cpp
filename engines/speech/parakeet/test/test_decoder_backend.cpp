#include "decoder_backend.h"

#include <cstdio>
#include <string>

int main() {
    struct Case {
        const char * name;
        const char * encoder;
        const char * cpu;
        bool decoder_on_cpu;
        const char * coreml;
        bool include_coreml;
        const char * expected;
    };
    const Case cases[] = {
        {"OpenCL TDT/RNN-T CPU decode", "OpenCL", "CPU", true, "", false, "CPU"},
        {"Metal native decode", "Metal", "CPU", false, "", false, "Metal"},
        {"unused Core ML with OpenCL CPU decode", "OpenCL", "CPU", true, "coreml-ane", false, "CPU"},
        {"unused Core ML with Metal decode", "Metal", "CPU", false, "coreml-ane", false, "Metal"},
        {"observed Core ML decode", "OpenCL", "CPU", true, "coreml-ane", true, "coreml-ane"},
        {"missing Core ML with CPU decode", "OpenCL", "CPU", true, "", true, "CPU"},
        {"CPU backend unavailable", "Metal", "", true, "", false, "Metal"},
    };
    for (const auto & test : cases) {
        const auto actual = parakeet::resolve_decoder_backend_name(
            test.encoder, test.cpu, test.decoder_on_cpu, test.coreml, test.include_coreml);
        if (actual != test.expected) {
            std::fprintf(stderr, "FAIL: %s: got %s, expected %s\n",
                         test.name, actual.c_str(), test.expected);
            return 1;
        }
    }
    std::printf("decoder backend policy: 7 cases passed\n");
    return 0;
}
