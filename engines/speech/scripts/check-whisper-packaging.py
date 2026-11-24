#!/usr/bin/env python3
"""Check the installed release version and VitisAI export generation.

Run after an umbrella build. VitisAI uses a configure-only SDK stand-in;
this checks CMake export wiring, not compilation or inference with FlexMLRT.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ggml-prefix", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    ggml = args.ggml_prefix.resolve()
    version = (source / "third_party/whisper.cpp/UPSTREAM_PIN").read_text().strip().removeprefix("v")

    with tempfile.TemporaryDirectory(prefix="whisper-packaging-") as directory:
        work = Path(directory)
        install = work / "install"
        run("cmake", "--install", build / "third_party/whisper.cpp",
            "--prefix", install, "--config", "Release")
        consumer = work / "consumer"
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text(f"""\
cmake_minimum_required(VERSION 3.20)
project(whisper_package_consumer LANGUAGES CXX)
find_package(whisper {version} EXACT CONFIG REQUIRED)
if (NOT WHISPER_VERSION STREQUAL "{version}")
    message(FATAL_ERROR "Expected release {version}, got ${{WHISPER_VERSION}}")
endif()
add_executable(check-version main.cpp)
target_link_libraries(check-version PRIVATE whisper::whisper)
enable_testing()
add_test(NAME installed-whisper-version COMMAND check-version)
""", encoding="utf-8")
        (consumer / "main.cpp").write_text(f"""\
#include <whisper.h>
#include <cstdio>
#include <cstring>
int main() {{
    const char * version = whisper_version();
    std::printf("Installed whisper_version(): %s\\n", version);
    return std::strcmp(version, "{version}") != 0;
}}
""", encoding="utf-8")
        consumer_build = work / "consumer-build"
        run("cmake", "-S", consumer, "-B", consumer_build,
            f"-DCMAKE_PREFIX_PATH={install};{ggml}")
        run("cmake", "--build", consumer_build, "--config", "Release")
        run("ctest", "--test-dir", consumer_build, "-C", "Release", "--output-on-failure")

        sdk = work / "flexmlrt"
        sdk.mkdir()
        (sdk / "FlexmlRTConfig.cmake").write_text(
            "# Configure-only SDK stand-in; no headers or runtime.\n"
            "add_library(flexmlrt::flexmlrt INTERFACE IMPORTED)\n",
            encoding="utf-8",
        )
        wrapper = work / "vitisai-source"
        wrapper.mkdir()
        (wrapper / "CMakeLists.txt").write_text(f"""\
cmake_minimum_required(VERSION 3.20)
project(vitisai_export_check LANGUAGES C CXX)
add_subdirectory("{source.as_posix()}" speech)
# Generate build-tree imports so a consumer can check the dependency graph
# without compiling against the unavailable FlexMLRT SDK.
export(EXPORT whisper-targets NAMESPACE whisper::
    FILE "${{CMAKE_BINARY_DIR}}/speech/third_party/whisper.cpp/whisper-targets.cmake")
""", encoding="utf-8")
        for shared in ("ON", "OFF"):
            vitis_build = work / f"vitisai-{shared}"
            run("cmake", "-S", wrapper, "-B", vitis_build,
                "-DCMAKE_BUILD_TYPE=Release", f"-DBUILD_SHARED_LIBS={shared}",
                "-DSPEECH_BUILD_PARAKEET=OFF", "-DSPEECH_BUILD_TTS=OFF",
                "-DSPEECH_BUILD_AUDIOGEN=OFF", "-DSPEECH_BUILD_EXECUTABLES=OFF",
                "-DWHISPER_VITISAI=ON",
                "-DWHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE=OFF",
                f"-DFlexmlRT_DIR={sdk}", f"-DCMAKE_PREFIX_PATH={ggml}")
            run("cmake", "-S", consumer, "-B", work / f"vitisai-consumer-{shared}",
                f"-Dwhisper_DIR={vitis_build}/speech/third_party/whisper.cpp",
                f"-DFlexmlRT_DIR={sdk}", f"-DCMAKE_PREFIX_PATH={ggml}")

    print("PASS: installed release version and shared/static VitisAI export generation")


if __name__ == "__main__":
    main()
