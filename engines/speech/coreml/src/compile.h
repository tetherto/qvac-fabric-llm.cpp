#pragma once

// Step 6 of the prepare pipeline on Apple platforms: compile a generated
// .mlpackage on the device and move the compiled bundle into staging.
// Implemented by src/apple/compile.mm (Objective-C++, no class definitions).

#include "cache.h"

#include <string>
#include <string_view>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace apple {

// +[MLModel compileModelAtURL:completionHandler:], waited for synchronously.
// Core ML leaves the bundle at $TMPDIR/<package stem>.mlmodelc, or at
// <package stem>_<UUID>.mlmodelc when that name is taken
// (CompileResult::tmp_output). The pipeline names each attempt's package
// qvac-coreml-<key>-<pid>-<random>.mlpackage: concurrent prepares of the same
// key in other processes or cache directories never share an output, and
// routine prune's $TMPDIR sweep finds the leftovers of a process killed
// here. The bundle is moved to `mlmodelc_out` with rename, or copied and
// removed when the two are on different volumes.
//
// Errors: a compile failure carries Core ML's NSError domain and code and
// `stage` compile: transient, and a second failure with the same code from
// another process on the same OS build is promoted by the failure records. A
// missing package or a failed move or copy is an I/O error with `stage`
// prepare: transient and never promoted. On success `os_build` (kern.osversion), `coremlc`
// (from the compiled model.mil) and `compile_peak_bytes` (growth of the
// process's physical footprint peak during the compile) are filled.
cache::CompileResult compile_package(const std::string & mlpackage, const std::string & mlmodelc_out);

// Moves a compiled bundle: rename(2), or, across volumes (EXDEV) or with
// `force_copy`, a recursive copy followed by removing `from`. A failed copy
// leaves nothing at `to`.
Error move_bundle(const std::string & from, const std::string & to, bool force_copy = false);

// The `coremlc-version` value of a compiled model.mil's buildInfo
// ("" when absent).
std::string coremlc_version(std::string_view model_mil_head);

}  // namespace apple
}  // namespace QVAC_COREML_NS
