#pragma once

// Process exit and Core ML (libqvac-parakeet: Parakeet and the diarizers).
//
// A Core ML compile or load cannot be cancelled, and exit() destroys
// framework state a running one still reads. When exit starts on the main
// thread and the main thread is armed, this library first waits for the job
// its Core ML worker is running (automatic default bound: 15 s on macOS, 2 s
// on iOS; one deadline for the library), before every static destructor and
// atexit handler. The main thread is armed automatically when it makes a
// Core ML stage or requests a load. A host that makes its engines on another
// thread (a Bare or Node addon whose load runs on a worker) arms it with
// coreml_arm_exit_guard() on its main thread, or calls
// coreml_shutdown_for_exit() right before it exits. Without either, a process
// that exits while a load runs can crash inside the frameworks.
//
// In builds without Core ML these do nothing (and return false / true).

#include "parakeet/export.h"

#include <cstdint>

// Visible in static builds too (PARAKEET_API is empty there and the library
// builds with hidden visibility), so a host that links the archive into its
// own shared library can call and re-export them.
#if !defined(PARAKEET_SHARED) && (defined(__GNUC__) || defined(__clang__))
#  define PARAKEET_COREML_EXIT_API __attribute__((visibility("default")))
#else
#  define PARAKEET_COREML_EXIT_API PARAKEET_API
#endif

namespace parakeet {

// Call on the main thread (the JS thread of a Bare or Node host when that is
// the process's main thread, e.g. where the addon instance is created).
// Returns true when the guard is armed; false, doing nothing, on another
// thread, once process finalization has begun, or in a build without Core ML.
PARAKEET_COREML_EXIT_API bool coreml_arm_exit_guard();

// Terminal, for every user of this library in the process: stops new Core ML
// work for good (reloading the addon does not reopen it), then waits up to
// timeout_ms for the running job (clamped to one hour; 0 or less only stops
// new work). Returns true when no job is running; false on timeout or when
// called from inside a Core ML job. Call it right before the process exits,
// from any thread; not on an addon unload in a process that goes on, and not
// from an atexit handler or a static destructor (too late: it only reports).
PARAKEET_COREML_EXIT_API bool coreml_shutdown_for_exit(int64_t timeout_ms);

// Per library. false: an automatically armed guard does not wait, whenever
// this is called (one armed by coreml_arm_exit_guard() still does). For hosts
// whose main thread may end with pthread_exit() while the process goes on
// (dispatch_main()): on Apple platforms, after an armed main thread ends,
// later Core ML requests fail with "the process is exiting". Such hosts call
// coreml_shutdown_for_exit() right before they exit.
PARAKEET_COREML_EXIT_API void coreml_set_automatic_exit_guard(bool on);

}  // namespace parakeet
