# ACE-Step comparison reports

Reviewed, committed results live under `reports/<target>/`.

```text
reports/
  README.md
  <target>/
    verification-report.md
    cpu.json
    cpu.md
    <gpu-backend>.json
    <gpu-backend>.md
    environment.json
    manifest.json
    logs/          # optional; gitignored if large
    samples/       # optional WAVs; prefer checksums if large
```

Copy files from `out/` after a completed run. Copy the exact prompt manifest
used for the run. Do not commit GGUFs or full sample trees unless they are
small enough for git.

Each target needs a human-authored `verification-report.md`. Copy
`verification-report.template.md` and complete every section:

1. Hardware and operating system.
2. QVAC, qvac-ext-ggml, acestep.cpp, and acestep.cpp ggml revisions.
3. Compiler, CMake, GPU toolkit, and backend build flags.
4. Model filenames and SHA-256 hashes.
5. Prompt manifest and run configuration.
6. Backend evidence for both engines, including fallback checks.
7. Result validation: failures, malformed or silent audio, and CLAP coverage.
8. Known asymmetries, limitations, and reviewer sign-off.
