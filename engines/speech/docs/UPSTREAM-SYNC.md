# Syncing `third_party/whisper.cpp` with upstream

`third_party/whisper.cpp` is a **git subtree** of
[ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), pinned at the
tag recorded in `third_party/whisper.cpp/UPSTREAM_PIN`. Divergence policy:
*upstream-tracking, minimally divergent* — every deliberate delta is listed in
`third_party/whisper.cpp/PATCHES.md`, and the `whisper divergence guard` CI
job fails any PR whose subtree differs from the pin outside that manifest.

## Pulling a new upstream release

```sh
git remote add upstream https://github.com/ggml-org/whisper.cpp.git 2>/dev/null
git fetch upstream --tags

git checkout -b feat/whisper-sync-vX.Y.Z master
git subtree pull --prefix third_party/whisper.cpp upstream vX.Y.Z --squash
```

`git subtree pull` performs a real 3-way merge against the previous squashed
snapshot, so **conflicts appear only in the files we patch** (see PATCHES.md —
typically `src/whisper.cpp`, `src/CMakeLists.txt`, `CMakeLists.txt`,
`include/whisper.h`). Resolve them keeping both upstream's changes and the
QVAC delta; the `QVAC` comment markers show which side is ours.

If the previous subtree import was squash-merged as part of a repository PR,
its original subtree parent may no longer be in the history. The initial
v1.9.1 import in #102 has this layout: the commit containing the subtree
trailers has the entire repository as its tree, rather than an upstream-only
snapshot. Do not use that commit as the subtree merge base. Apply the exact
release-to-release diff with three-way conflict resolution instead:

```sh
OLD_PIN=$(tr -d '[:space:]' < third_party/whisper.cpp/UPSTREAM_PIN)
git diff --binary "$OLD_PIN" vX.Y.Z > /tmp/whisper-upstream.patch
git apply --3way --directory=third_party/whisper.cpp /tmp/whisper-upstream.patch
```

This stages the upstream changes and leaves conflicts for resolution without
creating a commit. Complete the same checklist below; the pin and divergence
guard establish the resulting upstream baseline. Use this path for subsequent
syncs too while the original subtree ancestry is absent.

## Post-pull checklist

1. `echo vX.Y.Z > third_party/whisper.cpp/UPSTREAM_PIN`
2. Re-verify every PATCHES.md row still applies (upstream may have absorbed or
   moved things — e.g. `WHISPER_BUILD_PARAKEET` is an upstreaming candidate
   that may land upstream eventually). Update the manifest if the delta
   changed shape; delete rows upstream made redundant.
3. `ggml/` inside the subtree must remain byte-identical to the new upstream
   tag — **never** resolve a conflict by editing vendored ggml. ggml changes
   go to the `ggml/` tree of [qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp).
4. Local sanity: umbrella configure/build + `ctest -LE 'gpu|perf'` + the
   whisper-cli smoke (see `.github/workflows/whisper-ci.yml` for the exact
   commands).
5. Open the PR and verify whisper CI and the divergence guard. The guard
   re-derives the allowed-file set from PATCHES.md, so a stale manifest fails
   loudly. A subtree-only change does not trigger the separate engine CI or
   engine cross-platform workflows, and those cross lanes do not build
   Whisper. Validate available GPU paths locally and record remaining
   platform coverage in the PR.
6. Coordinate the `speech-cpp` registry source-pin bump and downstream ASR/BCI
   SDK validation, including Windows, Android, and iOS. `speech-cpp[whisper]`
   now builds this subtree through the umbrella; the standalone `whisper-cpp`
   port is retired. Update the source REF and archive hash in the registry,
   then advance the consuming SDK's registry baseline in its own change.

## Rules of thumb

- Never hand-edit `third_party/whisper.cpp` outside a sync PR or a PR that
  updates PATCHES.md in the same change.
- Keep the QVAC delta small: prefer `engines/whisper` glue, upstream PRs, or
  the fabric ggml over growing the manifest.
- One sync PR per upstream release; don't batch unrelated work into it.
