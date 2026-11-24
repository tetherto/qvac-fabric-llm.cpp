# CLAP dependency capability review

Reviewed September 14, 2026 against the shared wheel lock and Socket's report for
PR #247 at `3859f172`. Socket displays 23 Medium warnings across 15 packages:
12 shell-access and 11 dynamic-execution warnings. Its broader summary counts
37 project alerts; that is a different inventory. No exception or suppression
is applied by this review.

## Scope and disposition

The review inspected installed CPython 3.12 Linux wheels, both scorer entry
points, and one complete desktop CLAP invocation using the pinned model and
retained ACE-Step WAV. Socket's package links point to source archives; a
capability absent from the inspected wheel is not proof that the source-archive
finding is incorrect. Native extensions were not exhaustively audited.

No reviewed finding establishes caption/audio-controlled shell execution or
Python evaluation in the scoring path. The concrete runtime improvement was
removing `platform.platform()` from provenance collection: it launched `uname`
to probe the processor. Reading only the system, kernel release and machine
fields of `platform.uname()` preserves useful metadata without that subprocess.
The `platform` provenance field remains a string, now containing those three
fields rather than Python's extended platform description.

The remaining installed packages are required dependencies. Dropping their pins
would invalidate the complete hash-locked environment. Keep them visible pending
security-owner triage; this document does not approve risk acceptance or declare
all findings false positives. Re-review when pins, loaders, inputs or execution
environments change.

## Package findings

S = shell access; D = dynamic execution. Paths below are relative to the named
package's installed Python module directory.

| Package | Warnings | Evidence and scoring-path assessment |
| --- | --- | --- |
| AnyIO 4.15.1 | S, D | `_backends/_asyncio.py:open_process` and the Trio backend support string shell commands; `to_process.py` supports executing callables in workers. The scorer uses neither API. `_lazyimport.py` parses installed source with `ast.PyCF_ONLY_AST`; this is not execution of caption text. Required through HTTPX. |
| Click 8.5.0 | S, D | `_termui_impl.py` implements pagers, editors and OS openers; `shell_completion.py` probes Bash. Our scorer interfaces use argparse/JSON and call none of these helpers. Required by Hugging Face Hub. |
| gmpy2 2.3.1 | S | Not installed. Socket reaches it through mpmath's optional `gmpy` extra, which the lock does not request. Reconcile this dependency edge during scanner triage. |
| h11 0.16.0 | S, D | No direct execution primitive found in inspected installed Python modules. The source archive's `setup.py` executes its own version file, and `docs/source/make-state-diagrams.py` invokes Graphviz `dot`; wheel installation/scoring runs neither. These are candidate explanations, not a mapping to Socket's exact detection locations. |
| httpcore 1.0.9 | D | Installed-module and source-archive Python inspection found no direct execution primitive. This is an unmatched scanner finding, not an established false positive. Part of the HTTP transport dependency chain; desktop model loading is local-only. The comparison scorer can intentionally download a model. |
| httpx 0.28.1 | D | Installed-module and source-archive Python inspection found no direct execution primitive. This is an unmatched scanner finding; no caption-to-code path identified. Required by Hugging Face Hub; network behavior differs between the two entry points. |
| idna 3.19 | S | No direct process/shell invocation found in installed Python modules. Source-archive `tests/test_idna_cli.py` launches subprocesses to test the CLI; those tests are not run during wheel installation/scoring. |
| markdown-it-py 4.2.0 | S, D | No matching execution/process primitive found in inspected `markdown_it` Python modules. Neither scorer renders Markdown. Required by Rich. Exact source-archive location remains a triage item. |
| mpmath 1.3.0 | D | Dynamic execution does occur during normal imports: `ctx_mp_python.py:binary_op` generates arithmetic methods from package-owned templates; `libmp/libhyper.py:make_hyp_summator` generates numerical routines. Separately, `identification.py:identify` evaluates caller-supplied symbolic constant expressions. Scorers do not call that API or route captions to symbolic evaluation. Required by SymPy/Torch; do not classify all its execution as unreachable. |
| Pygments 2.21.0 | S, D | `load_lexer_from_file` and `load_formatter_from_file` execute custom Python plugin files; `formatters/img.py` invokes `fc-list`. Scorers do not pass inputs to those loaders or formatters. Required by Rich. |
| Rich 15.0.0 | S, D | `pager.py:SystemPager` calls pydoc's pager; Unicode-table loading imports bundled modules. Scorers do not invoke a pager or render captions as Rich markup. `markup.py` uses `ast.literal_eval`. |
| safetensors 0.8.0 | S | No direct process invocation found in the installed Python wrapper. Source-archive `bindings/python/stub.py` invokes Ruff while generating stubs; scoring does not run that developer tool. The native extension is outside the Python inspection. Required by Transformers. |
| setuptools 83.0.0 | S, D | Contains actual build/install subprocess helpers and setup-code execution APIs. Wheel-only installation avoids running package source builds; CLAP does not call setuptools build APIs. Required by Torch. |
| Shellingham 1.5.4 | S | `posix/ps.py:iter_process_parents` invokes a fixed `ps` argument list for shell detection. Scorers do not call shell detection. Required by Typer. |
| typing-extensions 4.16.0 | S, D | `get_annotations(eval_str=True)` and forward-reference evaluation can evaluate Python annotations; no caption/audio-to-annotation path identified. No direct shell call found in the installed module. Source-archive `src/test_typing_extensions.py` contains subprocess tests, which scoring does not execute. |

## Runtime evidence and limits

An audit hook installed before importing the scorer observed
`subprocess.Popen`, `os.system`, `os.posix_spawn`, `os.spawn`, `socket.connect`,
`socket.bind` and `socket.getaddrinfo` events. Before the metadata change, one
`subprocess.Popen` came from the standard library's `platform.processor` probe;
no watched network event occurred. Afterward, the watched event counts were all
zero. CLAP returned `ok` with cosine **0.2931456406420641**, exactly matching the
preceding runtime baseline. All 14 scorer tests passed.

This is evidence for one Linux workload, not a sandbox or proof of every path's
behavior. Python audit events do not exhaustively cover native code. Importing
Python modules normally executes code; blocking all `exec` events would break
the interpreter and is not an appropriate response to a package capability
warning.

Desktop scoring verifies the immutable model revision and artifact hashes, then
loads locally. The older comparison post-pass intentionally supports downloading
a user-selected model; do not extend the desktop offline/hash guarantee to that
entry point. Both use the same pinned wheel lock and preprocessing policy.

Socket recommends reviewing whether flagged functionality is necessary and
examining its actual commands before choosing removal or replacement:
[Shell access guidance](https://socket.dev/alerts/shellAccess).
