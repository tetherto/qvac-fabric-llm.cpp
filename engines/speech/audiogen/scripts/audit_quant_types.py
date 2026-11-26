#!/usr/bin/env python3
"""Audit a quantized GGUF for tensors that must never be quantized.

Flags any tensor matching a deny-list of name substrings whose type is not
F32/F16/BF16 (a protected tensor keeps its converted source precision, not
necessarily F32). The per-type byte summary also shows whether the file hit
its size target or silently stayed mostly at source precision through the
quantizer's block-size alignment fallback.

The default deny-list covers the MM3 synth tensors that stay unquantized, as
the converter names them: the condition encoder, the vocoder, the DiT timestep
Fourier basis, and the depth decoder's positional embedding, which the depth
graph views raw. The rest of the depth decoder is quantized, so it is
deliberately absent from the list: its weights follow the requested variant like
the DiT, and its acoustic code table is held at Q8_0 for CUDA's get_rows.

usage: audit_quant_types.py in.gguf [deny_substr,deny_substr,...]
Exit 0 and prints a per-type byte summary plus a deny-list violation list
(empty on success). Exit 1 on any deny-list violation.
"""
import sys

from gguf import GGUFReader

DEFAULT_DENY = ["cond.", "voc.", "time_fourier", "depth.pos_embd.weight"]

UNQUANTIZED_TYPES = {"F32", "F16", "BF16"}


def bytes_by_type(reader):
    totals = {}
    for tensor in reader.tensors:
        type_name = tensor.tensor_type.name
        totals[type_name] = totals.get(type_name, 0) + tensor.n_bytes
    return totals


def deny_list_violations(reader, deny):
    violations = []
    for tensor in reader.tensors:
        if any(d in tensor.name for d in deny) and tensor.tensor_type.name not in UNQUANTIZED_TYPES:
            violations.append((tensor.name, tensor.tensor_type.name))
    return violations


def print_type_summary(totals):
    total_bytes = sum(totals.values())
    print(f"{'type':<10}{'bytes':>16}{'share':>10}")
    for type_name, n_bytes in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = n_bytes / total_bytes if total_bytes else 0.0
        print(f"{type_name:<10}{n_bytes:>16}{share:>10.1%}")
    print(f"{'total':<10}{total_bytes:>16}")


def print_violations(violations):
    if not violations:
        print("\ndeny-list check: PASS (no protected tensor was quantized)")
        return
    print(f"\ndeny-list check: FAIL ({len(violations)} protected tensor(s) quantized)")
    for name, type_name in violations:
        print(f"  {name}: expected F32/F16/BF16, got {type_name}")


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 1
    path = sys.argv[1]
    deny = [s for s in sys.argv[2].split(",") if s] if len(sys.argv) == 3 else DEFAULT_DENY

    reader = GGUFReader(path)
    print_type_summary(bytes_by_type(reader))
    violations = deny_list_violations(reader, deny)
    print_violations(violations)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
