#!/usr/bin/env python3
"""Check locally provisioned music GGUF stage coverage before invoking generation.

This is a presence/header check, not a claim that tensor payloads are valid.
Native model loading performs full format/architecture validation.
"""
import sys
from pathlib import Path


MINIMAX_TIERS = ('q8_0', 'f16', 'bf16', 'q6_k', 'q5_k_m', 'q5_k_s',
                 'q4_k_m', 'q4_k_s', 'q3_k_l', 'q3_k_m', 'q3_k_s', 'q2_k', 'f32')


def minimax_pair(root):
    """Mirror directory discovery in minimax/logic.cpp::resolve_model_pair.

    Count every canonical candidate before checking headers: native discovery
    rejects duplicate names even when one candidate is malformed or unused.
    """
    candidates = {'lm': {}, 'synth': {}}
    for directory in (root / 'mm3', root):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            name = path.name.lower()
            for role in candidates:
                prefix = f'mm3-{role}-'
                if name.startswith(prefix) and name.endswith('.gguf'):
                    tier = name[len(prefix):-5]
                    if tier:
                        candidates[role].setdefault(tier, []).append(path)
    if any(len(paths) > 1 for role in candidates.values() for paths in role.values()):
        return None
    for tier in MINIMAX_TIERS:
        if tier in candidates['lm'] and tier in candidates['synth']:
            return candidates['lm'][tier][0], candidates['synth'][tier][0]
    return None


def has_header(path):
    try:
        with path.open('rb') as stream:
            return stream.read(4) == b'GGUF'
    except OSError:
        return False


def validate(directory, family):
    root = Path(directory)
    if not root.is_dir():
        return False
    if family == 'minimax':
        try:
            pair = minimax_pair(root)
            return pair is not None and all(has_header(path) for path in pair)
        except OSError:
            return False
    paths = list(root.glob('*.gguf'))
    names = []
    for p in paths:
        try:
            with p.open('rb') as f:
                if f.read(4) == b'GGUF':
                    names.append(p.name.lower())
        except OSError:
            continue
    if family == 'acestep':
        stages = set()
        for n in names:
            for stage, tokens in [('text', ('embedding', 'text-enc', 'textenc')),
                                  ('vae', ('vae',)), ('lm', ('-lm', 'lm-', '_lm', 'ace-lm', '5hz-lm')),
                                  ('dit', ('turbo', 'dit', 'v15', 'sft'))]:
                if any(t in n for t in tokens):
                    stages.add(stage)
                    break
        return stages == {'text', 'vae', 'lm', 'dit'}
    return False


if __name__ == '__main__':
    raise SystemExit(0 if validate(sys.argv[1], sys.argv[2]) else 1)
