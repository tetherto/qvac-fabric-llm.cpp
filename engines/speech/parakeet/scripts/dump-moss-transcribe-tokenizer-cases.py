#!/usr/bin/env python3
"""Dumps Hugging Face tokenizer ids for a fixed set of prompt-shaped strings
(default, English and hotword prompts, CJK punctuation, full-width forms,
whitespace and newline runs, digits, contractions, emoji) so
test-moss-transcribe-parity can check the native byte-level BPE against the
reference tokenizer. Output: <out_dir>/tokenizer_cases.bin, a sequence of
(u32 text bytes, text, u32 id count, i32 ids) records.

usage: dump-moss-transcribe-tokenizer-cases.py <checkpoint_dir> <out_dir>
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
HOTWORD_PREFIX = "热词提示："
ENGLISH_PROMPT = ("Transcribe the audio. For each segment, start with the timestamp and speaker ID "
                  "([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.")

CASES = [
    "\n" + DEFAULT_PROMPT,
    "\n" + DEFAULT_PROMPT + HOTWORD_PREFIX + "Tether, QVAC, vcpkg, Parakeet, Nemotron",
    "\n" + DEFAULT_PROMPT + HOTWORD_PREFIX + "北京, 上海, 深度学习, GPT-4o, café",
    "\n" + ENGLISH_PROMPT,
    "\n" + ENGLISH_PROMPT + " Hotwords: hotword1, hotword2, hotword3",
    "\n转录为文本，使用 [S01] [S02] [S03]等说话人标签。",
    "system\nYou are a helpful assistant.",
    "user\n",
    "assistant\n",
    "¿Qué tal? Él dijo: «¡Hola!» — 12,5 € y 3½ kg.",
    "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ　１２３，テスト。",
    "  leading and  double  spaces  ",
    "tabs\tand\r\nwindows\n\n\nnewlines  \n  x",
    "I'm sure they'll say it's fine, don't you think? WE'RE",
    "2026-09-28 at 14:05, room 101B, 3.14159",
    "emoji 😀👍 and symbols ©®™ ×÷ ±",
    "混合English和中文，没有空格！标点？“引号”（括号）",
]


def write_cases(path: Path, tokenizer) -> None:
    with path.open("wb") as out:
        for text in CASES:
            data = text.encode("utf-8")
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            out.write(struct.pack("<I", len(data)) + data)
            out.write(struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}i", *ids))


def main() -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_cases(args.out_dir / "tokenizer_cases.bin", Tokenizer.from_file(str(args.model_dir / "tokenizer.json")))
    print(f"wrote {len(CASES)} cases to {args.out_dir / 'tokenizer_cases.bin'}")


if __name__ == "__main__":
    main()
