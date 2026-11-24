# tts engine: voice conditioning (cross-engine)

Part of the [tts engine documentation](../README.md).

## Voice conditioning (cross-engine)

`emotion` and `pace` mean the same thing on every engine that has them, and
the vocabulary lives in exactly one place: `include/tts-cpp/voice_controls.h`.
Each engine declares the subset it supports; an unsupported value throws
naming that engine and listing its set, so nothing is silently degraded and no
untested prompt is ever invented.

| | emotion | pace | exact rate knob |
|---|---|---|---|
| Parler-TTS | all 12 | slow / moderate / fast | — |
| CosyVoice3 | anger, happy, neutral, sad | slow / moderate / fast | — |
| Supertonic | not supported | slow / moderate / fast | `speed` (multiplier) |
| Chatterbox | not supported | not supported | `speed` (multiplier) |
| Audio8 | not supported | not supported | — |

The 12 canonical emotions: command, anger, narration, conversation, disgust,
fear, happy, neutral, proper noun, news, sad, surprise.  Case-insensitive.
Note `anger`, not `angry` — the latter is deliberately rejected.

Every CLI accepts `--emotion` / `--pace`, plus `--list-emotions` /
`--list-paces` to print what the engine in question actually supports.
`tts-cli` routes several model families, so it lists one line per family.

Three per-engine properties worth knowing:

- **Parler** renders both into the training-caption text, so `pace` shows up
  verbatim in the description ("at a moderate pace").
- **CosyVoice3** is trained on one instruction per synthesis, so engaging two
  controls throws rather than silently picking a winner.  All four emotions,
  `neutral` included, are instructions and therefore conflict with `pace` or
  a raw `instruct`.  `pace=moderate` is the one value that engages nothing:
  CosyVoice3 has no middle-pace instruction to send, so it falls back to the
  plain zero-shot path — which keeps the prompt speech tokens and places the
  transcript after `<|endofprompt|>`, where an instruction would drop them and
  sit before it.
- **Supertonic** maps the step onto its duration multiplier *relative to the
  GGUF's own `default_speed`*, so `pace=moderate` is bit-identical to setting
  nothing.  `pace` and `speed` together throw — pick one.
- **Audio8** is zero-shot cloning driven by a reference waveform and has no
  named conditioning at all; its only prosody knobs are the sampling
  parameters.
