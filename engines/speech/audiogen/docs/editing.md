# audiogen engine: audio editing

Part of the [audiogen engine documentation](../README.md).

## Audio editing

Repaint and FlowEdit are separate operations. Repaint regenerates a time
range while preserving the surrounding audio; FlowEdit morphs source
caption/lyrics conditioning toward target caption/lyrics conditioning. Both
can run independently, can be repeated, and can be composed in any order.
Because each operation consumes the previous operation's result,
`FlowEdit -> Repaint` is intentionally different from
`Repaint -> FlowEdit`.

Editing requires `GenerateParams::source_audio` as normalized, interleaved,
stereo PCM at 48 kHz. The engine derives duration from this buffer and bypasses
the LM and FSQ detokenizer. Repaint ranges must remain inside the source;
outpainting is not supported.

```cpp
tts_cpp::acestep::GenerateParams params;
params.source_audio = source_pcm_48khz_stereo;
params.seed = 22883;

tts_cpp::acestep::FlowEditParams flow;
flow.source_caption = "bright late-1990s pop";
flow.source_lyrics = original_lyrics;
flow.target_caption = "dark analog synthwave";
flow.target_lyrics = "[Instrumental]";
flow.n_min = 0.0f;
flow.n_max = 1.0f;
flow.n_avg = 1;
params.edit_plan.emplace_back(std::move(flow));

tts_cpp::acestep::RepaintParams repaint;
repaint.start_seconds = 10.0f;
repaint.end_seconds = 20.0f;
repaint.mode = tts_cpp::acestep::RepaintMode::Balanced;
repaint.strength = 0.5f;
repaint.caption = "expressive analog synthesizer solo";
repaint.lyrics = "[Instrumental]";
params.edit_plan.emplace_back(std::move(repaint));

auto result = engine->generate(params);
```

`GenerateParams::seed` seeds the first operation. Operation at index `i` uses
`seed + i`, so changing plan order also changes its deterministic noise.

### Repaint options

| Field | Default | Meaning |
|---|---|---|
| `start_seconds` | `0` | inclusive start of the repaint region |
| `end_seconds` | `-1` | end of the region; `-1` means source end |
| `mode` | `Balanced` | `Conservative`, `Balanced`, or `Aggressive` source preservation |
| `strength` | `0.5` | balanced-mode regeneration strength in `[0, 1]`; `0` preserves more and `1` regenerates more |
| `caption` | `GenerateParams::caption` | target description for this operation |
| `lyrics` | `GenerateParams::lyrics` | target lyrics for this operation |

`Conservative` maximizes source injection, boundary blending, and
post-decode waveform preservation. `Aggressive` disables those preservation
steps. `strength` controls the interpolation only in `Balanced` mode.

### FlowEdit options

| Field | Default | Meaning |
|---|---|---|
| `source_caption` | required | description of the current audio |
| `source_lyrics` | `[Instrumental]` | current lyrics |
| `target_caption` | required | desired description |
| `target_lyrics` | `[Instrumental]` | desired lyrics |
| `n_min` | `0` | beginning of the active diffusion window, in `[0, 1]` |
| `n_max` | `1` | end of the active diffusion window, in `[0, 1]` |
| `n_avg` | `1` | forward-noise samples averaged per active step; must be at least `1` |

FlowEdit v1 is the validated Turbo, Euler, no-CFG path. It requires
`diffusion_guidance_scale == 1`, with DCW, ADG, and Heun disabled; unsupported
combinations are rejected rather than silently ignored.
