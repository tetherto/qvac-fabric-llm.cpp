#pragma once

// Umbrella include for libqvac-parakeet (each header below can also be included alone).
//
//   <parakeet/export.h>      - PARAKEET_API visibility macro
//   <parakeet/cli.h>         - parakeet_cli_main / parakeet_fit_cli_main C
//                                   entry points
//   <parakeet/fit.h>         - fit_params memory-fit preflight (project
//                                   whether a GGUF fits the device without
//                                   loading weights)
//   <parakeet/log.h>         - parakeet_log_set host log sink
//   <parakeet/engine.h>      - Engine + EngineOptions / EngineResult
//                                   (CTC, RNN-T, TDT, EOU, Sortformer behind one
//                                   class)
//   <parakeet/streaming.h>   - StreamingOptions / StreamingSegment /
//                                   StreamSession + cross-engine
//                                   StreamEvent + VadState +
//                                   StreamEventType
//   <parakeet/diarization.h> - DiarizationOptions / Result +
//                                   SortformerStreamingOptions /
//                                   SortformerStreamSession
//   <parakeet/attributed.h>  - transcribe_with_speakers + the
//                                   attributed-segment types it emits
//   <parakeet/moss_transcribe.h> - MOSS-Transcribe-Diarize TranscribeEngine
//                                   (separate from Engine; one-pass
//                                   transcription with speaker labels)
//
// Engine families behind the umbrella `Engine` (auto-routed by GGUF
// metadata at load time):
//
//   - Parakeet-CTC 0.6B / 1.1B  -- English transcription
//   - Parakeet-TDT 0.6B-v3 -- multilingual transcription with punctuation
//     and capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Parakeet-TDT 1.1B -- English-only transcription without punctuation
//     or capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Parakeet-EOU 120M (`parakeet_realtime_eou_120m-v1`) -- low-latency
//     streaming ASR with native `<EOU>` end-of-utterance token
//   - Sortformer 4-spk -- offline diarization; v1/v2 support sliding-history
//     streaming and v2.1 adds Audio-Online Speaker Cache streaming.

#include "export.h"
#include "cli.h"
#include "fit.h"
#include "log.h"
#include "engine.h"
#include "streaming.h"
#include "diarization.h"
#include "attributed.h"
#include "moss_transcribe.h"
