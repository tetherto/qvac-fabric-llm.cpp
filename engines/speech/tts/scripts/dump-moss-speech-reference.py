#!/usr/bin/env python3
"""Dump MOSS-Speech stage references from the PyTorch pipeline for
test-moss-speech-parity, as raw little-endian .bin files in two folders:

  lm/     the processor's prompt grid, both prefill heads, and a greedy reply
  codec/  the user's 16 kHz audio, log-mel, and speech tokens; the voice prompt
          (24 and 16 kHz audio, tokens, trimmed mel, CAM++ embedding); and the
          reply decode (codes, flow noise, flow mel, 24 kHz waveform)

usage: dump-moss-speech-reference.py <MOSS-Speech dir> <MOSS-Speech-Codec dir>
           <out_dir> --upstream <MOSS-Speech checkout> --user-wav question.wav
           [--device mps|cuda|cpu] [--max-new-tokens 600]

The upstream checkout provides the CosyVoice2 decoder code, Matcha-TTS, and the
default voice (assets/prompt_en.wav). The LM runs in bf16 on a GPU device and in
fp32 on the CPU; the codec always runs on the CPU in fp32.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPEECH_END = 16384
MIN_NEW_TOKENS = 10
TOKEN_MEL_RATIO = 4
FLOW_RATE = 24000
VQ_RATE = 16000
SYSTEM_PROMPT = "You are a helpful voice assistant. Answer the user's questions with spoken responses."


def save(out: Path, name: str, array) -> None:
    array = np.asarray(array)
    array.astype(np.int32 if array.dtype.kind in "iu" else np.float32).tofile(out / f"{name}.bin")


def load_on_cpu(torch) -> None:
    original = torch.load

    def cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original(*args, **kwargs)

    torch.load = cpu_load


def load_codec(codec_dir: Path):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    codec_class = get_class_from_dynamic_module("modeling_moss_speech_codec.MossSpeechCodec", str(codec_dir))
    decoder_class = sys.modules[codec_class.__module__].AudioDecoder
    defaults = list(decoder_class.__init__.__defaults__)
    defaults[-1] = "cpu"
    decoder_class.__init__.__defaults__ = tuple(defaults)
    return codec_class.from_pretrained(str(codec_dir)).eval()


def load_processor(model_dir: Path, codec):
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor_class = get_class_from_dynamic_module("processing_moss_speech.MossSpeechProcessor", str(model_dir))
    return processor_class(tokenizer=AutoTokenizer.from_pretrained(str(model_dir)), audio_codec=codec)


def stop_criteria(processor):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_id: int):
            self.stop_id = stop_id

        def __call__(self, input_ids, scores):
            return input_ids[0, -1].item() == self.stop_id

    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    return StoppingCriteriaList([StopOnToken(processor.tokenizer.pad_token_id), StopOnToken(im_end)])


def dump_lm(args, processor, out: Path) -> np.ndarray:
    import torch
    from transformers import AutoModel, GenerationConfig

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    messages = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": {"path": str(args.user_wav)}}]]
    encoded = processor(messages, "audio")
    input_ids = encoded["input_ids"]
    save(out, "input_ids", input_ids[0].numpy())
    model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, dtype=dtype).to(device).eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids.to(device), use_cache=False)
        save(out, "prefill_text_logits", logits.text_logits[0, -1].float().cpu().numpy())
        save(out, "prefill_audio_logits", logits.audio_logits[0, -1].float().cpu().numpy())
        config = GenerationConfig(max_new_tokens=args.max_new_tokens, min_new_tokens=MIN_NEW_TOKENS, do_sample=False,
                                  use_cache=True)
        tokens = model.generate(input_ids=input_ids.to(device), attention_mask=encoded["attention_mask"].to(device),
                                generation_config=config, stopping_criteria=stop_criteria(processor))
    return new_rows(tokens[0].cpu().numpy(), input_ids[0].numpy(), out)


def new_rows(generated: np.ndarray, prompt: np.ndarray, out: Path) -> np.ndarray:
    echoes_prompt = generated.shape[0] > prompt.shape[0] and np.array_equal(generated[: prompt.shape[0]], prompt)
    rows = generated[prompt.shape[0]:] if echoes_prompt else generated
    save(out, "generated_ids", rows)
    return rows


def reply_codes(generated: np.ndarray) -> list[int]:
    audio = generated.reshape(-1, 2)[:-1, 1].tolist()
    return audio[: audio.index(SPEECH_END)] if SPEECH_END in audio else audio


def dump_user(codec, user_wav: Path, out: Path) -> None:
    import torchaudio

    save(out, "user_tokens", codec.encode([str(user_wav)])[0])
    user, rate = torchaudio.load(str(user_wav))
    user = torchaudio.transforms.Resample(rate, VQ_RATE)(user) if rate != VQ_RATE else user
    save(out, "user_audio_16k", user.mean(dim=0).numpy())
    features = codec.feature_extractor([user.mean(dim=0).numpy()], sampling_rate=VQ_RATE, return_attention_mask=True,
                                       return_tensors="pt", padding="longest", pad_to_multiple_of=1280)
    save(out, "user_mel", features["input_features"][0].numpy())


def dump_voice(codec, voice_wav: Path, out: Path):
    import torch
    import torchaudio

    voice_tokens = codec.encode([str(voice_wav)])[0]
    wav, rate = torchaudio.load(str(voice_wav))
    wav24 = torchaudio.transforms.Resample(rate, FLOW_RATE)(wav) if rate != FLOW_RATE else wav
    feat, _ = codec._extract_speech_feat(wav24)
    token_len = min(int(feat.shape[1] / TOKEN_MEL_RATIO), len(voice_tokens))
    feat = feat[:, : TOKEN_MEL_RATIO * token_len]
    tokens = torch.tensor(voice_tokens[:token_len]).unsqueeze(0)
    wav16 = torchaudio.transforms.Resample(FLOW_RATE, VQ_RATE)(wav24)
    embedding = codec._extract_spk_embedding(wav16)
    save(out, "voice_audio_24k", wav24[0].numpy())
    save(out, "voice_audio_16k", wav16[0].numpy())
    save(out, "voice_prompt_tokens", tokens[0].numpy())
    save(out, "voice_prompt_feat", feat[0].numpy())
    save(out, "voice_embedding", embedding[0].numpy())
    return tokens, feat, embedding


def capture_encoder_projection(flow, captured: dict):
    def hook(module, inputs, output):
        captured["mu"] = output.detach().clone()

    return flow.encoder_proj.register_forward_hook(hook)


def lengths(tensor):
    import torch

    return torch.tensor([tensor.shape[1]], dtype=torch.int32)


def dump_reply(codec, voice, codes: list[int], out: Path) -> float:
    import torch
    import torchaudio

    tokens, feat, embedding = voice
    flow = codec.audio_decoder.flow
    captured: dict = {}
    handle = capture_encoder_projection(flow, captured)
    code_tensor = torch.tensor(codes).unsqueeze(0)
    with torch.inference_mode():
        mel = flow.inference(token=code_tensor, token_len=lengths(code_tensor), prompt_token=tokens,
                             prompt_token_len=lengths(tokens), prompt_feat=feat, prompt_feat_len=lengths(feat),
                             embedding=embedding, streaming=False, finalize=True)
        mel = mel[0] if isinstance(mel, tuple) else mel
        wav, _ = codec.audio_decoder.hift.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
    handle.remove()
    save(out, "reply_codes", np.asarray(codes, dtype=np.int32))
    save(out, "flow_noise", flow.decoder.rand_noise[0, :, : captured["mu"].shape[1]].numpy())
    save(out, "flow_mel", mel[0].numpy())
    save(out, "reply_audio_24k", wav[0].numpy())
    torchaudio.save(str(out / "reply.wav"), wav, FLOW_RATE)
    return wav.shape[-1] / FLOW_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("codec_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--user-wav", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    sys.path.insert(0, str(args.upstream))
    sys.path.insert(0, str(args.upstream / "Matcha-TTS"))
    load_on_cpu(torch)
    torch.manual_seed(0)
    lm_out, codec_out = args.out_dir / "lm", args.out_dir / "codec"
    lm_out.mkdir(parents=True, exist_ok=True)
    codec_out.mkdir(parents=True, exist_ok=True)
    codec = load_codec(args.codec_dir)
    generated = dump_lm(args, load_processor(args.model_dir, codec), lm_out)
    dump_user(codec, args.user_wav, codec_out)
    codes = reply_codes(generated)
    seconds = dump_reply(codec, dump_voice(codec, args.upstream / "assets" / "prompt_en.wav", codec_out), codes,
                         codec_out)
    meta = {"prompt_rows": int(np.fromfile(lm_out / "input_ids.bin", dtype=np.int32).size // 2),
            "generated_rows": int(generated.size // 2), "reply_codes": len(codes), "reply_seconds": round(seconds, 2)}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
