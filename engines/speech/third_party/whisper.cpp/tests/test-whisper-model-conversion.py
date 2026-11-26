#!/usr/bin/env python3
"""Check actual model serializers with small NumPy-backed loader fixtures.

The torch/transformers shims replace checkpoint ingestion only. Both repository
conversion scripts execute unchanged, and an independent reader checks every
serialized shape, type, value and the end of the file. These are format tests,
not model inference tests; no Torch, transformers or model download is needed.
"""

import base64
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


TORCH_FIXTURE = """
import pickle
import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data

    def squeeze(self):
        return Tensor(self.data.squeeze())

    def numpy(self):
        return self.data

def load(file, map_location=None):
    checkpoint = pickle.load(file)
    checkpoint['model_state_dict'] = {
        name: Tensor(data) for name, data in checkpoint['model_state_dict'].items()
    }
    return checkpoint

def from_numpy(data):
    return data
"""

TRANSFORMERS_FIXTURE = """
from pathlib import Path
import pickle
from torch import Tensor

class WhisperForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        with (Path(path) / 'fixture.pkl').open('rb') as file:
            model.tensors = pickle.load(file)
        return model

    def state_dict(self):
        return {name: Tensor(data) for name, data in self.tensors.items()}
"""


class ModelConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = Path(__file__).resolve().parents[1] / 'models'
        self.shims = self.root / 'shims'
        self.shims.mkdir()
        (self.shims / 'torch.py').write_text(TORCH_FIXTURE)
        (self.shims / 'transformers.py').write_text(TRANSFORMERS_FIXTURE)
        self.assets = self.root / 'whisper' / 'whisper' / 'assets'
        self.assets.mkdir(parents=True)
        self.filters = np.array([[.1, .2, .3], [.4, .5, .6]], dtype=np.float32)
        np.savez(self.assets / 'mel_filters.npz', mel_2=self.filters)
        self.tokens = [b'a', b'b', b'c']
        (self.assets / 'gpt2.tiktoken').write_text(''.join(
            base64.b64encode(token).decode() + ' ' + str(rank) + '\n'
            for rank, token in enumerate(self.tokens)))
        self.dims = dict(n_vocab=51864, n_audio_ctx=4, n_audio_state=4,
                         n_audio_head=1, n_audio_layer=4, n_text_ctx=4,
                         n_text_state=4, n_text_head=1, n_text_layer=4, n_mels=2)
        self.shapes = {
            'encoder.conv1.weight': (4, 2, 3),
            'encoder.conv1.bias': (4,),
            'encoder.positional_embedding': (4, 4),
            'decoder.blocks.0.attn.query.weight': (4, 4),
        }

    def state(self, dtype):
        # FP32 values deliberately contain information lost by an FP16 round trip.
        return {
            name: (np.arange(np.prod(shape), dtype=np.float32) * np.float32(.001337)
                   + np.float32(.10001)).reshape(shape).astype(dtype)
            for name, shape in self.shapes.items()
        }

    def convert(self, script, checkpoint, use_f32):
        output = self.root / 'output'
        output.mkdir()
        env = dict(os.environ)
        env['PYTHONPATH'] = str(self.shims)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [sys.executable, str(self.models / script), str(checkpoint),
                   str(self.assets.parent.parent), str(output)]
        if use_f32:
            command.append('use-f32')
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output / ('ggml-model-f32.bin' if use_f32 else 'ggml-model.bin')

    def check_model(self, path, state, use_f32):
        with path.open('rb') as file:
            def read(size):
                data = file.read(size)
                self.assertEqual(len(data), size, 'truncated model payload')
                return data

            magic, *header = struct.unpack('<12i', read(48))
            self.assertEqual(magic, 0x67676d6c)
            self.assertEqual(header, list(self.dims.values()) + [0 if use_f32 else 1])
            self.assertEqual(struct.unpack('<2i', read(8)), self.filters.shape)
            self.assertEqual(read(self.filters.nbytes), self.filters.tobytes())
            self.assertEqual(struct.unpack('<i', read(4))[0], len(self.tokens))
            for token in self.tokens:
                length = struct.unpack('<I', read(4))[0]
                self.assertEqual(read(length), token)

            seen = set()
            for _ in state:
                ndims, length, ftype = struct.unpack('<3i', read(12))
                self.assertIn(ndims, (1, 2, 3))
                self.assertGreater(length, 0)
                self.assertLess(length, 128)
                shape = struct.unpack('<' + 'i' * ndims, read(ndims * 4))
                name = read(length).decode()
                self.assertIn(name, state)
                self.assertNotIn(name, seen)
                seen.add(name)
                # Explicit expectations for these fixture tensors, independent
                # of the serializer's squeeze/dimension-selection expressions.
                if name == 'encoder.conv1.weight':
                    expected_shape = (3, 2, 4)
                    expected_type = 0 if use_f32 else 1
                elif name == 'encoder.conv1.bias':
                    expected_shape = (1, 4)
                    expected_type = 0
                elif name == 'encoder.positional_embedding':
                    expected_shape = (4, 4)
                    expected_type = 0
                else:
                    expected_shape = (4, 4)
                    expected_type = 0 if use_f32 else 1
                self.assertEqual(shape, expected_shape, name)
                self.assertEqual(ftype, expected_type, name)
                dtype = np.dtype('<f4' if expected_type == 0 else '<f2')
                expected = state[name].astype(dtype).tobytes()
                self.assertEqual(read(len(expected)), expected, name)
            self.assertEqual(seen, set(state))
            self.assertEqual(file.read(), b'', 'unexpected bytes after final tensor')

    def check_pt(self, dtype, use_f32):
        state = self.state(dtype)
        checkpoint = self.root / 'model.pt'
        with checkpoint.open('wb') as file:
            pickle.dump({'dims': self.dims, 'model_state_dict': state}, file)
        self.check_model(self.convert('convert-pt-to-ggml.py', checkpoint, use_f32), state, use_f32)

    def check_hf(self, use_f32):
        state = self.state(np.float32)
        checkpoint = self.root / 'hf-model'
        checkpoint.mkdir()
        (checkpoint / 'vocab.json').write_text(json.dumps({'a': 0, 'b': 1, 'c': 2}))
        (checkpoint / 'added_tokens.json').write_text('{}')
        config = dict(vocab_size=51864, max_source_positions=4, d_model=4,
                      encoder_attention_heads=1, encoder_layers=4, max_length=4,
                      max_target_positions=4, decoder_attention_heads=1,
                      decoder_layers=4, num_mel_bins=2)
        (checkpoint / 'config.json').write_text(json.dumps(config))
        mapped = {
            'model.encoder.conv1.weight': state['encoder.conv1.weight'],
            'model.encoder.conv1.bias': state['encoder.conv1.bias'],
            'model.encoder.embed_positions.weight': state['encoder.positional_embedding'],
            'model.decoder.layers.0.self_attn.q_proj.weight': state['decoder.blocks.0.attn.query.weight'],
        }
        with (checkpoint / 'fixture.pkl').open('wb') as file:
            pickle.dump(mapped, file)
        self.check_model(self.convert('convert-h5-to-ggml.py', checkpoint, use_f32), state, use_f32)

    def test_pt_fp16_default(self):
        self.check_pt(np.float16, False)

    def test_pt_fp16_to_fp32(self):
        self.check_pt(np.float16, True)

    def test_pt_fp32_to_fp16(self):
        self.check_pt(np.float32, False)

    def test_pt_fp32_preserved(self):
        self.check_pt(np.float32, True)

    def test_hf_fp32_default(self):
        self.check_hf(False)

    def test_hf_fp32_preserved(self):
        self.check_hf(True)


if __name__ == '__main__':
    unittest.main()
