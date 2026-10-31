import sys, json
import numpy as np
sys.path.insert(0, "/Users/ishan/qwen38-accuracy-20261029/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize
imx = GGUFReader("imatrix-calib.gguf")
stats = {t.name: np.array(t.data) for t in imx.tensors}
model = GGUFReader("/Users/Shared/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
downs = {t.name: t for t in model.tensors if t.name.endswith("ffn_down.weight")}
out = {}
for layer in range(64):
    name = f"blk.{layer}.ffn_down.weight"
    s2 = stats[name + ".in_sum2"].reshape(-1).astype(np.float64)
    n = float(stats[name + ".counts"].reshape(-1)[0])
    t = downs[name]
    w = dequantize(t.data, t.tensor_type).astype(np.float32)  # (5120, 17408)
    e = (s2 / n) * (w.astype(np.float64) ** 2).sum(axis=0)
    blocks = e.reshape(-1, 256).sum(axis=1)
    total = blocks.sum()
    trailing = blocks[-24:].sum() / total
    order = np.argsort(blocks)
    lowest = np.sort(order[:24])
    out[layer] = dict(type=t.tensor_type.name, trailing=trailing, lowest=blocks[lowest].sum() / total, blocks=lowest.tolist())
    low = out[layer]["lowest"]
    print(layer, t.tensor_type.name, f"trailing={trailing:.4f} lowest={low:.4f} ratio={trailing/low:.2f}", flush=True)
json.dump(out, open("block-energy.json", "w"), indent=1)
