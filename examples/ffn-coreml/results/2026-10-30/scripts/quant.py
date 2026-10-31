import sys, json, shutil
from pathlib import Path
import coremltools as ct
src, dst, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
m = ct.models.MLModel(str(src / "ffn-512.mlpackage"), skip_model_load=True)
if mode == "int4b32":
    cfg = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear", dtype="int4", granularity="per_block", block_size=32, weight_threshold=4096)
elif mode == "int8ch":
    cfg = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=4096)
elif mode == "int8b32":
    cfg = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear", dtype="int8", granularity="per_block", block_size=32, weight_threshold=4096)
q = ct.optimize.coreml.linear_quantize_weights(m, ct.optimize.coreml.OptimizationConfig(global_config=cfg))
q.user_defined_metadata.update(m.user_defined_metadata)
dst.mkdir(parents=True)
q.save(str(dst / "ffn-512.mlpackage"))
print(mode, sum(p.stat().st_size for p in (dst/"ffn-512.mlpackage").rglob("*") if p.is_file()))
