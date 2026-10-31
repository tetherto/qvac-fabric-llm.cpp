"""Build K small distinct Core ML models and one multifunction package holding all of them."""
import shutil
import sys
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

K = int(sys.argv[1])
root = Path(sys.argv[2])
rng = np.random.default_rng(11259)
singles = root / "singles"
singles.mkdir(parents=True, exist_ok=True)
for i in range(K):
    path = singles / f"m{i}.mlpackage"
    if path.exists():
        continue
    w = (rng.standard_normal((512, 5120, 1, 1)) * 0.02).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(512, 5120))], opset_version=ct.target.macOS15)
    def program(x):
        x = mb.expand_dims(x=mb.transpose(x=x, perm=[1, 0]), axes=[0, 2])
        return mb.transpose(x=mb.squeeze(x=mb.conv(x=x, weight=w), axes=[0, 2]), perm=[1, 0], name="y")

    ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
               compute_precision=ct.precision.FLOAT16, skip_model_load=True).save(str(path))
desc = ct.utils.MultiFunctionDescriptor()
for i in range(K):
    desc.add_function(str(singles / f"m{i}.mlpackage"), src_function_name="main", target_function_name=f"f{i}")
desc.default_function_name = "f0"
multi = root / "multi.mlpackage"
shutil.rmtree(multi, ignore_errors=True)
ct.utils.save_multifunction(desc, str(multi))
print("built", K)
