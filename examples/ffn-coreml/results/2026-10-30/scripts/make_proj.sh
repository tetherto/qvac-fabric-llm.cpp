#!/bin/bash
# usage: make_proj.sh layer -> proj48/layerN/{proj-512,proj-512-1}.mlmodelc + manifests
set -e
L=$1; R=~/ffn-probe-20261029; A=~/qwen38-accuracy-20261029/examples/ffn-coreml; P=~/llama-coreml-tools/bin/python; M=/Users/Shared/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
O=$R/proj48/layer$L; T=$R/tmp-proj-$L; mkdir -p $O; rm -rf $T; mkdir -p $T
part() {
  name=$1; start=$2
  $P $R/export_proj.py $M $T/fp16-$name --layer $L --row-start $start --row-count 4096 > $T/export-$name.log 2>&1
  $P $A/palettize_ffn.py $T/fp16-$name $T/b8-$name --bits 8 --tokens 512 --workers 8 --no-reference-files > $T/palette-$name.log 2>&1
  xcrun coremlcompiler compile $T/b8-$name/ffn-512.mlpackage $T/out-$name > /dev/null
  mv $T/out-$name/ffn-512.mlmodelc $O/$name.mlmodelc
  cp $T/b8-$name/manifest.json $O/manifest-$name.json
}
part proj-512 8192
part proj-512-1 12288
rm -rf $T
echo "done $L"
