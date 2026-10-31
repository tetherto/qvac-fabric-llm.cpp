#!/bin/bash
# usage: make_dual.sh layer  -> dual27/layerN/{ffn-512,ffn-512-1}.mlmodelc + manifests
set -e
L=$1; R=~/ffn-probe-20261029; A=~/qwen38-accuracy-20261029/examples/ffn-coreml; P=~/llama-coreml-tools/bin/python; M=/Users/Shared/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
O=$R/dual27/layer$L; T=$R/tmp-$L; mkdir -p $O; rm -rf $T; mkdir -p $T
part() {
  name=$1; start=$2
  $P $A/export_ffn.py $M $T/fp16-$name --layer $L --tokens 512 --chunk 2048 --activation tanh --output-scale 1 --channel-start $start --channel-count 4608 --no-reference-files > $T/export-$name.log 2>&1
  $P $A/palettize_ffn.py $T/fp16-$name $T/b8-$name --bits 8 --tokens 512 --workers 8 --no-reference-files > $T/palette-$name.log 2>&1
  rm -rf $T/fp16-$name
  xcrun coremlcompiler compile $T/b8-$name/ffn-512.mlpackage $T/out-$name > /dev/null
  mv $T/out-$name/ffn-512.mlmodelc $O/$name.mlmodelc
  cp $T/b8-$name/manifest.json $O/manifest-$name.json
}
part ffn-512 8192
part ffn-512-1 12800
rm -rf $T
echo "done $L"
