# llama.cpp/examples/gen-tunable

## gen-tunable

This directory contains an example tool to generate a tunable config file that can be used to tune
ops to best match backends with better peformance.

### Basic Usage

```sh
# Create ops json file
./build/bin/llama-cli -m model.gguf -ngl 999 -c 512 -fa off -st -p "Here's a small song, please review it for me" --dump-ops-file model-all-ops.json

# Generate tunable config
./build/bin/llama-gen-tunable-config --ops-file model-all-ops.json --output-tunable-config model-tunable-config.json

# Use tunable config
./build/bin/llama-cli -m model.gguf -ngl 999 -c 512 -fa off -st -p "Here's a small song, please review it for me" --tunable-config model-tunable-config.json
```

### Help

Run with `--help` or `-h` to see all available parameters:
```sh
./build/bin/llama-gen-tunable-config --help
```
