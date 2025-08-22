# llama.cpp/examples/training

## finetune
This directory contains examples related to language model training using llama.cpp/GGML.
So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP.
Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory.
**For CPU training, compile llama.cpp without any additional backends such as CUDA.**
**For CUDA training, use the maximum number of GPU layers.**

Proof of concept:

``` sh
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune --file wikitext-2-raw/wiki.test.raw -ngl 999 --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
./build/bin/llama-perplexity --file wikitext-2-raw/wiki.test.raw -ngl 999 --model finetuned-model.gguf
```

The perplexity value of the finetuned model should be lower after training on the test set for 2 epochs.


## finetune-lora

LoRA (Low-Rank Adaptation) fine-tuning for efficient model training. This approach trains only a small set of additional parameters while keeping
the base model frozen, making it memory-efficient.

### Basic Usage

```sh
# Create new LoRA adapter with default settings (rank=8, alpha=16, attention modules)
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512

# Custom LoRA parameters(creates new lora adapter and trains it from scratch)
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --lora-rank 16 --lora-alpha 32 --lora-modules "attn_q,attn_k,attn_v,attn_o"

# Fine-tune existing LoRA adapter
./build/bin/llama-finetune-lora -m base_model.gguf -f dataset.txt --lora existing_adapter.gguf \
  --output-adapter improved_adapter.gguf -ngl 999 -c 512 -b 512 -ub 512

# Training with checkpointing
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --checkpoint-save-steps 50 --checkpoint-save-dir "./lora_checkpoints"

# Resume training from checkpoint
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --resume-from-checkpoint "./lora_checkpoints/checkpoint_step_00000150/"
```


### Parameters

#### LoRA Configuration
- `--lora-rank N` - LoRA rank (default: 8)
  - Lower rank = smaller adapter, less capacity
  - Higher rank = larger adapter, more capacity
- `--lora-alpha N` - LoRA alpha scaling factor (default: 16.0)
  - Controls adaptation strength
  - Common rule: alpha = 2 × rank
- `--lora-modules MODULES` - Target modules as comma-separated list
  - Available: `attn_q`, `attn_k`, `attn_v`, `attn_o`, `ffn_gate`, `ffn_up`, `ffn_down`, `embed`, `output`, `all`
  - Default: `attn_q,attn_k,attn_v,attn_o` (attention modules)
- `--output-adapter PATH` - Output adapter filename (default: auto-generated)

#### Checkpointing
- `--checkpoint-save-steps N` - Save checkpoint every N training steps (default: 100)
- `--checkpoint-save-dir PATH` - Directory for checkpoints (default: `./checkpoints`)
- `--resume-from-checkpoint PATH` - Resume training from specific checkpoint directory
- `--auto-resume` - Automatically resume from latest checkpoint in save directory

#### Standard Parameters
- `-m MODEL` - Base model file (.gguf)
- `-f FILE` - Training dataset
- `-ngl N` - GPU layers (use 999 for full GPU training)
- `-c N` - Context length (512 recommended for mobile)


### Using Trained Adapters

After training, you'll get a small adapter file. Use it with the original base model:

```sh
./build/bin/llama-cli -m base_model.gguf --lora trained_adapter.gguf -ngl 999
```

### Checkpointing

The LoRA fine-tuning supports automatic checkpointing to save and resume training progress:

#### Features
- **Automatic saving**: Model and optimizer state saved every N training steps
- **Complete state**: Includes LoRA weights, optimizer momentum, and training metadata
- **Resume capability**: Continue training from exact step with full optimizer state
- **Auto-resume**: Automatically find and resume from latest checkpoint

#### Checkpoint Structure
Each checkpoint directory contains:
- `model.gguf` - LoRA adapter weights
- `optimizer.gguf` - Optimizer state (momentum, variance, iteration)
- `metadata.json` - Training parameters and step information

### Troubleshooting

- **Out of memory**: Reduce context length (`-c 256`), lower rank, or use fewer target modules
- **Poor quality**: Increase rank, add more target modules, or train longer
- **Large adapter**: Reduce rank or limit target modules
- **Checkpoint issues**: Ensure checkpoint directory contains all required files (model.gguf, optimizer.gguf, metadata.json)

### Help

Run with `--help` or `-h` to see all available parameters:
```sh
./build/bin/llama-finetune-lora --help
```
