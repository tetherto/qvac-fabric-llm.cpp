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
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 -fa off

# Custom LoRA parameters(creates new lora adapter and trains it from scratch)
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --lora-rank 16 --lora-alpha 32 --lora-modules "attn_q,attn_k,attn_v,attn_o" -fa off

# Fine-tune existing LoRA adapter
./build/bin/llama-finetune-lora -m base_model.gguf -f dataset.txt --lora existing_adapter.gguf \
  --output-adapter improved_adapter.gguf -ngl 999 -c 512 -b 512 -ub 512

# Training with checkpointing
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --checkpoint-save-steps 50 --checkpoint-save-dir "./lora_checkpoints"

# Resume training from checkpoint
./build/bin/llama-finetune-lora -m model.gguf -f dataset.txt -ngl 999 -c 512 -b 512 -ub 512 \
  --resume-from "./lora_checkpoints/checkpoint_step_00000150/"
  --output-adapter improved_adapter.gguf -ngl 999 -c 512 -b 512 -ub 512 -fa off

# Supervised FineTuning with Assistant only loss
./build/bin/llama-finetune-lora -m model.gguf -f dataset.jsonl -ngl 999 -c 512 -b 512 -ub 512 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o" --assistant-loss-only -fa off
```

### SFT(Instruction Fine Tuning) with Assistant Only Loss
- Masks the system and user tokens and only computes loss on assistant tokens
- Requires the dataset to be in json format just like huggingface with `role` and `content` for each role
- Allows users to optionally pass a jinja chat template with `--chat-template chat-ml-template.jinja`

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
- `--resume-from PATH` - Resume training from specific checkpoint directory
- `--auto-resume` - Automatically resume from latest checkpoint in save directory

#### Standard Parameters
- `-m MODEL` - Base model file (.gguf)
- `-f FILE` - Training dataset
- `-ngl N` - GPU layers (use 999 for full GPU training)
- `-c N` - Context length (512 recommended for mobile)
- `--assistant-loss-only` - Trains only on assistant tokens
- `--chat-template` - Jinja chat template for chat ML formatting to train on assistant tokens only
- `--learning-rate` - AdamW learning rate (default: 1e-5)
- `--weight-decay` - AdamW weight decay (default: 1e-2)
- `--lr-scheduler` - Learning rate scheduler: constant, cosine, linear (default: constant)
- `--lr-min` - Minimum LR for cosine/linear schedulers (default: 0)
- `--warmup-ratio` - Fraction of total steps for LR warmup (default: 0.0)
- `--warmup-steps` - Explicit warmup steps (overrides warmup-ratio)


### Using Trained Adapters

After training, you'll get a small adapter file. Use it with the original base model:

```sh
./build/bin/llama-cli -m base_model.gguf --lora trained_adapter.gguf -ngl 999
```

### Checkpointing

The LoRA fine-tuning supports automatic checkpointing to save and resume training progress:
- **Automatic saving**: Model and optimizer state saved every N training steps
- **Complete state**: Includes LoRA weights, optimizer momentum, and training metadata
- **Resume capability**: Continue training from exact step with full optimizer state
- **Auto-resume**: Automatically find and resume from latest checkpoint

#### Checkpoint Structure
Each checkpoint directory contains:
- `model.gguf` - LoRA adapter weights
- `optimizer.gguf` - Optimizer state (momentum, variance, iteration)
- `metadata.json` - Training parameters and step information

### Architecture Overview

This section explains how LoRA fine-tuning is implemented in llama.cpp:

**LoRA Adapter Management (`src/llama-lora-training.cpp`):**
This file manages the complete lifecycle of LoRA adapters:

1. **Adapter Creation (`llama_lora_create_adapter()`):**
  - Iterates through all model tensors to find target modules
  - Creates low-rank matrix pairs (A, B) for each selected module
  - Tensor naming: `blk.{layer}.{module}.lora_a` and `blk.{layer}.{module}.lora_b`
  - Dimensions: `A ∈ R^(d×r)`, `B ∈ R^(r×k)` where r is the rank

2. **Weight Initialization (`llama_lora_init_tensor_weights()`):**
  - Matrix A: Initialized with Gaussian distribution N(0, init_std)
  - Matrix B: Initialized to zeros
  - This ensures ΔW = BA starts at zero (no adaptation initially)
  - Supports both CPU and GPU tensors via `ggml_backend_tensor_set()`

3. **Buffer Allocation (`llama_lora_allocate_buffers()`):**
  - Auto-detects backend from base model (CPU/CUDA/Vulkan)
  - Allocates LoRA tensors on same device as model layers
  - Uses `ggml_backend_alloc_ctx_tensors_from_buft()` for optimal placement

4. **Module Selection:**
  - Scans tensor names for patterns: `attn_q`, `attn_k`, `attn_v`, `attn_output`
  - FFN modules: `ffn_gate`, `ffn_up`, `ffn_down`
  - Controlled by `target_modules` bitmask (lines 194-211)

5. **Optimizer Integration (`llama_opt_param_filter_lora()`):**
  - Filter function for `ggml-opt` to identify trainable parameters
  - Returns `true` only for tensors with `.lora_a` or `.lora_b` suffix
  - Ensures base model weights are excluded from gradient computation

6. **Checkpointing (`llama_lora_save_checkpoint()`):**
  - Creates checkpoint directory structure
  - Saves `model.gguf` (LoRA weights via `llama_lora_save_adapter()`)
  - Saves `optimizer.gguf` (optimizer state via `ctx->opt_save_state()`)
  - Both files required for resuming training

**Forward Pass (`ggml-opt.cpp:ggml_opt_forward()`):**
1. Input batch flows through base model with LoRA injections
2. Loss computation uses only trainable LoRA parameters, we mark only the lora-parameters as trainable with `llama_opt_param_filter_lora`
3. For instruction tuning with `--assistant-loss-only`, loss masking is applied to system/user tokens

**Backward Pass (`ggml-opt.cpp:ggml_opt_backward()`):**
1. Gradients computed only for LoRA adapters (matrices A and B)
2. Base model weights are excluded from gradient computation
3. Memory efficient: only stores gradients for low-rank matrices

**Optimizer State (`ggml-opt.cpp`):**
- Uses AdamW optimizer by default
- Maintains first moment (momentum) and second moment (variance) for each LoRA parameter
- State tensors: `opt_state_m` and `opt_state_v` for each adapter matrix
- Checkpoint format includes full optimizer state for seamless resumption

### Adding Support for a New Model Architecture in llama.cpp

Supporting a new Transformer model in llama.cpp is not automatic — it requires
implementing missing operators, adding tokenizer + prompt formatting, and
validating training/inference parity with a reference framework. The process is
repeatable and model families like Gemma, Qwen, Mistral, Grok, Phi can be
onboarded by following a consistent workflow.

Below is a generalized end-to-end porting procedure, derived from our work enabling
Gemma/Qwen inference and LoRA finetuning across CPU, Vulkan, Metal, and CUDA backends.

#### Step 1 — Analyze the Architecture

| Component           | Example Differences Across Models                     |
| ------------------- | ----------------------------------------------------- |
| Attention structure | Grouped-QKV, Multi-query, Sliding-window, Flash-attn  |
| FFN type            | SiLU-MLP (LLaMA), GEGLU (Gemma), SWIGLU/MoE (Mixtral) |
| Positional encoding | RoPE, XPos, ALiBi                                     |
| Tokenizer format    | BPE, SentencePiece, Unigram                           |
| Chat/Prompt style   | ChatML, Gemini-style blocks, Turn-format              |

If the model deviates from LLaMA in any FFN or attention component — new ops will be required.

#### Step 2 — Implement Missing Operators

Almost all new model bring at least one of these requirements:

| Op Type                  | Purpose                         | Required for           |
| ------------------------ | ------------------------------- | ---------------------- |
| Feed-forward activations | GEGLU, SWIGLU, MoE dispatch     | inference + training   |
| Loss + grad kernels      | CROSS_ENTROPY_BACK, MASKED_LOSS | LoRA/SFT training      |
| Matrix update ops        | OUT_PROD, OUT_PROD_BACK         | LoRA parameter updates |

#### Step 3 — Backend Kernel Extension

Each operator must exist in at least one backend to work — but training must support
Vulkan, CUDA, CPU, optionally Metal.

| Backend | Required For                                               |
| ------- | ---------------------------------------------------------- |
| CPU     | reference correctness + unit tests                         |
| Vulkan  | cross-platform inference + LoRA (Adreno, Mali, AMD, Intel) |
| CUDA    | high throughput training                                   |
| Metal   | iOS / Apple Silicon finetuning                             |

Special attention for mobile Vulkan:

- operators must tile to fit GPU SSBO memory windows
- OUT_PROD + MUL_MAT need dynamic splitting
- quantized INT4/INT8 variants reduce VRAM footprint dramatically

#### Step 4 — Add Tokenizer, Prompt Format, Chat Template

Even if inference works, instruction finetuning will fail without chat formatting.

You must implement:

```
tokenizer.json / spm.model: convert to tokenizer.gguf
default chat.jinja: system/user/assistant roles
assistant-only masking: loss applies only to assistant tokens
```

Then train:

```bash
./llama-finetune-lora -m newmodel.gguf -f data.jsonl \
    --assistant-loss-only --chat-template template.jinja \
    --lora-rank 16 -ngl 999
```

#### Step 5 — Validation Workflow

Before claiming model support:

| Test                      | Pass Criteria                                   |
| ------------------------- | ----------------------------------------------- |
| Generate text             | No NaNs, stable token distribution              |
| Forward-only parity       | Output ≈ PyTorch within float tolerance         |
| 50–200 step LoRA run      | Loss decreases consistently                     |
| Merge-adapter → inference | identical behavior to runtime adapter injection |
| Finetune resumption       | checkpoint restore reproducible                 |

If inference works but training diverges, most of the time it's a missing backward op.

#### Example — Adding Support for Gemma (Inference + LoRA)

Gemma is a non-LLaMA architecture requiring GEGLU feed-forward layers, which means
new forward + backward operators must be implemented before LoRA finetuning becomes
functional. The reference implementation for this exists in the [Gemma integration
PR](https://github.com/tetherto/qvac-ext-lib-llama.cpp/pull/63).

This PR demonstrates a complete integration path: inference, instruction fine-tuning,
adapter merging, making it an ideal template when porting additional architectures.

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
