# lsd — Limited Sequence Depth

A research tool for experimenting with token-level memory constraints in LLMs.

The core idea: run a local LLM with a sliding context window so that as the model generates tokens, its own earlier output falls out of its visible context. This simulates degraded medium- and long-term memory of what the model has just produced — including its own chain-of-thought reasoning.

## What it does

- Runs any HuggingFace chat model locally via a custom token-by-token generation loop
- **Sliding window**: the model only sees the last N tokens of the full conversation at each generation step — applied per-token during generation, not just between turns
- **KV cache** (default, window disabled): standard fast inference with full context
- Logs all conversations continuously to a plain text file
- Auto-detects GPU VRAM and chooses between 4-bit quantization (consumer GPUs) and bfloat16 (A100/H100)

## Install

```bash
pip install torch transformers accelerate bitsandbytes
```

## Usage

```bash
# Default: full context, KV cache enabled
python3 chat.py --model Qwen/Qwen2.5-7B-Instruct

# With a sliding window of 100 tokens
python3 chat.py --model Qwen/Qwen2.5-7B-Instruct --window 100

# Custom log file
python3 chat.py --log my_experiment.log
```

### Commands

| Command | Description |
|---|---|
| `window <n>` | Set sliding window to N tokens. `window 0` disables it (re-enables KV cache) |
| `clear` | Wipe conversation history and start fresh |
| `help` | Show command reference |
| `quit` | Exit |

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | Any HuggingFace causal LM |
| `--window` | `0` | Sliding window size in tokens (0 = disabled) |
| `--max-new` | `512` | Max tokens to generate per response |
| `--log` | `conversation.log` | Log file path (appended, not overwritten) |

## How the window works

At each token generation step, the model receives only the last N tokens of the full conversation as input. This means:

- A token generated at step K cannot attend to tokens from step K-N or earlier
- The model's own reasoning (if using CoT) progressively loses access to what it thought earlier in the same response
- User messages are subject to the same window — there is no privileged "system" vs "user" distinction at the token level

When the window is disabled (`window 0`), a KV cache is used for efficient inference. When a window is active, the cache is disabled and the context is recomputed from scratch each step.

## Hardware

- **Consumer GPU (< 40GB VRAM)**: loads in 4-bit quantization via bitsandbytes
- **A100 / H100 (≥ 40GB VRAM)**: loads in bfloat16, no quantization
- Multi-GPU: `device_map="auto"` handles tensor parallelism transparently
