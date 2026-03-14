# autoresearch-mac-mini

**Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that runs without an NVIDIA GPU.**

Auto-detects your hardware and runs on **Apple Silicon (MPS)**, **CPU**, or **CUDA** — no code changes needed.

## What changed from upstream

| Change | Why |
|--------|-----|
| FA3 → PyTorch SDPA | Flash Attention 3 is CUDA-only. SDPA works everywhere with automatic backend selection |
| Auto device detection | `cuda` → `mps` → `cpu`, no hardcoded device |
| `torch.compile` disabled on MPS | Crashes or produces wrong results on Metal |
| Float32 on MPS/CPU | bfloat16 autocast is unreliable on Metal. Float32 is slower but correct |
| Smaller defaults | DEPTH=4, BATCH_SIZE=8, TOTAL_BATCH=2^16, WINDOW="L" (per Karpathy's own recommendations for small compute) |
| Reduced eval tokens | Faster validation on slower hardware |
| Removed `kernels` dependency | CUDA-only package, not needed with SDPA fallback |

Everything else is identical to upstream — same model architecture, same Muon+AdamW optimizer, same `program.md` agent loop, same metric (val_bpb).

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). Any Mac, Linux box, or GPU machine.

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

The script will print which device it detected:
```
Device: mps
Compute dtype: torch.float32
```

## Running the agent

Same as upstream. Spin up Claude Code, Codex, or any coding agent:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Platform-specific notes

### Apple Silicon (MPS)
- Runs in float32 (no mixed precision). Slower than CUDA but works out of the box.
- `torch.compile` is disabled. All computation is eager mode.
- The agent will find hardware-optimal configs — smaller models that train faster tend to win on Mac because more optimizer steps fit in the 5-minute budget.

### CPU
- Slowest option but works anywhere (Linux servers, CI, etc.).
- Consider lowering `DEVICE_BATCH_SIZE` further if memory is tight.

### CUDA
- If you have an NVIDIA GPU, this fork auto-detects it and uses FA3 (if `kernels` is installed) + torch.compile + bfloat16 autocast — same as upstream.
- For full CUDA performance, install the `kernels` package: `uv pip install kernels`

## Tuning for your hardware

The defaults (DEPTH=4, BATCH_SIZE=8) are conservative starting points. The whole point of autoresearch is the agent optimizes these. But if you want to manually tune:

- **More memory available?** Increase `DEPTH` (6, 8) and `DEVICE_BATCH_SIZE` (16, 32)
- **OOM errors?** Decrease `DEVICE_BATCH_SIZE` (4, 2) or `DEPTH` (2, 3)
- **Want better results on small models?** Use [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) — lower entropy data produces more meaningful results at small scale

## Credits

All credit to [@karpathy](https://github.com/karpathy) for the original [autoresearch](https://github.com/karpathy/autoresearch) concept and codebase.

## License

MIT
