# autoresearch-mac-mini

**Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that runs on a Mac Mini — no NVIDIA GPU required.**

Auto-detects your hardware and runs on **Apple Silicon (MPS)**, **CPU**, or **CUDA** — no code changes needed.

## Baseline results (Mac Mini M4, MPS)

| Metric | Mac Mini (this fork) | H100 (upstream) |
|--------|---------------------|-----------------|
| **val_bpb** | **1.723** | ~0.998 |
| training time | 300s (5 min) | 300s (5 min) |
| model params | 11.5M | 50.3M |
| depth | 4 | 8 |
| tok/sec | ~18,000 | ~1,600,000 |
| peak memory | 198 MB | ~44 GB |
| steps completed | 96 | ~953 |
| batch size | 8 | 128 |
| precision | float32 | bfloat16 |
| torch.compile | no | yes |

The val_bpb is higher (worse) than H100 — that's expected. An H100 is ~90x faster and runs a 4x larger model. **But the whole point of autoresearch is the agent optimizes for YOUR hardware.** Let the agent run overnight and it will find the best architecture for your Mac.

The [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) fork pushed val_bpb down to **1.294** on M4 Max and **1.353** on Mac Mini over long overnight runs. Similar results should be achievable here.

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

## How it compares to other Mac forks

| Fork | Backend | Key difference |
|------|---------|---------------|
| **This repo** | PyTorch + MPS | Minimal diff from upstream, auto-detects CUDA/MPS/CPU, works everywhere |
| [autoresearch-macos](https://github.com/miolini/autoresearch-macos) | PyTorch + MPS | Similar approach, MPS-specific optimizations |
| [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | MLX (no PyTorch) | Full rewrite to Apple's MLX framework, fastest on Mac but bigger diff |

## Credits

All credit to [@karpathy](https://github.com/karpathy) for the original [autoresearch](https://github.com/karpathy/autoresearch) concept and codebase.

## License

MIT
