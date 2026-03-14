# I Ran Karpathy's Autoresearch on a Mac Mini with Claude Haiku

Karpathy's autoresearch needs an H100 ($30K). I made it run on a Mac Mini ($599) with Claude Haiku ($0.25/hr).

The idea is simple: give an AI agent a small LLM training setup, let it experiment autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, repeats. You go to sleep, wake up to a better model.

## The Setup

- Mac Mini M4, 16GB RAM
- PyTorch with MPS (Metal Performance Shaders)
- Claude Haiku as the research agent
- Float32 (no mixed precision — MPS doesn't reliably support bfloat16)

I forked autoresearch, replaced Flash Attention 3 with PyTorch SDPA, added auto device detection, and tuned the defaults for small compute. One repo that runs on CUDA, MPS, or CPU.

## What Haiku Did

Starting from the default config (val_bpb 1.729), Haiku ran 10 experiments autonomously:

**Experiment 1 — Baseline:** 1.729. The starting point.

**Experiment 2 — Smaller batch size:** 1.519. Huge jump. Haiku figured out that on slow hardware, smaller batches (2^14 instead of 2^16) mean more optimizer steps in 5 minutes. More steps > bigger batches.

**Experiment 3 — Shallower model:** 1.470. Another big win. Depth 3 instead of 4. Same logic — fewer layers = faster per step = more total steps.

**Experiment 4 — Lower weight decay:** 1.470. Tiny improvement from 0.2 to 0.1.

**Experiments 5-10 — Searching:** All discarded. Haiku tried smaller head dim, different LR schedules, different adam betas, zero weight decay. Nothing beat 1.470.

![progress chart](https://raw.githubusercontent.com/Venkateshwar-PortoAI/autoresearch-mac-mini/master/examples/mac-mini-m4/progress.png)

## The Key Insight

**On slow hardware, smaller models win.** Not because they're architecturally better, but because they're faster per step. In a fixed 5-minute budget, a 3-layer model that does 360 steps beats a 4-layer model that does 96 steps.

This is the exact same finding the autoresearch-mlx fork discovered on M4 Max. Haiku rediscovered it independently in 10 experiments.

## What Surprised Me

1. **Haiku is good enough.** The cheapest Claude model correctly identified the winning direction (smaller + faster) in just 3 experiments.

2. **The agent gets stuck at local optima.** After finding depth=3 + batch 2^14, it couldn't improve further with incremental tweaks. A longer run with more radical changes (different activations, architecture changes) might break through.

3. **Agents don't loop well.** Both Codex and Claude stopped after a few experiments instead of running indefinitely. Getting true overnight autonomous runs is still the hard part.

## The Numbers

| Metric | Before | After | H100 (Karpathy) |
|--------|--------|-------|-----------------|
| val_bpb | 1.729 | 1.470 | ~0.998 |
| depth | 4 | 3 | 8 |
| batch size | 2^16 | 2^14 | 2^19 |
| params | 11.5M | 10.7M | 50.3M |
| memory | 198 MB | 198 MB | ~44 GB |

15% improvement. Zero human intervention. $2 in API costs.

## Try It

The fork is open source: [autoresearch-mac-mini](https://github.com/Venkateshwar-PortoAI/autoresearch-mac-mini)

Clone it, run `uv sync && uv run prepare.py`, point any AI agent at `program.md`, and leave it running. Works on any Mac, Linux box, or GPU machine.

Full results with charts: [examples/mac-mini-m4](https://github.com/Venkateshwar-PortoAI/autoresearch-mac-mini/tree/master/examples/mac-mini-m4)

If you get results on different hardware, add them to `examples/` and open a PR.

Built on [@karpathy](https://x.com/karpathy)'s [autoresearch](https://github.com/karpathy/autoresearch).
