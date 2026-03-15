# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device (GPU, MPS, or CPU). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, activation functions, attention patterns, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.723225
training_seconds: 300.1
total_seconds:    450.5
peak_memory_mb:   197.8
total_tokens_M:   6.3
num_steps:        96
num_params_M:     11.5
depth:            4
device:           mps
```

You can extract the key metric from the log file:

```
grep "^val_bpb:\|^peak_memory_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 0.2) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

**IMPORTANT — ALWAYS log to results.tsv.** Append a new line after EVERY experiment, no exceptions. Example:
```
echo -e "abc1234\t1.523\t0.2\tkeep\treduce batch size" >> results.tsv
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Pick your next experiment** (see Research Strategy below)
3. Modify `train.py` with the change
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_memory_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Log to results.tsv (MANDATORY)
9. Update the progress chart: `uv run plot_progress.py`
10. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
11. If val_bpb is equal or worse, you git reset back to where you started
12. GOTO 1

## Research strategy

This is the most important section. You are a researcher, not a hyperparameter tuner.

### Rule 1: NEVER repeat a failed experiment

Before every experiment, read `results.tsv` and check if you've already tried something similar. If `head_dim 64` failed at experiment 5, do NOT try `head_dim 64` again at experiment 30. This wastes time.

### Rule 2: Change ONE thing at a time

Each experiment should change exactly ONE variable. If you change batch size AND depth AND learning rate at the same time, you learn nothing — you don't know which change helped or hurt. The only exception is when you're stuck for 15+ experiments and need a radical reset.

### Rule 3: Follow the 50/50 rule

Spend roughly half your experiments on **hyperparameter tuning** and half on **architectural changes**:

**Hyperparameter tuning** (the easy stuff):
- Batch size, learning rates, weight decay, warmdown ratio
- Adam betas, scalar LR, embedding LR
- These find the obvious wins fast but plateau quickly

**Architectural changes** (the hard stuff that breaks through plateaus):
- Activation function: try `F.gelu(x)`, `F.silu(x)`, or `F.relu(x)` instead of `F.relu(x).square()`
- MLP ratio: try `3 * config.n_embd` or `6 * config.n_embd` instead of `4 * config.n_embd`
- Remove value embeddings entirely (delete `self.value_embeds` and related code)
- Change the softcap value (15 → 30, or remove it entirely)
- Different rotary embedding base (10000 → 50000 or 100000)
- Remove `resid_lambdas` and `x0_lambdas` (simplify the residual stream)
- Try different head dimensions (64, 96, 256)
- Window patterns: try "SL", "SSL", "SSSL" if using SDPA
- Add or remove layer normalization
- Try weight tying (share embedding and unembedding weights)

### Rule 4: Analyze results every 10 experiments

Every 10 experiments, stop and read `results.tsv`. Ask yourself:
- What direction consistently helps? (e.g. "smaller models always win" → push further)
- What direction consistently fails? (e.g. "larger batch always worse" → stop trying it)
- What haven't I tried yet? Look at the architectural changes list above.
- Am I stuck in a local optimum? If no improvement in 10 experiments, try something radical.

### Rule 5: The "stuck" protocol

If you haven't improved in 10 consecutive experiments:

1. Read ALL of results.tsv carefully
2. Identify the 3 biggest improvements and what they had in common
3. Identify the 3 worst failures and what they had in common
4. Pick ONE architectural change you have NEVER tried before
5. Try it — even if it seems risky

### Key insight for this hardware

This runs on Apple Silicon (MPS) in float32 without torch.compile. That means:
- **Step count is king.** Anything that makes each step faster = more steps in 5 min = better val_bpb
- **Smaller models often beat larger ones** because they train faster
- **Smaller batches often help** because they mean more optimizer updates
- Don't waste time trying to scale up — scale down and iterate faster

**Timeout**: Each experiment should take ~7 minutes total (5 min training + ~2 min eval overhead on MPS). If a run exceeds 12 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the architectural changes list above, re-read train.py for new angles, try combining previous near-misses, try radical simplifications. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~7 minutes then you can run approx 8/hour, for a total of about 60 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
