"""
Auto-generate progress.png from results.tsv.
Run after each experiment or anytime to see current progress.

Usage: uv run plot_progress.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

TSV_PATH = "results.tsv"
OUT_PATH = "progress.png"

def main():
    if not os.path.exists(TSV_PATH):
        print(f"No {TSV_PATH} found. Run some experiments first.")
        sys.exit(1)

    df = pd.read_csv(TSV_PATH, sep="\t")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()

    n_total = len(df)
    n_kept = len(df[df["status"] == "KEEP"])

    if n_total == 0:
        print("No experiments in results.tsv yet.")
        sys.exit(1)

    # Filter out crashes for plotting
    valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
    if len(valid) == 0:
        print("All experiments crashed. Nothing to plot.")
        sys.exit(1)

    baseline_bpb = valid.loc[0, "val_bpb"]
    kept_mask = valid["status"] == "KEEP"
    kept_idx = valid.index[kept_mask]
    kept_bpb = valid.loc[kept_mask, "val_bpb"]
    best_bpb = kept_bpb.min() if len(kept_bpb) > 0 else baseline_bpb

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Only plot points at or below baseline region
    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

    # Discarded: faint dots
    disc = below[below["status"] == "DISCARD"]
    ax.scatter(disc.index, disc["val_bpb"],
               c="#555555", s=12, alpha=0.5, zorder=2, label="Discarded")

    # Kept: prominent green dots
    kept_v = below[below["status"] == "KEEP"]
    ax.scatter(kept_v.index, kept_v["val_bpb"],
               c="#2ecc71", s=60, zorder=4, label="Kept",
               edgecolors="white", linewidths=0.5)

    # Running best step line
    if len(kept_bpb) > 0:
        running_min = kept_bpb.cummin()
        ax.step(kept_idx, running_min, where="post", color="#27ae60",
                linewidth=2.5, alpha=0.8, zorder=3, label="Running best")

    # Label each kept experiment
    for idx, bpb in zip(kept_idx, kept_bpb):
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."
        ax.annotate(desc, (idx, bpb),
                    textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5,
                    color="#5dde8e", alpha=0.9,
                    rotation=30, ha="left", va="bottom")

    # Style
    ax.set_xlabel("Experiment #", fontsize=12, color="white")
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12, color="white")
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
                 fontsize=14, color="white", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="white")
    ax.grid(True, alpha=0.15, color="#30363d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Y-axis range
    if best_bpb < baseline_bpb:
        margin = (baseline_bpb - best_bpb) * 0.15
        ax.set_ylim(best_bpb - margin, baseline_bpb + margin)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    improvement = baseline_bpb - best_bpb
    print(f"Saved {OUT_PATH} | {n_total} experiments | best: {best_bpb:.6f} | improvement: {improvement:.6f}")


if __name__ == "__main__":
    main()
