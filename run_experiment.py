"""
Experiment runner with sticky terminal header.
Shows experiment progress at the top while training output scrolls below.

Usage: uv run run_experiment.py [description]
  e.g. uv run run_experiment.py "reduce depth to 3"

The agent should call this instead of `uv run train.py` directly.
"""

import os
import sys
import csv
import subprocess
import shutil
import signal
import re

TSV_PATH = "results.tsv"
LOG_PATH = "run.log"

# Terminal colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"

HEADER_LINES = 6


def read_results():
    """Read results.tsv and return stats."""
    if not os.path.exists(TSV_PATH):
        return 0, 0, 0, 0, None
    with open(TSV_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    total = len(rows)
    kept = sum(1 for r in rows if r["status"].strip().lower() == "keep")
    discarded = sum(1 for r in rows if r["status"].strip().lower() == "discard")
    crashed = sum(1 for r in rows if r["status"].strip().lower() == "crash")
    # Best val_bpb from kept experiments
    best = None
    for r in rows:
        if r["status"].strip().lower() == "keep":
            try:
                bpb = float(r["val_bpb"])
                if bpb > 0 and (best is None or bpb < best):
                    best = bpb
            except (ValueError, KeyError):
                pass
    return total, kept, discarded, crashed, best


def render_header(exp_num, description, best_bpb, kept, discarded, crashed, status="RUNNING"):
    """Render the sticky header."""
    cols = shutil.get_terminal_size().columns
    bar = "━" * cols

    if status == "RUNNING":
        status_color = YELLOW
        status_icon = "⏳"
    elif status == "KEEP":
        status_color = GREEN
        status_icon = "✓"
    elif status == "DISCARD":
        status_color = RED
        status_icon = "✗"
    elif status == "CRASH":
        status_color = RED
        status_icon = "💥"
    else:
        status_color = DIM
        status_icon = "?"

    best_str = f"{best_bpb:.6f}" if best_bpb else "—"
    desc = description[:cols - 30] if description else "—"

    lines = [
        f"{CYAN}{bar}{RESET}",
        f"  {BOLD}EXPERIMENT #{exp_num}{RESET}  {status_color}{status_icon} {status}{RESET}    {DIM}Testing:{RESET} {desc}",
        f"  {GREEN}Best: {best_str}{RESET}  │  Done: {kept + discarded + crashed}  │  {GREEN}Kept: {kept}{RESET}  │  {DIM}Discarded: {discarded}{RESET}  │  {RED}Crashed: {crashed}{RESET}",
        f"{CYAN}{bar}{RESET}",
        "",
    ]
    return lines


def setup_scroll_region():
    """Set terminal scroll region below the header."""
    rows = shutil.get_terminal_size().lines
    # Move cursor to top, clear screen
    sys.stdout.write("\033[2J\033[H")
    # Set scroll region: from HEADER_LINES+1 to bottom
    sys.stdout.write(f"\033[{HEADER_LINES + 1};{rows}r")
    sys.stdout.flush()


def restore_terminal():
    """Restore full terminal scroll region."""
    rows = shutil.get_terminal_size().lines
    sys.stdout.write(f"\033[1;{rows}r")
    sys.stdout.write(f"\033[{rows};1H\n")
    sys.stdout.flush()


def draw_header(lines):
    """Draw header in the fixed top area."""
    # Save cursor position
    sys.stdout.write("\033[s")
    # Move to top-left
    sys.stdout.write("\033[1;1H")
    for i, line in enumerate(lines):
        sys.stdout.write(f"\033[{i+1};1H{CLEAR_LINE}{line}")
    # Restore cursor position
    sys.stdout.write("\033[u")
    sys.stdout.flush()


def main():
    description = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

    total, kept, discarded, crashed, best_bpb = read_results()
    exp_num = total + 1

    # Setup terminal
    setup_scroll_region()

    # Draw initial header
    header = render_header(exp_num, description, best_bpb, kept, discarded, crashed, "RUNNING")
    draw_header(header)

    # Move cursor to scroll region
    sys.stdout.write(f"\033[{HEADER_LINES + 1};1H")
    sys.stdout.flush()

    # Run train.py, tee to log and scroll region
    try:
        with open(LOG_PATH, "w") as logfile:
            proc = subprocess.Popen(
                [sys.executable, "train.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                logfile.write(line)
                logfile.flush()

                # Write to scroll region (handles \r for in-place updates)
                sys.stdout.write(line)
                sys.stdout.flush()

            proc.wait()

    except KeyboardInterrupt:
        proc.kill()
        restore_terminal()
        print(f"\n{RED}Interrupted.{RESET}")
        sys.exit(1)

    # Parse result from log
    val_bpb = None
    peak_mem = None
    try:
        with open(LOG_PATH) as f:
            for line in f:
                m = re.match(r"val_bpb:\s+([\d.]+)", line)
                if m:
                    val_bpb = float(m.group(1))
                m = re.match(r"peak_memory_mb:\s+([\d.]+)", line)
                if m:
                    peak_mem = float(m.group(1))
    except Exception:
        pass

    # Determine status
    if proc.returncode != 0 or val_bpb is None:
        status = "CRASH"
    elif best_bpb is not None and val_bpb < best_bpb:
        status = "KEEP"
    elif best_bpb is None:
        status = "KEEP"
    else:
        status = "DISCARD"

    # Update header with result
    header = render_header(exp_num, description, best_bpb, kept, discarded, crashed, status)
    draw_header(header)

    # Print result summary in scroll region
    if val_bpb is not None:
        color = GREEN if status == "KEEP" else RED
        delta = ""
        if best_bpb:
            d = val_bpb - best_bpb
            delta = f"  (Δ {d:+.6f})"
        print(f"\n{color}{BOLD}>>> {status}: val_bpb = {val_bpb:.6f}{delta}{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}>>> CRASH: no val_bpb in output{RESET}\n")

    restore_terminal()
    sys.exit(0 if status in ("KEEP", "DISCARD") else 1)


if __name__ == "__main__":
    main()
