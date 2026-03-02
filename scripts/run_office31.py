#!/usr/bin/env python3
"""
Run all Office-31 domain adaptation tasks for a given method.

Usage:
    python scripts/run_office31.py                    # default: dcfm
    python scripts/run_office31.py --method sourceonly
    python scripts/run_office31.py --method dcfm --seed 0
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

DOMAINS = ["amazon", "webcam", "dslr"]


def get_all_tasks():
    """Generate all (source, target) pairs for Office-31."""
    return [(s, t) for s in DOMAINS for t in DOMAINS if s != t]


def run_single_task(method: str, source: str, target: str, seed: int, exp_base: str) -> float:
    """Run one task and return the best accuracy parsed from logs."""
    exp_name = f"{exp_base}/{source[0]}2{target[0]}"
    cmd = [
        str(PYTHON), "-u", "main.py",
        f"method={method}",
        "dataset=office-31",
        f"dataset.source={source}",
        f"dataset.target={target}",
        f"exp_name={exp_name}",
        f"seed={seed}",
    ]

    print(f"\n{'='*60}")
    print(f"  {source} → {target}")
    print(f"{'='*60}")

    best_acc = 0.0
    try:
        process = subprocess.Popen(
            cmd, cwd=SRC_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            # Match "Best Acc: XX.XX%" at end of training
            m = re.search(r"Best Acc:\s*([\d.]+)%", line)
            if m:
                best_acc = max(best_acc, float(m.group(1)))

            # Also match per-epoch "Acc=XX.XX%"
            m2 = re.search(r"Acc=([\d.]+)%", line)
            if m2:
                best_acc = max(best_acc, float(m2.group(1)))

            # Match H-score for OSDA settings
            m3 = re.search(r"H-score:\s*([\d.]+)%", line)
            if m3:
                best_acc = max(best_acc, float(m3.group(1)))

        process.wait()
        if process.returncode != 0:
            print(f"  [FAILED] exit code {process.returncode}")
    except KeyboardInterrupt:
        process.terminate()
        raise

    return best_acc


def print_results(results: dict, method: str, duration: float):
    """Print a formatted results table."""
    header = f"\n{'='*60}\n  Office-31 Results — method={method}\n  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n  Total time: {duration/60:.1f} min\n{'='*60}\n"
    print(header)

    lines = [header]
    accs = []

    # Print table
    row_fmt = "  {:<12s} → {:<12s}   {:>6.2f}%"
    for (src, tgt), acc in results.items():
        row = row_fmt.format(src, tgt, acc)
        print(row)
        lines.append(row)
        accs.append(acc)

    avg = sum(accs) / len(accs) if accs else 0.0
    sep = "  " + "-" * 40
    avg_line = f"  {'Average':<27s}   {avg:>6.2f}%"
    print(sep)
    print(avg_line)
    lines.extend([sep, avg_line])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all Office-31 tasks")
    parser.add_argument("--method", default="dcfm", help="Method name (default: dcfm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base = f"{args.method}_office31/{timestamp}"

    tasks = get_all_tasks()
    print(f"Running {len(tasks)} tasks for method={args.method}, seed={args.seed}")

    results = {}
    start = time.time()

    try:
        for src, tgt in tasks:
            acc = run_single_task(args.method, src, tgt, args.seed, exp_base)
            results[(src, tgt)] = acc
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    duration = time.time() - start
    summary = print_results(results, args.method, duration)

    # Save summary to file
    out_dir = PROJECT_ROOT / "results" / exp_base
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
