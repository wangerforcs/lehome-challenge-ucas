#!/usr/bin/env python3
"""Analyze evaluation log files to extract success rates and plot training curves.

Usage:
    python scripts/analyze_eval_logs.py [--log_dir LOG_DIR] [--output_dir OUTPUT_DIR]

Reads log files named like <garment><steps>.log (e.g. pant_short6000.log)
from the log directory, extracts per-garment success rates, computes
seen/unseen averages, and plots a training curve per garment type.
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log_file(filepath: Path) -> dict | None:
    """Parse a single evaluation log file.

    Returns dict with keys:
        garment_type: str (e.g. 'pant_short', 'top_long')
        steps: int
        total_success_rate: float
        garments: list of (name, success_rate) tuples
    """
    text = filepath.read_text()

    # Extract total success rate
    m = re.search(r"Success Rate:\s*([\d.]+)%", text)
    if not m:
        return None
    total_success_rate = float(m.group(1))

    # Extract per-garment success rates
    garment_pattern = re.compile(
        r"  (\w+_\w+_\w+_\d+):\s*Success Rate\s*=\s*([\d.]+)%"
    )
    garments = []
    for match in garment_pattern.finditer(text):
        name = match.group(1)
        rate = float(match.group(2))
        garments.append((name, rate))

    if not garments:
        return None

    # Determine garment type and steps from filename
    stem = filepath.stem  # e.g. 'pant_short6000'
    m_name = re.match(r"([a-z_]+?)(\d+)$", stem)
    if not m_name:
        # Try without trailing digits
        m_name = re.match(r"([a-z_]+?)(\d+)", stem)
    if m_name:
        garment_type = m_name.group(1).rstrip("_")
        steps = int(m_name.group(2))
    else:
        garment_type = stem
        steps = 0

    return {
        "garment_type": garment_type,
        "steps": steps,
        "total_success_rate": total_success_rate,
        "garments": garments,
    }


def classify_garment(name: str) -> bool:
    """Return True if garment is 'seen' (contains 'Seen'), False if 'unseen'."""
    return "Seen" in name


def compute_summary(result: dict) -> dict:
    """Compute seen/unseen averages for a single evaluation result."""
    seen_rates = []
    unseen_rates = []
    for name, rate in result["garments"]:
        if classify_garment(name):
            seen_rates.append(rate)
        else:
            unseen_rates.append(rate)

    seen_avg = sum(seen_rates) / len(seen_rates) if seen_rates else 0.0
    unseen_avg = sum(unseen_rates) / len(unseen_rates) if unseen_rates else 0.0
    overall_avg = (seen_avg + unseen_avg) / 2 if (seen_rates and unseen_rates) else seen_avg

    return {
        "total": result["total_success_rate"],
        "seen_avg": seen_avg,
        "unseen_avg": unseen_avg,
        "mean_seen_unseen": overall_avg,
        "seen_count": len(seen_rates),
        "unseen_count": len(unseen_rates),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation logs")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "logs"),
        help="Directory containing log files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "outputs" / "eval_analysis"),
        help="Directory to save output plots and CSV",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all log files matching <garment><steps>.log pattern
    log_files = sorted(log_dir.glob("*.log"))
    # Filter out generic eval logs (only keep ones with garment name + digits)
    eval_logs = [f for f in log_files if re.match(r"[a-z_]+\d+\.log$", f.name)]

    if not eval_logs:
        print(f"No evaluation log files found in {log_dir}")
        sys.exit(1)

    print(f"Found {len(eval_logs)} evaluation log files in {log_dir}\n")

    # Parse all logs
    results = []
    for lf in eval_logs:
        result = parse_log_file(lf)
        if result is None:
            print(f"  WARNING: Could not parse {lf.name}, skipping")
            continue
        results.append(result)

    # Group by garment type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["garment_type"]].append(r)

    # Print summary table
    print("=" * 80)
    print(f"{'Garment Type':<15} {'Steps':>8} {'Total':>8} {'Seen Avg':>10} {'Unseen Avg':>12} {'Mean(S,U)':>10}")
    print("=" * 80)

    csv_lines = ["garment_type,steps,total,seen_avg,unseen_avg,mean_seen_unseen"]

    for gtype in sorted(by_type.keys()):
        entries = sorted(by_type[gtype], key=lambda x: x["steps"])
        print(f"\n--- {gtype} ---")
        for e in entries:
            s = compute_summary(e)
            print(
                f"  {gtype:<13} {e['steps']:>8} "
                f"{s['total']:>7.2f}% {s['seen_avg']:>9.2f}% "
                f"{s['unseen_avg']:>11.2f}% {s['mean_seen_unseen']:>9.2f}%"
            )
            csv_lines.append(
                f"{gtype},{e['steps']},{s['total']:.2f},{s['seen_avg']:.2f},"
                f"{s['unseen_avg']:.2f},{s['mean_seen_unseen']:.2f}"
            )

    # Save CSV
    csv_path = output_dir / "eval_summary.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"\nCSV saved to: {csv_path}")

    # Plot training curves per garment type
    for gtype, entries in by_type.items():
        entries = sorted(entries, key=lambda x: x["steps"])
        steps_list = [e["steps"] for e in entries]
        summaries = [compute_summary(e) for e in entries]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            steps_list,
            [s["total"] for s in summaries],
            "o-",
            label="Total Success Rate",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            steps_list,
            [s["seen_avg"] for s in summaries],
            "s--",
            label="Seen Avg",
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            steps_list,
            [s["unseen_avg"] for s in summaries],
            "^--",
            label="Unseen Avg",
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            steps_list,
            [s["mean_seen_unseen"] for s in summaries],
            "d:",
            label="Mean(Seen, Unseen)",
            linewidth=1.5,
            markersize=6,
        )

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(f"{gtype} — Evaluation Success Rate vs Training Steps", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps_list)
        ax.set_xticklabels([f"{s // 1000}k" if s >= 1000 else str(s) for s in steps_list])

        fig.tight_layout()
        plot_path = output_dir / f"{gtype}_training_curve.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to: {plot_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
