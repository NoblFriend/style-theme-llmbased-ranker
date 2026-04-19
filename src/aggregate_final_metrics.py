#!/usr/bin/env python3
"""
Aggregate metrics across all final_* folders in results/.

For each folder:
  1. Runs calculate_metrics.py on ranked.csv and scored_detailed.csv
  2. Extracts the "(76+) no Theme, no Style, General" group
  3. Renames the unique non-baseline column to "OURS"

Output: results/aggregated_metrics.csv
  - Rows: OURS, halfsum, instruct, k_lora, style_instruct, super_baseline, theme_instruct
  - Columns: grouped by style, then ranker metrics first, then scorer metrics
"""

import argparse
import os
import csv
import subprocess
import sys
from collections import OrderedDict

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "aggregated_metrics.csv")

BASELINES = {"halfsum", "instruct", "k_lora", "style_instruct", "super_baseline", "theme_instruct"}
ROW_ORDER = ["OURS", "super_baseline", "k_lora", "halfsum", "instruct", "style_instruct", "theme_instruct"]

TARGET_GROUP = "(76+) no Theme, no Style, General"

RANK_METRICS = ["normalized_rank", "top2_rate", "top3_rate"]
SCORE_METRICS = ["mean_topic_rel", "mean_style", "mean_depth", "mean_coherence", "mean_score"]

METRIC_DISPLAY = {
    "normalized_rank": r"Norm.\ rank",
    "top2_rate": "Top-2 rate",
    "top3_rate": "Top-3 rate",
    "mean_topic_rel": "Topic hit rate",
    "mean_style": "Style Q",
    "mean_depth": "Topic Q",
    "mean_coherence": "Coherence Q",
    "mean_score": "Score",
}


def run_metrics(csv_path, extra_args=None):
    """Run calculate_metrics.py and return the metrics CSV path."""
    base = os.path.splitext(csv_path)[0]
    metrics_path = f"{base}_metrics.csv"
    
    cmd = [sys.executable, "calculate_metrics.py", csv_path]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return metrics_path


def read_metrics_csv(path):
    """Read metrics CSV into list of dicts."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_group(rows, group_name):
    """Extract rows matching a specific group."""
    return [r for r in rows if r['group'] == group_name]


def normalize_column_name(name, baselines):
    """Rename non-baseline column to OURS."""
    if name in baselines:
        return name
    return "OURS"


def main():
    # Find all final_* folders
    folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if d.startswith("final") and os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])
    
    if not folders:
        print("No final_* folders found in results/")
        return
    
    print(f"Found {len(folders)} folders: {', '.join(folders)}")
    
    # Collect data: style -> {metric_type -> {column_name -> {metric_name -> value}}}
    all_data = OrderedDict()
    
    for folder in folders:
        style = folder.replace("final_", "")
        folder_path = os.path.join(RESULTS_DIR, folder)
        ranked_csv = os.path.join(folder_path, "ranked.csv")
        detailed_csv = os.path.join(folder_path, "scored_detailed.csv")
        
        print(f"\n{'='*60}")
        print(f"Processing: {folder} (style: {style})")
        print(f"{'='*60}")
        
        style_data = {"ranked": {}, "scored": {}}
        
        # 1. Ranked metrics
        if os.path.exists(ranked_csv):
            print(f"\n  [Ranked]")
            metrics_path = run_metrics(ranked_csv)
            rows = read_metrics_csv(metrics_path)
            group_rows = extract_group(rows, TARGET_GROUP)
            
            for r in group_rows:
                col = normalize_column_name(r['column'], BASELINES)
                style_data["ranked"][col] = {m: r[m] for m in RANK_METRICS}
        else:
            print(f"  WARNING: {ranked_csv} not found, skipping")
        
        # 2. Scored detailed metrics
        if os.path.exists(detailed_csv):
            print(f"\n  [Scored Detailed]")
            metrics_path = run_metrics(detailed_csv)
            rows = read_metrics_csv(metrics_path)
            group_rows = extract_group(rows, TARGET_GROUP)
            
            for r in group_rows:
                col = normalize_column_name(r['column'], BASELINES)
                style_data["scored"][col] = {m: r[m] for m in SCORE_METRICS}
        else:
            print(f"  WARNING: {detailed_csv} not found, skipping")
        
        all_data[style] = style_data
    
    # Build output CSV (transposed):
    # Row 1: style labels (repeated for each baseline column)
    # Row 2: baseline/OURS names as column headers
    # Remaining rows: one per metric
    
    styles = list(all_data.keys())
    
    # All metric names in order: ranked first, then scored
    all_metrics = [("rank", m) for m in RANK_METRICS] + [("score", m) for m in SCORE_METRICS]
    
    # Build header rows and data
    # Columns: metric_name | style1_OURS | style1_halfsum | ... | style2_OURS | ...
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Row 1: style labels
        style_row = [""]
        for style in styles:
            for _ in ROW_ORDER:
                style_row.append(style)
        writer.writerow(style_row)
        
        # Row 2: column (baseline) names
        col_row = ["metric"]
        for style in styles:
            for col_name in ROW_ORDER:
                col_row.append(col_name)
        writer.writerow(col_row)
        
        # Data rows: one per metric
        def fmt(val):
            """Round to 2 decimals, strip leading zero: 0.41 -> .41"""
            if val == "":
                return ""
            try:
                v = float(val)
                s = f"{v:.2f}"
                if s.startswith("0."):
                    return s[1:]  # ".41"
                elif s.startswith("-0."):
                    return "-" + s[2:]  # "-.41"
                return s
            except (ValueError, TypeError):
                return val
        
        for metric_type, metric_name in all_metrics:
            row = [f"{metric_type}_{metric_name}"]
            for style in styles:
                style_data = all_data[style]
                for col_name in ROW_ORDER:
                    if metric_type == "rank":
                        val = style_data.get("ranked", {}).get(col_name, {}).get(metric_name, "")
                    else:
                        val = style_data.get("scored", {}).get(col_name, {}).get(metric_name, "")
                    row.append(fmt(val))
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Aggregated metrics saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\nStyles: {', '.join(styles)}")
    print(f"Metric rows: {len(all_metrics)}")
    print(f"Columns per style: {len(ROW_ORDER)}")
    print(f"Total columns: {len(styles) * len(ROW_ORDER) + 1}")

    return all_data, styles


def output_latex(all_data, styles):
    """Print LaTeX table body.

    For each metric, prints the display name followed by one line per style
    with `& val & val ...`.  The best value in each style group (excluding
    super_baseline) is wrapped in \\textbf{}.
    """
    all_metrics = [("rank", m) for m in RANK_METRICS] + [("score", m) for m in SCORE_METRICS]

    blocks = []
    for metric_type, metric_name in all_metrics:
        display = METRIC_DISPLAY.get(metric_name, metric_name)
        lines = [display]

        for si, style in enumerate(styles):
            style_data = all_data[style]
            vals = []
            for col_name in ROW_ORDER:
                src = "ranked" if metric_type == "rank" else "scored"
                raw = style_data.get(src, {}).get(col_name, {}).get(metric_name, "")
                try:
                    vals.append(float(raw))
                except (ValueError, TypeError):
                    vals.append(None)

            # Best among non-super_baseline (higher = better for all metrics)
            candidates = [
                v for col, v in zip(ROW_ORDER, vals)
                if col != "super_baseline" and v is not None
            ]
            best_val = max(candidates) if candidates else None

            # Format each value
            parts = []
            for col_name, v in zip(ROW_ORDER, vals):
                if v is None:
                    parts.append("")
                else:
                    s = f"{v:.2f}"
                    if col_name != "super_baseline" and best_val is not None and abs(v - best_val) < 1e-9:
                        s = f"\\textbf{{{s}}}"
                    parts.append(s)

            suffix = " \\\\" if si == len(styles) - 1 else ""
            lines.append("& " + " & ".join(parts) + suffix)

        blocks.append("\n".join(lines))

    print("\n".join(blocks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate final metrics")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table body")
    args = parser.parse_args()

    all_data, styles = main()
    if args.latex:
        print("\n" + "=" * 60)
        print("LaTeX table body:")
        print("=" * 60 + "\n")
        output_latex(all_data, styles)
