#!/usr/bin/env python3
"""
Pipeline wrapper: Process multiple folders through scoring → ranking → metrics.

For each folder:
  1. Run scoring from folder → scored.csv, scored_detailed.csv
  2. Run ranking from folder → ranked.csv
  3. Calculate metrics on both → *_metrics.csv
  4. JOIN metrics by group (method, data_type)
  5. Save to results/{folder_name}/metrics.csv

Output structure:
  - Each folder gets results/{folder_name}/ with:
    - scored.csv, scored_detailed.csv, scored_detailed_metrics.csv
    - ranked.csv, ranked_metrics.csv
    - metrics.csv (final joined result)
  
  - metrics.csv structure: rows are groups like "(1-25) Style, Theme"
    columns: method, data_type, ranker_* (3 metrics), scorer_* (5 metrics)

Usage:
  python process_pipeline.py --topic astrophysics
  python process_pipeline.py data/folder1 data/folder2 --topic animals
  python process_pipeline.py --output-base ./results --topic physics
"""

import os
import sys
import csv
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class FolderMetadata:
    """Metadata extracted from folder name and structure."""
    path: str
    folder_name: str
    method: str  # extracted method (e.g., 'qwen', 'llama')
    data_type: str  # extracted data type (e.g., 'sad', 'agit')
    style: Optional[str] = None
    topic: Optional[str] = None  # global topic (set by pipeline, not extracted)


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Extract method and data type from folder name.
    
    Examples:
      "09_02_agit" → ("", "agit")
      "02_03_qwen_sad" → ("qwen", "sad")
      "16_03_llama_aggr" → ("llama", "aggr")
      "10_02_qwen_sad" → ("qwen", "sad")
    """
    # Pattern: digit_digit[_method][_datatype] or digit_digit_datatype or digit_digit_method_datatype
    parts = folder_name.split('_')
    
    # Skip date part (first 2 parts are usually dates like "09_02")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        remaining = parts[2:]
    else:
        remaining = parts
    
    if len(remaining) == 0:
        return "", ""
    elif len(remaining) == 1:
        # Could be either method or data_type, guess based on known patterns
        data_type = remaining[0]
        return "", data_type
    else:
        # Usually: [method, data_type, ...]
        method = remaining[0]
        data_type = remaining[1]
        return method, data_type


def infer_style_from_folder(folder_name: str) -> Optional[str]:
    """Try to infer style name from folder name."""
    # Common patterns in folder structure
    style_mappings = {
        'sad': 'melancholic',
        'agit': 'agitational',
        'aggr': 'aggressive',
        'cheer': 'cheerful',
        'refl': 'reflective',
    }
    
    for key, style in style_mappings.items():
        if key in folder_name.lower():
            return style
    
    return None


def discover_folders(base_dir: str = "data") -> List[FolderMetadata]:
    """Discover all data folders."""
    folders = []
    
    if not os.path.isdir(base_dir):
        print(f"Warning: Base directory {base_dir} not found")
        return folders
    
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        method, data_type = parse_folder_name(folder_name)
        style = infer_style_from_folder(folder_name)
        
        folders.append(FolderMetadata(
            path=folder_path,
            folder_name=folder_name,
            method=method or "default",
            data_type=data_type or "unknown",
            style=style or data_type,
            topic=None  # Could be inferred from folder structure
        ))
    
    return folders


def run_scoring(folder: FolderMetadata, output_dir: str) -> Tuple[str, str]:
    """
    Run run_scoring_from_folder.py.
    
    Returns:
        (scored_detailed_path, scored_path) or raises if failed
    """
    print(f"\n{'='*60}")
    print(f"SCORING: {folder.folder_name}")
    print(f"{'='*60}")
    
    output_prefix = os.path.join(output_dir, folder.folder_name, "scored")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_openai_from_folder.py"),
        folder.path,
        "--mode", "scoring",
        "--topic", folder.topic or "",
        "--style", folder.style or folder.data_type,
        "--scoring-output", output_prefix,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in scoring:\n{result.stderr}")
        raise RuntimeError(f"Scoring failed for {folder.folder_name}")
    
    print(result.stdout)
    
    detailed_path = f"{output_prefix}_detailed.csv"
    simple_path = f"{output_prefix}.csv"
    
    if not os.path.exists(detailed_path):
        raise RuntimeError(f"Expected output not found: {detailed_path}")
    
    return detailed_path, simple_path


def run_ranking(folder: FolderMetadata, output_dir: str) -> str:
    """
    Run run_evaluation_from_folder.py.
    
    Returns:
        ranked_path or raises if failed
    """
    print(f"\n{'='*60}")
    print(f"RANKING: {folder.folder_name}")
    print(f"{'='*60}")
    
    output_file = os.path.join(output_dir, folder.folder_name, "ranked.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_openai_from_folder.py"),
        folder.path,
        "--mode", "ranking",
        "--topic", folder.topic or "",
        "--style", folder.style or folder.data_type,
        "--ranking-output", output_file,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in ranking:\n{result.stderr}")
        raise RuntimeError(f"Ranking failed for {folder.folder_name}")
    
    print(result.stdout)
    
    if not os.path.exists(output_file):
        raise RuntimeError(f"Expected output not found: {output_file}")
    
    return output_file


def run_calculate_metrics(csv_path: str) -> str:
    """
    Run calculate_metrics.py on a CSV file.
    
    Returns:
        metrics_csv_path or raises if failed
    """
    base = os.path.splitext(csv_path)[0]
    metrics_path = f"{base}_metrics.csv"
    
    cmd = [sys.executable, str(SCRIPT_DIR / "calculate_metrics.py"), csv_path]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR calculating metrics:\n{result.stderr}")
        raise RuntimeError(f"Metrics calculation failed for {csv_path}")
    
    print(result.stdout)
    
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"Expected output not found: {metrics_path}")
    
    return metrics_path


def read_metrics_csv(metrics_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Read metrics CSV and return data keyed by (column, group).
    
    Returns:
        Dict[(column, group)] = Dict[metric_name] = value
    """
    result = {}
    
    if not os.path.exists(metrics_path):
        return result
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            column = row.get('column', '')
            group = row.get('group', '')
            
            if not column or not group:
                continue
            
            key = (column, group)
            # Store all metrics (skip column and group)
            metrics = {k: v for k, v in row.items() if k not in ('column', 'group')}
            result[key] = metrics
    
    return result


def process_folder(folder: FolderMetadata, output_base: str) -> bool:
    """
    Process a single folder through the full pipeline.
    
    1. Run scoring → scored.csv, scored_detailed.csv
    2. Run ranking → ranked.csv
    3. Calculate metrics on scored.csv and ranked.csv (ignore _detailed)
    4. JOIN metrics by (column, group) - first two columns
    5. Save to results/{folder_name}/metrics.csv
    
    Result: column, group, ranker_normalized_rank, ranker_top2_rate, ranker_top3_rate,
            scorer_normalized_rank, scorer_top2_rate, scorer_top3_rate
    """
    try:
        # Run scoring
        scored_detailed_path, scored_simple_path = run_scoring(folder, output_base)
        
        # Run ranking
        ranked_path = run_ranking(folder, output_base)
        
        # Calculate metrics for scored.csv (NOT _detailed)
        print(f"\n{'='*60}")
        print(f"CALCULATING METRICS (Scorer from scored.csv)")
        print(f"{'='*60}")
        scorer_metrics_path = run_calculate_metrics(scored_simple_path)
        scorer_metrics = read_metrics_csv(scorer_metrics_path)
        
        # Calculate metrics for ranked.csv
        print(f"\n{'='*60}")
        print(f"CALCULATING METRICS (Ranker from ranked.csv)")
        print(f"{'='*60}")
        ranker_metrics_path = run_calculate_metrics(ranked_path)
        ranker_metrics = read_metrics_csv(ranker_metrics_path)
        
        # Collect all (column, group) keys
        all_keys = set(scorer_metrics.keys()) | set(ranker_metrics.keys())
        
        if not all_keys:
            raise RuntimeError("No metrics found in either scorer or ranker files")
        
        # Build output
        output_dir = os.path.join(output_base, folder.folder_name)
        output_file = os.path.join(output_dir, "metrics.csv")
        
        fieldnames = [
            'column', 'group',
            'ranker_normalized_rank', 'ranker_top2_rate', 'ranker_top3_rate',
            'scorer_normalized_rank', 'scorer_top2_rate', 'scorer_top3_rate',
        ]
        
        rows = []
        for column, group in sorted(all_keys):
            ranker_data = ranker_metrics.get((column, group), {})
            scorer_data = scorer_metrics.get((column, group), {})
            
            row = {
                'column': column,
                'group': group,
                'ranker_normalized_rank': ranker_data.get('normalized_rank', ''),
                'ranker_top2_rate': ranker_data.get('top2_rate', ''),
                'ranker_top3_rate': ranker_data.get('top3_rate', ''),
                'scorer_normalized_rank': scorer_data.get('normalized_rank', ''),
                'scorer_top2_rate': scorer_data.get('top2_rate', ''),
                'scorer_top3_rate': scorer_data.get('top3_rate', ''),
            }
            rows.append(row)
        
        # Write output
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Successfully processed {folder.folder_name}")
        print(f"  Rows: {len(rows)}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to process {folder.folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Process multiple folders through scoring → ranking → metrics pipeline'
    )
    parser.add_argument(
        'folders',
        nargs='*',
        help='Folder paths to process (if empty, auto-discovers from data/)'
    )
    parser.add_argument(
        '--output-base', '-o',
        default='results',
        help='Base directory for all outputs (default: results/)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory to auto-discover folders from (default: data/)'
    )
    parser.add_argument(
        '--topic', '-t',
        default='astrophysics',
        help='Topic name for scoring and ranking (default: astrophysics)'
    )
    
    args = parser.parse_args()
    
    # Discover or use provided folders
    if args.folders:
        # Use provided folder paths
        folders = []
        for folder_path in args.folders:
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} is not a directory, skipping")
                continue
            folder_name = os.path.basename(os.path.normpath(folder_path))
            method, data_type = parse_folder_name(folder_name)
            style = infer_style_from_folder(folder_name)
            folders.append(FolderMetadata(
                path=folder_path,
                folder_name=folder_name,
                method=method or "default",
                data_type=data_type or "unknown",
                style=style or data_type,
                topic=args.topic,
            ))
    else:
        # Auto-discover
        folders = discover_folders(args.data_dir)
        # Set topic for all discovered folders
        for folder in folders:
            folder.topic = args.topic
    
    if not folders:
        print("No folders found to process")
        return
    
    print(f"Processing {len(folders)} folders:")
    print(f"Topic: {args.topic}")
    for f in folders:
        print(f"  - {f.folder_name} (method={f.method}, type={f.data_type}, style={f.style})")
    
    # Process all folders
    success_count = 0
    for folder in folders:
        if process_folder(folder, args.output_base):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"Processed: {success_count}/{len(folders)} folders successfully")
    print(f"Output directory: {args.output_base}/")
    print(f"{'='*60}")
    print(f"\nEach folder has a metrics.csv with groups as rows:")
    print(f"  {args.output_base}/*/metrics.csv")


if __name__ == "__main__":
    main()
