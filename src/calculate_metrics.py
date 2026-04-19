#!/usr/bin/env python3
"""
Calculate ranking metrics for CSV files.

Supports both:
- Ranked CSV files (columns contain rank positions: 1, 2, 3, ...)
- Scored CSV files (columns contain scores, ending with _score suffix)

For scored files, converts scores to ranks first (lower score = higher rank number).

Calculates for each column:
- Normalized mean rank (0-1, where 1 = always first, 0 = always last)
- Top-2 hit rate (%)
- Top-3 hit rate (%)

Metrics computed for:
- Overall (all rows)
- style_theme (rows 1-25)
- not_style_theme (rows 26-50)
- style_not_theme (rows 51-75)
- not_style_not_theme (rows 76-100)
- General (rows 101+)
- theme (rows 1-50)
- no_theme (rows 51+)
- style (rows 1-25 and 51-75)
- no_style (rows 26-50 and 76-100)
- no_theme_no_style_general (rows 76+)

Total: 3 metrics × 11 groups = 33 metrics per column

Output: {original_csv_name}_metrics.csv
"""

import os
import csv
import argparse
import sys
from typing import List, Dict, Tuple


ALPHA_VALUES = [0.0, 0.5, 1.0]
WEIGHTS = {"style_match": 0.25, "topic_depth": 0.25, "topic_style_coherence": 0.50}


def is_detailed_csv(headers: List[str]) -> bool:
    """Check if CSV is a scored_detailed format (has _topic_rel, _style, _depth, _coherence columns)."""
    return any(h.endswith('_coherence') for h in headers) and any(h.endswith('_topic_rel') for h in headers)


def is_scored_csv(headers: List[str]) -> bool:
    """Check if CSV contains score columns (ending with _score)."""
    return any(header.endswith('_score') for header in headers)


def extract_score_columns(headers: List[str]) -> List[str]:
    """Extract column names that end with _score."""
    return [h for h in headers if h.endswith('_score')]


def convert_scores_to_ranks(score_row: List[float]) -> List[int]:
    """
    Convert scores to ranks. Lower score = higher rank number.
    Ties get average rank.
    
    Args:
        score_row: List of scores for a single row
        
    Returns:
        List of ranks (1 = best, N = worst)
    """
    # Create list of (score, original_index) pairs
    indexed_scores = [(score, idx) for idx, score in enumerate(score_row)]
    
    # Sort by score descending (highest score = best = rank 1)
    sorted_scores = sorted(indexed_scores, key=lambda x: -x[0])
    
    # Assign ranks, handling ties with average rank
    ranks = [0] * len(score_row)
    i = 0
    while i < len(sorted_scores):
        # Find all items with same score (ties)
        j = i
        while j < len(sorted_scores) and sorted_scores[j][0] == sorted_scores[i][0]:
            j += 1
        
        # Assign average rank to all tied items
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            original_idx = sorted_scores[k][1]
            ranks[original_idx] = int(round(avg_rank))
        
        i = j
    
    return ranks


def read_ranked_csv(filepath: str) -> Tuple[List[str], List[List[int]]]:
    """
    Read a ranked CSV file.
    
    Returns:
        (column_names, rank_data) where rank_data is list of rank rows
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        raise ValueError(f"CSV file must have at least header and one data row")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    # Convert to integers
    rank_data = []
    for row_idx, row in enumerate(data_rows, start=2):
        try:
            rank_row = [int(cell) for cell in row if cell.strip()]
            rank_data.append(rank_row)
        except ValueError as e:
            print(f"Warning: Skipping row {row_idx} - cannot convert to ranks: {e}")
            continue
    
    return headers, rank_data


def read_scored_csv(filepath: str) -> Tuple[List[str], List[List[int]]]:
    """
    Read a scored CSV file and convert scores to ranks.
    
    Returns:
        (column_names, rank_data) where rank_data is list of rank rows
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        raise ValueError(f"CSV file must have at least header and one data row")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    # Extract score columns
    score_columns = extract_score_columns(headers)
    if not score_columns:
        raise ValueError(f"No score columns (ending with _score) found in CSV")
    
    print(f"Found {len(score_columns)} score columns")
    print(f"Score columns: {', '.join(score_columns)}")
    
    # Convert scores to ranks
    rank_data = []
    for row_idx, row in enumerate(data_rows, start=2):
        try:
            score_row = [float(cell) for cell in row if cell.strip()]
            rank_row = convert_scores_to_ranks(score_row)
            rank_data.append(rank_row)
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping row {row_idx} - cannot convert scores: {e}")
            continue
    
    # Clean column names (remove _score suffix)
    clean_headers = [h.replace('_score', '') for h in score_columns]
    
    return clean_headers, rank_data


def compute_final_score(style: float, depth: float, coherence: float, alpha: float, topic_relevant: bool) -> float:
    """Compute final score with given alpha (geometric mean fraction)."""
    if not topic_relevant:
        return 0.0
    arithmetic = (
        style * WEIGHTS["style_match"] +
        depth * WEIGHTS["topic_depth"] +
        coherence * WEIGHTS["topic_style_coherence"]
    )
    geometric = (style * depth * coherence) ** (1/3) if (style > 0 and depth > 0 and coherence > 0) else 0.0
    final = (1 - alpha) * arithmetic + alpha * geometric
    return max(0.0, min(1.0, final))


def read_detailed_csv(filepath: str) -> Tuple[List[str], List[List[Dict[str, float]]]]:
    """
    Read a scored_detailed CSV file.
    
    Returns:
        (column_names, rows_data) where rows_data[row][col] = {topic_rel, style, depth, coherence}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        raise ValueError("CSV file must have at least header and one data row")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    # Detect columns: every 6 headers is one logical column
    # pattern: name, name_score, name_topic_rel, name_style, name_depth, name_coherence
    column_names = []
    col_indices = []  # list of (topic_rel_idx, style_idx, depth_idx, coherence_idx)
    
    i = 0
    while i < len(headers):
        if i + 5 < len(headers) and headers[i+2].endswith('_topic_rel'):
            col_name = headers[i]
            col_indices.append((i+2, i+3, i+4, i+5))  # topic_rel, style, depth, coherence
            column_names.append(col_name)
            i += 6
        else:
            i += 1
    
    print(f"Found {len(column_names)} detailed columns: {', '.join(column_names)}")
    
    rows_data = []
    for row in data_rows:
        row_entries = []
        for topic_rel_idx, style_idx, depth_idx, coherence_idx in col_indices:
            try:
                topic_rel = row[topic_rel_idx].strip() == '1' if topic_rel_idx < len(row) else False
                style = float(row[style_idx]) if style_idx < len(row) and row[style_idx].strip() else 0.0
                depth = float(row[depth_idx]) if depth_idx < len(row) and row[depth_idx].strip() else 0.0
                coherence = float(row[coherence_idx]) if coherence_idx < len(row) and row[coherence_idx].strip() else 0.0
                row_entries.append({
                    'topic_relevant': topic_rel,
                    'style': style,
                    'depth': depth,
                    'coherence': coherence,
                })
            except (ValueError, IndexError):
                row_entries.append({
                    'topic_relevant': False,
                    'style': 0.0,
                    'depth': 0.0,
                    'coherence': 0.0,
                })
        rows_data.append(row_entries)
    
    return column_names, rows_data


def get_groups(num_rows: int) -> Tuple[Dict, Dict]:
    """Return (simple_groups, compound_groups) definitions."""
    simple_groups = {
        'Overall': (0, num_rows),
        '(1-25) Style, Theme': (0, 25),
        '(26-50) not Style, Theme': (25, 50),
        '(51-75) Style, not Theme': (50, 75),
        '(76-100) not Style, not Theme': (75, 100),
        '(101+) General': (100, num_rows),
        '(1-50) Theme': (0, 50),
        '(51+) no Theme': (50, num_rows),
        '(76+) no Theme, no Style, General': (75, num_rows),
    }
    compound_groups = {
        '(1-25, 51-75) Style': [(0, 25), (50, 75)],
        '(26-50, 76-100) no Style': [(25, 50), (75, 100)],
    }
    return simple_groups, compound_groups


def _collect_rows(data, num_rows, start, end):
    end = min(end, num_rows)
    if start >= num_rows:
        return []
    return data[start:end]


DETAILED_FIELDS_BASIC = ['mean_topic_rel', 'mean_style', 'mean_depth', 'mean_coherence', 'mean_score']
DETAILED_FIELDS_ALPHA = [f'mean_score_a{a:.1f}' for a in ALPHA_VALUES]


def calculate_detailed_metrics(
    column_names: List[str],
    rows_data: List[List[Dict[str, float]]],
    use_alpha: bool = False
) -> Tuple[List[Dict], List[str]]:
    """
    Calculate detailed score metrics for each column across groups.
    
    Default mode: 5 metrics per column per group:
      mean_topic_rel, mean_style, mean_depth, mean_coherence,
      mean_score = mean(topic_rel * (style + depth + coherence) / 3)
    
    Alpha mode (--alpha): 3 metrics per column per group:
      mean_score_a0.0, mean_score_a0.5, mean_score_a1.0
    
    Returns:
        (results, fieldnames)
    """
    num_rows = len(rows_data)
    simple_groups, compound_groups = get_groups(num_rows)
    
    results = []
    
    def process_group(group_name, group_data):
        if not group_data:
            return
        print(f"\nProcessing {group_name} ({len(group_data)} rows)...")
        for col_idx, col_name in enumerate(column_names):
            entries = [row[col_idx] for row in group_data if col_idx < len(row)]
            if not entries:
                continue
            
            row_result = {'column': col_name, 'group': group_name}
            
            if use_alpha:
                for alpha in ALPHA_VALUES:
                    scores = [
                        compute_final_score(e['style'], e['depth'], e['coherence'], alpha, e['topic_relevant'])
                        for e in entries
                    ]
                    row_result[f'mean_score_a{alpha:.1f}'] = round(sum(scores) / len(scores), 4)
            else:
                topic_rels = [1.0 if e['topic_relevant'] else 0.0 for e in entries]
                styles = [e['style'] for e in entries]
                depths = [e['depth'] for e in entries]
                coherences = [e['coherence'] for e in entries]
                final_scores = [
                    tr * (s + d + c) / 3.0
                    for tr, s, d, c in zip(topic_rels, styles, depths, coherences)
                ]
                n = len(entries)
                row_result['mean_topic_rel'] = round(sum(topic_rels) / n, 4)
                row_result['mean_style'] = round(sum(styles) / n, 4)
                row_result['mean_depth'] = round(sum(depths) / n, 4)
                row_result['mean_coherence'] = round(sum(coherences) / n, 4)
                row_result['mean_score'] = round(sum(final_scores) / n, 4)
            
            results.append(row_result)
    
    for group_name, (start, end) in simple_groups.items():
        group_data = _collect_rows(rows_data, num_rows, start, end)
        process_group(group_name, group_data)
    
    for group_name, ranges in compound_groups.items():
        group_data = []
        for start, end in ranges:
            group_data.extend(_collect_rows(rows_data, num_rows, start, end))
        process_group(group_name, group_data)
    
    fieldnames = DETAILED_FIELDS_ALPHA if use_alpha else DETAILED_FIELDS_BASIC
    return results, fieldnames


def save_detailed_metrics(results: List[Dict], fieldnames: List[str], output_path: str):
    """Save detailed score metrics to CSV file."""
    if not results:
        print("No metrics to save")
        return
    
    all_fields = ['column', 'group'] + fieldnames
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nMetrics saved to: {output_path}")


def calculate_column_metrics(ranks: List[int], num_columns: int) -> Dict[str, float]:
    """
    Calculate metrics for a single column.
    
    Args:
        ranks: List of ranks for this column across rows
        num_columns: Total number of columns (for normalization)
        
    Returns:
        Dictionary with metrics: normalized_rank, top2_rate, top3_rate
    """
    if not ranks:
        return {
            'normalized_rank': 0.0,
            'top2_rate': 0.0,
            'top3_rate': 0.0
        }
    
    # Mean rank (1 = best, N = worst)
    mean_rank = sum(ranks) / len(ranks)
    
    # Normalize to 0-1 where 1 = always first (rank=1), 0 = always last (rank=N)
    # Formula: (N - mean_rank) / (N - 1)
    if num_columns > 1:
        normalized_rank = (num_columns - mean_rank) / (num_columns - 1)
    else:
        normalized_rank = 1.0
    
    # Top-2 hit rate
    top2_count = sum(1 for r in ranks if r <= 2)
    top2_rate = (top2_count / len(ranks))
    
    # Top-3 hit rate
    top3_count = sum(1 for r in ranks if r <= 3)
    top3_rate = (top3_count / len(ranks))
    
    return {
        'normalized_rank': normalized_rank,
        'top2_rate': top2_rate,
        'top3_rate': top3_rate
    }


def calculate_metrics_for_groups(
    column_names: List[str], 
    rank_data: List[List[int]]
) -> List[Dict]:
    """
    Calculate metrics for all columns across all groups.
    
    Groups:
    - Overall: all rows
    - style_theme: rows 0-24 (1-25)
    - not_style_theme: rows 25-49 (26-50)
    - style_not_theme: rows 50-74 (51-75)
    - not_style_not_theme: rows 75-99 (76-100)
    - General: rows 100+ (101+)
    - theme: rows 0-49 (1-50)
    - no_theme: rows 50+ (51+)
    - style: rows 0-24 and 50-74 (1-25 and 51-75)
    - no_style: rows 25-49 and 75-99 (26-50 and 76-100)
    - no_theme_no_style_general: rows 75+ (76+)
    
    Returns:
        List of metric dictionaries
    """
    num_columns = len(column_names)
    num_rows = len(rank_data)
    
    simple_groups, compound_groups = get_groups(num_rows)
    
    results = []
    
    # Process simple groups
    for group_name, (start_idx, end_idx) in simple_groups.items():
        # Skip if group is empty
        if start_idx >= num_rows:
            continue
        
        end_idx = min(end_idx, num_rows)
        group_data = rank_data[start_idx:end_idx]
        
        if not group_data:
            continue
        
        print(f"\nProcessing {group_name} ({len(group_data)} rows)...")
        
        # Calculate metrics for each column
        for col_idx, col_name in enumerate(column_names):
            # Extract ranks for this column
            column_ranks = [row[col_idx] for row in group_data if col_idx < len(row)]
            
            if not column_ranks:
                continue
            
            metrics = calculate_column_metrics(column_ranks, num_columns)
            
            results.append({
                'column': col_name,
                'group': group_name,
                'normalized_rank': round(metrics['normalized_rank'], 2),
                'top2_rate': round(metrics['top2_rate'], 2),
                'top3_rate': round(metrics['top3_rate'], 2)
            })
    
    # Process compound groups
    for group_name, ranges in compound_groups.items():
        # Collect data from all ranges
        group_data = []
        for start_idx, end_idx in ranges:
            if start_idx >= num_rows:
                continue
            end_idx = min(end_idx, num_rows)
            group_data.extend(rank_data[start_idx:end_idx])
        
        if not group_data:
            continue
        
        print(f"\nProcessing {group_name} ({len(group_data)} rows)...")
        
        # Calculate metrics for each column
        for col_idx, col_name in enumerate(column_names):
            # Extract ranks for this column
            column_ranks = [row[col_idx] for row in group_data if col_idx < len(row)]
            
            if not column_ranks:
                continue
            
            metrics = calculate_column_metrics(column_ranks, num_columns)
            
            results.append({
                'column': col_name,
                'group': group_name,
                'normalized_rank': round(metrics['normalized_rank'], 2),
                'top2_rate': round(metrics['top2_rate'], 2),
                'top3_rate': round(metrics['top3_rate'], 2)
            })
    
    return results


def save_metrics(results: List[Dict], output_path: str):
    """Save metrics to CSV file."""
    if not results:
        print("No metrics to save")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=['column', 'group', 'normalized_rank', 'top2_rate', 'top3_rate']
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nMetrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate ranking metrics from CSV files (ranked or scored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'csv_file',
        help='Input CSV file (ranked or scored format)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output CSV file (default: {input}_metrics.csv)'
    )
    parser.add_argument(
        '--alpha',
        action='store_true',
        default=False,
        help='For detailed CSVs: use alpha-blend mode (a=0, 0.5, 1) instead of per-criteria means'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        output_path = f"{base_name}_metrics.csv"
    
    print(f"Input file: {args.csv_file}")
    print(f"Output file: {output_path}")
    
    # Read CSV and detect format
    with open(args.csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
    
    if is_detailed_csv(headers):
        print("\nDetected: SCORED DETAILED CSV")
        column_names, rows_data = read_detailed_csv(args.csv_file)
        
        print(f"\nColumns: {', '.join(column_names)}")
        print(f"Number of data rows: {len(rows_data)}")
        
        if args.alpha:
            print("Mode: alpha-blend (a=0, 0.5, 1)")
        else:
            print("Mode: per-criteria means (topic_rel, style, depth, coherence, score)")
        
        results, fieldnames = calculate_detailed_metrics(column_names, rows_data, use_alpha=args.alpha)
        save_detailed_metrics(results, fieldnames, output_path)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        overall_results = [r for r in results if r['group'] == 'Overall']
        if overall_results:
            if args.alpha:
                print("\nOverall Mean Scores (alpha = fraction of geometric mean):")
                print(f"{'Column':<30} {'a=0.0':>10} {'a=0.5':>10} {'a=1.0':>10}")
                print("-" * 62)
                sorted_results = sorted(overall_results, key=lambda x: x['mean_score_a0.5'], reverse=True)
                for r in sorted_results:
                    print(f"{r['column']:<30} {r['mean_score_a0.0']:>10.4f} {r['mean_score_a0.5']:>10.4f} {r['mean_score_a1.0']:>10.4f}")
            else:
                print("\nOverall Per-Criteria Means:")
                print(f"{'Column':<30} {'TopicRel':>10} {'Style':>10} {'Depth':>10} {'Coher':>10} {'Score':>10}")
                print("-" * 82)
                sorted_results = sorted(overall_results, key=lambda x: x['mean_score'], reverse=True)
                for r in sorted_results:
                    print(f"{r['column']:<30} {r['mean_topic_rel']:>10.4f} {r['mean_style']:>10.4f} {r['mean_depth']:>10.4f} {r['mean_coherence']:>10.4f} {r['mean_score']:>10.4f}")
    
    elif is_scored_csv(headers):
        print("\nDetected: SCORED CSV (contains _score columns)")
        print("Converting scores to ranks...")
        column_names, rank_data = read_scored_csv(args.csv_file)
        _run_rank_metrics(column_names, rank_data, output_path)
    else:
        print("\nDetected: RANKED CSV")
        column_names, rank_data = read_ranked_csv(args.csv_file)
        _run_rank_metrics(column_names, rank_data, output_path)


def _run_rank_metrics(column_names, rank_data, output_path):
    """Run rank-based metrics and print summary."""
    print(f"\nColumns: {', '.join(column_names)}")
    print(f"Number of data rows: {len(rank_data)}")
    
    # Calculate metrics
    results = calculate_metrics_for_groups(column_names, rank_data)
    
    # Save results
    save_metrics(results, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Group by column and show overall metrics
    overall_results = [r for r in results if r['group'] == 'Overall']
    if overall_results:
        print("\nOverall Rankings (Normalized Rank: 1.0 = best, 0.0 = worst):")
        print(f"{'Column':<30} {'Norm.Rank':>10} {'Top-2%':>10} {'Top-3%':>10}")
        print("-" * 62)
        
        # Sort by normalized rank descending
        sorted_results = sorted(overall_results, key=lambda x: x['normalized_rank'], reverse=True)
        for r in sorted_results:
            print(f"{r['column']:<30} {r['normalized_rank']:>10.4f} {r['top2_rate']*100:>9.1f}% {r['top3_rate']*100:>9.1f}%")


if __name__ == "__main__":
    main()
