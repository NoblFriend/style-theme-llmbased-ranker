"""
Scoring from folder - converts folder of .txt files to CSV and runs pointwise scoring.
Uses async requests for parallel processing.
Outputs two result files:
  - scored_detailed.csv - all 5 criteria for each column
  - scored.csv - only final scores
"""
import os
import csv
import argparse
import asyncio
import aiohttp
import time
from typing import Optional

SERVER_URL = "http://localhost:1337/score"
# Number of concurrent requests (adjust based on GPU memory and model)
DEFAULT_CONCURRENCY = 16


def convert_folder_to_csv(folder_path: str) -> tuple[list[str], list[list[str]]]:
    """
    Read folder of .txt files into memory.
    Returns (column_names, rows) where each row is a list of texts.
    """
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return [], []
    
    print(f"Found {len(txt_files)} files: {', '.join(txt_files)}")
    
    # Column names = filenames without .txt
    column_names = [os.path.splitext(f)[0] for f in txt_files]
    
    # Read all files
    file_contents = {}
    max_lines = 0
    
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            col_name = os.path.splitext(txt_file)[0]
            file_contents[col_name] = lines
            max_lines = max(max_lines, len(lines))
    
    print(f"Max lines per file: {max_lines}")
    
    # Build rows
    rows = []
    for i in range(max_lines):
        row = []
        for col in column_names:
            lines = file_contents[col]
            row.append(lines[i] if i < len(lines) else "")
        rows.append(row)
    
    return column_names, rows


async def score_text_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    style: str, 
    topic: str, 
    text: str,
    row_idx: int,
    col_idx: int,
    col_name: str,
    progress: dict
) -> dict:
    """Score a single text via /score endpoint asynchronously."""
    async with semaphore:
        try:
            async with session.post(SERVER_URL, json={
                "style_name": style,
                "topic": topic,
                "text": text
            }, timeout=aiohttp.ClientTimeout(total=120)) as response:
                response.raise_for_status()
                result = await response.json()
                
                progress['done'] += 1
                score = result['final_score']
                print(f"  [{progress['done']}/{progress['total']}] Row {row_idx+1}, {col_name}: {score:.3f}")
                
                return {
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'text': text,
                    'final_score': result['final_score'],
                    'topic_relevant': result['criteria']['topic_relevant'],
                    'style_match': result['criteria']['style_match'],
                    'topic_depth': result['criteria']['topic_depth'],
                    'topic_style_coherence': result['criteria']['topic_style_coherence'],
                }
        except Exception as e:
            progress['done'] += 1
            print(f"  [{progress['done']}/{progress['total']}] Error at row {row_idx+1}, {col_name}: {e}")
            return {
                'row_idx': row_idx,
                'col_idx': col_idx,
                'text': text,
                'final_score': 'ERROR',
                'topic_relevant': '',
                'style_match': '',
                'topic_depth': '',
                'topic_style_coherence': '',
            }


def wait_for_server(max_retries=30, delay=2):
    """Wait for server to be ready."""
    import requests
    print("Waiting for server...")
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:1337/docs", timeout=5)
            if response.status_code == 200:
                print("Server ready!")
                return True
        except:
            pass
        print(f"  Attempt {i+1}/{max_retries}...")
        time.sleep(delay)
    return False


async def run_scoring_async(
    column_names: list[str], 
    rows: list[list[str]], 
    topic: str, 
    style: str, 
    output_prefix: str,
    concurrency: int
):
    """
    Score all texts asynchronously and save two output files.
    """
    num_cols = len(column_names)
    num_rows = len(rows)
    
    # Build list of tasks
    tasks_info = []
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            if text.strip():
                tasks_info.append((row_idx, col_idx, text, column_names[col_idx]))
    
    total_texts = len(tasks_info)
    print(f"\nScoring {total_texts} texts ({num_rows} rows x {num_cols} columns)")
    print(f"Topic: {topic}, Style: {style}")
    print(f"Concurrency: {concurrency} parallel requests")
    
    # Storage for results
    results = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    
    # Progress tracker
    progress = {'done': 0, 'total': total_texts}
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    
    start_time = time.time()
    
    # Create async session and run all tasks
    async with aiohttp.ClientSession() as session:
        tasks = [
            score_text_async(
                session, semaphore, style, topic, text,
                row_idx, col_idx, col_name, progress
            )
            for row_idx, col_idx, text, col_name in tasks_info
        ]
        
        # Run all tasks concurrently
        completed = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {total_texts} requests in {elapsed:.1f}s ({total_texts/elapsed:.1f} req/s)")
    
    # Populate results matrix
    for r in completed:
        results[r['row_idx']][r['col_idx']] = r
    
    # === Output 1: Detailed (all 4 criteria per column) ===
    detailed_header = []
    for col in column_names:
        detailed_header.extend([
            col,
            f"{col}_score",
            f"{col}_topic_rel",
            f"{col}_style",
            f"{col}_depth",
            f"{col}_coherence"
        ])
    
    detailed_rows = [detailed_header]
    for row_idx in range(num_rows):
        out_row = []
        for col_idx in range(num_cols):
            r = results[row_idx][col_idx]
            if r is None:
                out_row.extend(["", "", "", "", "", ""])
            else:
                fs = r['final_score']
                out_row.extend([
                    r['text'],
                    f"{fs:.3f}" if isinstance(fs, float) else fs,
                    "1" if r['topic_relevant'] else "0" if r['topic_relevant'] is False else "",
                    f"{r['style_match']:.2f}" if isinstance(r['style_match'], float) else "",
                    f"{r['topic_depth']:.2f}" if isinstance(r['topic_depth'], float) else "",
                    f"{r['topic_style_coherence']:.2f}" if isinstance(r['topic_style_coherence'], float) else "",
                ])
        detailed_rows.append(out_row)
    
    detailed_path = f"{output_prefix}_detailed.csv"
    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(detailed_rows)
    print(f"\nSaved detailed results: {detailed_path}")
    
    # === Output 2: Scored (only final score per column) ===
    simple_header = []
    for col in column_names:
        simple_header.append(f"{col}_score")
    
    simple_rows = [simple_header]
    for row_idx in range(num_rows):
        out_row = []
        for col_idx in range(num_cols):
            r = results[row_idx][col_idx]
            if r is None:
                out_row.append("")
            else:
                fs = r['final_score']
                out_row.append(f"{fs:.3f}" if isinstance(fs, float) else fs)
        simple_rows.append(out_row)
    
    simple_path = f"{output_prefix}.csv"
    with open(simple_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(simple_rows)
    print(f"Saved results: {simple_path}")
    
    # === Statistics ===
    print_statistics(results, column_names)


def print_statistics(results, column_names):
    """Print score statistics."""
    all_scores = []
    col_scores = {col: [] for col in column_names}
    
    for row in results:
        for col_idx, r in enumerate(row):
            if r and isinstance(r.get('final_score'), float):
                score = r['final_score']
                all_scores.append(score)
                col_scores[column_names[col_idx]].append(score)
    
    if not all_scores:
        return
    
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    
    print(f"\nOverall ({len(all_scores)} texts):")
    print(f"  Mean: {sum(all_scores)/len(all_scores):.3f}")
    print(f"  Min:  {min(all_scores):.3f}")
    print(f"  Max:  {max(all_scores):.3f}")
    
    print(f"\nPer column:")
    for col in column_names:
        scores = col_scores[col]
        if scores:
            mean = sum(scores)/len(scores)
            print(f"  {col}: mean={mean:.3f}, min={min(scores):.3f}, max={max(scores):.3f}, n={len(scores)}")
    
    # Distribution
    print(f"\nDistribution:")
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = sum(1 for s in all_scores if lo <= s < hi)
        pct = count / len(all_scores) * 100
        label = f"{lo:.1f}-{hi:.1f}" if hi <= 1 else f"{lo:.1f}-1.0"
        bar = "#" * int(pct / 2)
        print(f"  {label}: {count:3d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(
        description='Score texts from a folder of .txt files (async version)'
    )
    parser.add_argument('folder_path', help='Path to folder with .txt files')
    parser.add_argument('--topic', '-t', required=True, help='Topic name (required)')
    parser.add_argument('--style', '-s', required=True, help='Style name (required)')
    parser.add_argument('--output', '-o', default=None, 
                        help='Output prefix (default: results/{folder_name}/scored)')
    parser.add_argument('--concurrency', '-c', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Number of concurrent requests (default: {DEFAULT_CONCURRENCY})')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a directory")
        return
    
    # Read folder
    column_names, rows = convert_folder_to_csv(args.folder_path)
    if not column_names:
        return
    
    # Output prefix
    folder_name = os.path.basename(os.path.normpath(args.folder_path))
    if args.output:
        output_prefix = args.output
    else:
        output_dir = os.path.join("results", folder_name)
        output_prefix = os.path.join(output_dir, "scored")
    
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    
    # Wait for server
    if not wait_for_server():
        print("Server not available. Start with: python rank_server.py")
        return
    
    # Run scoring asynchronously
    asyncio.run(run_scoring_async(
        column_names, rows, args.topic, args.style, output_prefix, args.concurrency
    ))


if __name__ == "__main__":
    main()
