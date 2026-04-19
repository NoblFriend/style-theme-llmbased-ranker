"""
Scoring evaluation script - evaluates texts using pointwise 0-1 scores.
Unlike run_evaluation.py (listwise ranking), this evaluates each text independently.
"""
import os
import csv
import requests
import glob
import time
from typing import Optional

SERVER_URL = "http://localhost:1337/score"
DATA_DIR = "evaluation_data"

def get_metadata_from_filename(filename):
    """
    Extract topic and style from filename.
    Expecting Topic_Style.csv or style_StyleName.csv
    """
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    
    if name.startswith("style_"):
        # Legacy: style_StyleName -> Topic=None, Style=StyleName
        return None, name[6:].replace("_", " ")
    
    # New format: Topic_Style
    parts = name.split('_', 1)
    if len(parts) == 2:
        topic = parts[0]
        style = parts[1].replace("_", " ")
        return topic, style
        
    return None, name

def score_text(style: str, topic: str, text: str) -> dict:
    """
    Score a single text via the /score endpoint.
    Returns dict with final_score and criteria breakdown.
    """
    payload = {
        "style_name": style,
        "topic": topic,
        "text": text
    }
    
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    return response.json()

def evaluate_file(filepath: str, output_format: str = "detailed"):
    """
    Evaluate all texts in a CSV file using pointwise scoring.
    
    Args:
        filepath: Path to input CSV
        output_format: 'detailed' (all criteria) or 'simple' (final score only)
    """
    print(f"Processing {filepath}...")
    topic, style = get_metadata_from_filename(filepath)
    
    if not topic:
        print(f"WARNING: No topic found in filename. Scoring requires a topic. Skipping.")
        return
    
    print(f"Detected topic: '{topic}', style: '{style}'")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Empty file")
            return

        all_rows = list(reader)
    
    num_columns = len(header)
    
    # Output structure depends on format
    if output_format == "detailed":
        # Header: original columns + _score, _topic_rel, _style, _depth, _coherence
        output_header = []
        for col in header:
            output_header.extend([
                col,
                f"{col}_score",
                f"{col}_topic_rel",
                f"{col}_style",
                f"{col}_depth", 
                f"{col}_coherence"
            ])
    else:
        # Simple: just original columns with _score suffix
        output_header = []
        for col in header:
            output_header.extend([col, f"{col}_score"])
    
    output_rows = [output_header]
    
    total_texts = sum(len([t for t in row if t.strip()]) for row in all_rows)
    processed = 0
    
    for row_idx, row in enumerate(all_rows):
        if not row:
            continue
        
        output_row = []
        
        for col_idx, text in enumerate(row):
            if not text.strip():
                # Empty cell
                if output_format == "detailed":
                    output_row.extend(["", "", "", "", "", ""])
                else:
                    output_row.extend(["", ""])
                continue
            
            try:
                result = score_text(style, topic, text)
                final_score = result['final_score']
                criteria = result['criteria']
                
                if output_format == "detailed":
                    output_row.extend([
                        text,
                        f"{final_score:.3f}",
                        "1" if criteria['topic_relevant'] else "0",
                        f"{criteria['style_match']:.2f}",
                        f"{criteria['topic_depth']:.2f}",
                        f"{criteria['topic_style_coherence']:.2f}"
                    ])
                else:
                    output_row.extend([text, f"{final_score:.3f}"])
                
                processed += 1
                print(f"  [{processed}/{total_texts}] Row {row_idx+1}, Col {col_idx+1}: score={final_score:.3f}")
                
            except Exception as e:
                print(f"  Error at row {row_idx+1}, col {col_idx+1}: {e}")
                if output_format == "detailed":
                    output_row.extend([text, "ERROR", "", "", "", ""])
                else:
                    output_row.extend([text, "ERROR"])
        
        output_rows.append(output_row)
    
    # Save results
    output_filename = os.path.join(DATA_DIR, f"scored_{os.path.basename(filepath)}")
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    
    print(f"Saved results to {output_filename}")
    
    # Print summary statistics
    print_summary(output_rows[1:], output_format)

def print_summary(data_rows, output_format):
    """Print summary statistics of scores."""
    scores = []
    
    for row in data_rows:
        if output_format == "detailed":
            # Scores are at indices 1, 7, 13, ... (every 6th starting from 1)
            step = 6
            for i in range(1, len(row), step):
                if row[i] and row[i] != "ERROR":
                    try:
                        scores.append(float(row[i]))
                    except ValueError:
                        pass
        else:
            # Scores are at indices 1, 3, 5, ... (every 2nd starting from 1)
            for i in range(1, len(row), 2):
                if row[i] and row[i] != "ERROR":
                    try:
                        scores.append(float(row[i]))
                    except ValueError:
                        pass
    
    if scores:
        print("\n--- Score Statistics ---")
        print(f"Total texts scored: {len(scores)}")
        print(f"Mean score: {sum(scores)/len(scores):.3f}")
        print(f"Min score: {min(scores):.3f}")
        print(f"Max score: {max(scores):.3f}")
        
        # Distribution
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
        counts = [0] * len(labels)
        
        for s in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= s < bins[i+1]:
                    counts[i] += 1
                    break
        
        print("Distribution:")
        for label, count in zip(labels, counts):
            pct = count / len(scores) * 100
            bar = "#" * int(pct / 2)
            print(f"  {label}: {count:3d} ({pct:5.1f}%) {bar}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Score texts using pointwise 0-1 evaluation")
    parser.add_argument("--file", "-f", help="Specific CSV file to process")
    parser.add_argument("--format", "-o", choices=["detailed", "simple"], default="detailed",
                        help="Output format: 'detailed' includes all criteria, 'simple' only final score")
    args = parser.parse_args()
    
    # Check if server is up
    try:
        requests.get("http://localhost:1337/docs", timeout=2)
    except requests.exceptions.ConnectionError:
        print("Warning: Server does not seem to be running at http://localhost:1337")
        print("Please start the server with: python rank_server.py")
        return
    
    if args.file:
        if os.path.exists(args.file):
            evaluate_file(args.file, args.format)
        else:
            print(f"File not found: {args.file}")
    else:
        # Process all CSV files
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        # Filter out already processed files
        files = [f for f in files if not os.path.basename(f).startswith(("ranked_", "scored_"))]
        
        if not files:
            print(f"No .csv files found in {DATA_DIR}/")
            return
        
        for f in files:
            evaluate_file(f, args.format)

if __name__ == "__main__":
    main()
