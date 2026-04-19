import os
import csv
import requests
import glob
import re
import time

SERVER_URL = "http://localhost:1337/rank"
DATA_DIR = "evaluation_data"

def get_metadata_from_filename(filename):
    # Expecting Topic_Style.csv or style_StyleName.csv
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

def evaluate_file(filepath):
    print(f"Processing {filepath}...")
    topic, style = get_metadata_from_filename(filepath)
    print(f"Detected topic: '{topic}', style: '{style}'")
    
    output_rows = []
    header = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Empty file")
            return

        output_rows.append(header)
        
        for row_idx, row in enumerate(reader):
            if not row: continue
            
            texts = row
            # Call server
            try:
                payload = {
                    "style_name": style,
                    "texts": texts
                }
                if topic:
                    payload["topic"] = topic

                response = requests.post(SERVER_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                ranked_indices = result['ranked_indices']
                
                # Convert indices to ranks
                # ranked_indices is list of indices ordered by rank (best first)
                # e.g. [1, 0, 2] -> Text at index 1 is #1, Text at 0 is #2, Text at 2 is #3
                
                ranks = [0] * len(texts)
                for rank, original_idx in enumerate(ranked_indices):
                    ranks[original_idx] = rank + 1
                
                output_rows.append(ranks)
                print(f"Row {row_idx+1} processed.")
                
            except Exception as e:
                print(f"Error processing row {row_idx+1}: {e}")
                output_rows.append(["ERROR"] * len(texts))

    output_filename = os.path.join(DATA_DIR, f"ranked_{os.path.basename(filepath)}")
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    
    print(f"Saved results to {output_filename}")

def main():
    # Check if server is up
    try:
        requests.get("http://localhost:1337/docs", timeout=1)
    except requests.exceptions.ConnectionError:
        print("Warning: Server does not seem to be running at http://localhost:1337")
        print("Please start the server with: python rank_server.py")
        # We continue anyway, maybe it's starting up or user will start it
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    # Filter out ranked_ files
    files = [f for f in files if not os.path.basename(f).startswith("ranked_")]

    if not files:
        print(f"No .csv files found in {DATA_DIR}/")
        return
        
    for f in files:
        evaluate_file(f)

if __name__ == "__main__":
    main()
