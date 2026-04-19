import csv
import re
from collections import defaultdict

def convert_tmp_to_csv(input_file, output_file):
    data = defaultdict(dict)
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Topic:'):
                continue
            match = re.match(r"\[(\d+)/(\d+)\] Row (\d+), (\w+): (\d+\.\d+)", line)
            if match:
                row = int(match.group(3))
                aggr = match.group(4)
                value = float(match.group(5))
                data[row][aggr] = value
    
    # Get all unique aggr types
    aggr_types = sorted(set(aggr for d in data.values() for aggr in d))
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = aggr_types
        writer.writerow(header)
        for row in sorted(data.keys()):
            row_data = [data[row].get(aggr, '') for aggr in aggr_types]
            writer.writerow(row_data)

if __name__ == "__main__":
    convert_tmp_to_csv('tmp.txt', 'output.csv')