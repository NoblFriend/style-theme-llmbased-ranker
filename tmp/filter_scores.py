import csv

def filter_score_columns(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        # Find indices of columns ending with '_score'
        score_indices = [i for i, col in enumerate(header) if col.endswith('_score')]
        
        # Filter header
        filtered_header = [header[i] for i in score_indices]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(filtered_header)
            
            for row in reader:
                filtered_row = [row[i] for i in score_indices]
                writer.writerow(filtered_row)

if __name__ == "__main__":
    filter_score_columns('evaluation_data/scored_27_01_simple.csv', 'evaluation_data/scored_27_01_simple_scores_only.csv')