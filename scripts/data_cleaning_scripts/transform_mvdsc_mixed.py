import pandas as pd
import json
import sys

def convert_csv_to_jsonl(csv_path):
    """
    Convert CSV to JSONL format
    The 'bug' column (0=no bug, 1=bug) becomes the label
    The 'code' column becomes the func
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert and output directly
    for idx, row in df.iterrows():
        data_point = {
            "idx": idx,
            "target": int(row['bug']),  # bug column becomes target
            "func": row['code']        # code column becomes func
        }
        print(json.dumps(data_point))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv>")
        print("The 'bug' column will be used as label, 'code' column as func")
        exit(1)
    
    csv_file = sys.argv[1]
    convert_csv_to_jsonl(csv_file)
