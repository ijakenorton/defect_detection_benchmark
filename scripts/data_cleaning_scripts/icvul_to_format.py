import pandas as pd
import json
import sys
import random

def read_file(file_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Convert CSV with before_change column to unified JSONL format
    Maps: before_change True -> label 1, before_change False -> label 0
    The 'code' column becomes 'func'
    """
    all_data = []
    # Read CSV file, handling the ^M line endings
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        df = pd.read_csv(file_path, encoding='latin-1')
    
    # Clean column names (remove any whitespace/special chars)
    df.columns = df.columns.str.strip()
    
    # Handle the ^M characters in the before_change column
    if 'before_change' in df.columns:
        df['before_change'] = df['before_change'].astype(str).str.replace('\r', '').str.strip()
    
    # Convert and output directly
    for idx, row in df.iterrows():
        # Convert before_change to label: True -> 1 (buggy/before), False -> 0 (fixed/after)
        if str(row['before_change']).lower() == 'true':
            target = 1
        elif str(row['before_change']).lower() == 'false':
            target = 0
        else:
            # Skip rows where before_change is not clear
            print(f"Warning: Skipping row {idx} with unclear before_change value: {row['before_change']}", file=sys.stderr)
            continue
            
        data_point = {
            "idx": idx,
            "target": target,
            "func": row['code']  
        }
        all_data.append(data_point)
    random.shuffle(all_data)
    
    # Calculate split indices
    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    
    # Split the data
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    val_data = all_data[train_size + test_size:]

    # Write to separate files
    base_name = file_path.rsplit('.', 1)[0]
    
    with open(f"{base_name}_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Data split complete:")
    print(f"Train: {len(train_data)} samples -> {base_name}_train.jsonl")
    print(f"Test: {len(test_data)} samples -> {base_name}_test.jsonl")
    print(f"Validation: {len(val_data)} samples -> {base_name}_val.jsonl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [train_ratio] [test_ratio] [val_ratio]")
        print("Default ratios: train=0.8, test=0.1, val=0.1")
        exit(1)
    
    file_name = sys.argv[1]
    
    # Optional custom ratios
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
        val_ratio = float(sys.argv[4])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
        
        read_file(file_name, train_ratio, test_ratio, val_ratio)
    else:
        read_file(file_name)
