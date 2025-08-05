import pandas as pd
import json
import sys
import random

def convert_csv_to_jsonl(csv_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Convert CSV to JSONL format and split into train/test/validation sets
    The 'bug' column (0=no bug, 1=bug) becomes the label
    The 'code' column becomes the func
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert to the required format
    all_data = []
    for idx, row in df.iterrows():
        data_point = {
            "idx": idx,
            "label": int(row['bug']),  # bug column becomes label
            "func": row['code']        # code column becomes func
        }
        all_data.append(data_point)
    
    # Shuffle the data for random split
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
    base_name = csv_path.rsplit('.', 1)[0]
    
    with open(f"{base_name}_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"CSV conversion and split complete:")
    print(f"Total samples: {len(all_data)}")
    print(f"Train: {len(train_data)} samples -> {base_name}_train.jsonl")
    print(f"Test: {len(test_data)} samples -> {base_name}_test.jsonl")
    print(f"Validation: {len(val_data)} samples -> {base_name}_val.jsonl")
    
    # Print label distribution for each split
    def print_label_dist(data, split_name):
        labels = [item['label'] for item in data]
        label_0 = labels.count(0)
        label_1 = labels.count(1)
        print(f"{split_name} - Label 0 (no bug): {label_0}, Label 1 (bug): {label_1}")
    
    print("\nLabel distribution:")
    print_label_dist(train_data, "Train")
    print_label_dist(test_data, "Test")
    print_label_dist(val_data, "Validation")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv> [train_ratio] [test_ratio] [val_ratio]")
        print("Default ratios: train=0.7, test=0.2, val=0.1")
        print("The 'bug' column will be used as label, 'code' column as func")
        exit(1)
    
    csv_file = sys.argv[1]
    
    # Optional custom ratios
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
        val_ratio = float(sys.argv[4])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
        
        convert_csv_to_jsonl(csv_file, train_ratio, test_ratio, val_ratio)
    else:
        convert_csv_to_jsonl(csv_file)
