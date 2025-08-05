import json
import sys
import random

def read_file(file_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Read file and split data into train/test/validation sets
    Default split: 70% train, 20% test, 10% validation
    """
    with open(file_path) as f:
        lines = f.read()
        js = json.loads(lines)
        
        # Collect all data points first
        all_data = []
        idx = 0
        
        for key in js.keys():
            all_data.append({"idx": idx, "target": 0, "func": js[key]["func_after"]})
            idx += 1
            all_data.append({"idx": idx, "target": 1, "func": js[key]["func_before"]})
            idx += 1
        
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
        print("Default ratios: train=0.7, test=0.2, val=0.1")
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
