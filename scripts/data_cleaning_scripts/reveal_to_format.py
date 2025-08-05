import json
import sys
import os
import random

def convert_reveal_jsons_to_jsonl(vulnerables_file, non_vulnerables_file, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Convert Reveal dataset JSON files to unified JSONL format with train/test/val split
    vulnerables.json -> target: 1
    non-vulnerables.json -> target: 0
    """
    data = []
    idx = 0
    
    # Process vulnerables (target = 1)
    if os.path.exists(vulnerables_file):
        with open(vulnerables_file, 'r') as f:
            vulnerables = json.load(f)
            
        print(f"Loading {len(vulnerables)} vulnerable functions", file=sys.stderr)
        
        for item in vulnerables:
            data_point = {
                "idx": idx,
                "target": 1,  # vulnerable
                "func": item["code"]
            }
            data.append(data_point)
            idx += 1
    else:
        print(f"Warning: {vulnerables_file} not found", file=sys.stderr)
    
    # Process non-vulnerables (target = 0)
    if os.path.exists(non_vulnerables_file):
        with open(non_vulnerables_file, 'r') as f:
            non_vulnerables = json.load(f)
            
        print(f"Loading {len(non_vulnerables)} non-vulnerable functions", file=sys.stderr)
        
        for item in non_vulnerables:
            data_point = {
                "idx": idx,
                "target": 0,  # non-vulnerable
                "func": item["code"]
            }
            data.append(data_point)
            idx += 1
    else:
        print(f"Warning: {non_vulnerables_file} not found", file=sys.stderr)
    
    print(f"Total loaded: {len(data)} functions", file=sys.stderr)
    
    # Shuffle the data for random split
    random.shuffle(data)
    
    # Calculate split indices
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]
    
    # Determine base name from the first file
    base_name = vulnerables_file.rsplit('.', 1)[0].replace('vulnerables', 'reveal')
    
    # Write to separate files
    with open(f"{base_name}_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Data split and saved:")
    print(f"Train: {len(train_data)} samples -> {base_name}_train.jsonl")
    print(f"Test: {len(test_data)} samples -> {base_name}_test.jsonl")
    print(f"Validation: {len(val_data)} samples -> {base_name}_val.jsonl")
    
    # Print target distribution for each split
    def print_target_dist(data, split_name):
        targets = [item['target'] for item in data]
        target_0 = targets.count(0)
        target_1 = targets.count(1)
        print(f"{split_name} - Target 0 (non-vulnerable): {target_0}, Target 1 (vulnerable): {target_1}")
    
    print("\nTarget distribution:")
    print_target_dist(train_data, "Train")
    print_target_dist(test_data, "Test")
    print_target_dist(val_data, "Validation")

def convert_single_reveal_json(json_file, target_value, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Convert a single Reveal JSON file to unified JSONL format with train/test/val split
    """
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    print(f"Loading {len(json_data)} functions with target={target_value}", file=sys.stderr)
    
    data = []
    for idx, item in enumerate(json_data):
        data_point = {
            "idx": idx,
            "target": target_value,
            "func": item["code"]
        }
        data.append(data_point)
    
    # Shuffle the data for random split
    random.shuffle(data)
    
    # Calculate split indices
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]
    
    # Determine base name
    base_name = json_file.rsplit('.', 1)[0]
    
    # Write to separate files
    with open(f"{base_name}_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{base_name}_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Data split and saved:")
    print(f"Train: {len(train_data)} samples -> {base_name}_train.jsonl")
    print(f"Test: {len(test_data)} samples -> {base_name}_test.jsonl")
    print(f"Validation: {len(val_data)} samples -> {base_name}_val.jsonl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py <vulnerables.json> <non-vulnerables.json> [train_ratio] [test_ratio] [val_ratio]")
        print("  python script.py <single_file.json> <target_value> [train_ratio] [test_ratio] [val_ratio]")
        print("\nDefault ratios: train=0.7, test=0.2, val=0.1")
        print("\nExamples:")
        print("  python script.py vulnerables.json non-vulnerables.json")
        print("  python script.py vulnerables.json non-vulnerables.json 0.8 0.15 0.05")
        print("  python script.py vulnerables.json 1")
        print("  python script.py non-vulnerables.json 0")
        exit(1)
    
    # Parse ratios if provided
    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[-3])
        test_ratio = float(sys.argv[-2]) 
        val_ratio = float(sys.argv[-1])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
    
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        # Single file mode with target value
        json_file = sys.argv[1]
        target_value = int(sys.argv[2])
        convert_single_reveal_json(json_file, target_value, train_ratio, test_ratio, val_ratio)
    elif len(sys.argv) >= 3:
        # Two file mode
        vulnerables_file = sys.argv[1]
        non_vulnerables_file = sys.argv[2]
        convert_reveal_jsons_to_jsonl(vulnerables_file, non_vulnerables_file, train_ratio, test_ratio, val_ratio)
    else:
        print("Error: Invalid arguments")
        exit(1)
