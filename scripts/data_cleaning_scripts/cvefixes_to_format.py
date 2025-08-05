import json
import sys
import random

def convert_method_change_to_jsonl(input_file, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Convert method change JSON format to unified JSONL format with train/test/val split
    Maps: before_change "True" -> target 1 (vulnerable), "False" -> target 0 (fixed)
    """
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"Loading {len(json_data)} method changes", file=sys.stderr)
    
    data = []
    
    for idx, item in enumerate(json_data):
        # Extract the before_change field and convert to target
        before_change = item.get('before_change', '')
        
        if before_change == "True":
            target = 1  # vulnerable/before change
        elif before_change == "False":
            target = 0  # fixed/after change
        else:
            print(f"Warning: Skipping item {idx} with unclear before_change value: {before_change}", file=sys.stderr)
            continue
        
        # Clean up the code - remove extra whitespace artifacts
        code = item.get('code', '')
        if not code.strip():
            print(f"Warning: Skipping item {idx} with empty code", file=sys.stderr)
            continue
        
        # Clean up formatting artifacts in the code
        cleaned_code = clean_code_formatting(code)
        
        data_point = {
            "idx": idx,
            "target": target,
            "func": cleaned_code
        }
        data.append(data_point)
    
    print(f"Successfully processed {len(data)} method changes", file=sys.stderr)
    
    # Count target distribution
    target_0_count = sum(1 for item in data if item['target'] == 0)
    target_1_count = sum(1 for item in data if item['target'] == 1)
    print(f"Target 0 (after/fixed): {target_0_count}", file=sys.stderr)
    print(f"Target 1 (before/vulnerable): {target_1_count}", file=sys.stderr)
    
    if len(data) == 0:
        print("No valid data found!", file=sys.stderr)
        return
    
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
    
    # Determine base name from input file
    base_name = input_file.rsplit('.', 1)[0]
    
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
    
    print(f"\nData split and saved:")
    print(f"Train: {len(train_data)} samples -> {base_name}_train.jsonl")
    print(f"Test: {len(test_data)} samples -> {base_name}_test.jsonl")
    print(f"Validation: {len(val_data)} samples -> {base_name}_val.jsonl")
    
    # Print target distribution for each split
    def print_target_dist(data, split_name):
        targets = [item['target'] for item in data]
        target_0 = targets.count(0)
        target_1 = targets.count(1)
        print(f"{split_name} - Target 0 (fixed): {target_0}, Target 1 (vulnerable): {target_1}")
    
    print("\nTarget distribution:")
    print_target_dist(train_data, "Train")
    print_target_dist(test_data, "Test")
    print_target_dist(val_data, "Validation")

def clean_code_formatting(code):
    """
    Clean up formatting artifacts in the code
    """
    # Remove excessive whitespace while preserving structure
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove trailing whitespace but preserve leading indentation
        cleaned_line = line.rstrip()
        
        # Fix spacing artifacts (multiple spaces that look odd)
        # But be careful not to break string literals or important formatting
        import re
        
        # Replace multiple consecutive spaces (except at line start) with single space
        # But preserve spaces in string literals
        if '"' not in cleaned_line and "'" not in cleaned_line:
            # Only clean spacing if no string literals present
            cleaned_line = re.sub(r'(?<=\S)  +', ' ', cleaned_line)
        
        cleaned_lines.append(cleaned_line)
    
    # Join lines and remove any completely empty lines at start/end
    result = '\n'.join(cleaned_lines).strip()
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_json_file> [train_ratio] [test_ratio] [val_ratio]")
        print("Default ratios: train=0.7, test=0.2, val=0.1")
        print("\nInput format: JSON array with objects containing:")
        print("  - 'code': function source code")
        print("  - 'before_change': 'True' (vulnerable) or 'False' (fixed)")
        print("\nExamples:")
        print("  python script.py method_changes.json")
        print("  python script.py method_changes.json 0.8 0.15 0.05")
        exit(1)
    
    input_file = sys.argv[1]
    
    # Parse ratios if provided
    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
        val_ratio = float(sys.argv[4])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
    
    convert_method_change_to_jsonl(input_file, train_ratio, test_ratio, val_ratio)
