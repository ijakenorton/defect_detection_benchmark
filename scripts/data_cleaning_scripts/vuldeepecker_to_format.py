import json
import sys
import random

def parse_cgd_file(file_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Parse VulDeePecker CGD format and convert to unified JSONL format with train/test/val split
    
    CGD format structure:
    - Header line: number filepath functype linenumber
    - Multiple lines of code
    - Label line: 0 or 1
    - Separator: ---------------------------------
    """
    data = []
    idx = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and separator lines
        if not line or line.startswith('-'):
            i += 1
            continue
            
        # Check if this looks like a header line (starts with number)
        if line and line.split()[0].isdigit():
            # This is a header line
            header_parts = line.split()
            if len(header_parts) >= 4:
                entry_num = header_parts[0]
                file_path_info = header_parts[1]
                func_type = header_parts[2]
                line_num = header_parts[3]
                
                # Collect code lines until we hit a label line
                code_lines = []
                i += 1
                
                while i < len(lines):
                    current_line = lines[i].strip()
                    
                    # Check if this is a label line (single digit 0 or 1)
                    if current_line in ['0', '1']:
                        target = int(current_line)
                        break
                    
                    # Check if we hit a separator or next entry
                    if current_line.startswith('-') or (current_line and current_line.split()[0].isdigit()):
                        # No label found, skip this entry
                        target = None
                        break
                        
                    # This is a code line
                    if current_line:  # Only add non-empty lines
                        code_lines.append(current_line)
                    
                    i += 1
                
                # Only add entry if we found a valid label
                if target is not None and code_lines:
                    func_code = '\n'.join(code_lines)
                    
                    data_point = {
                        "idx": idx,
                        "target": target,
                        "func": func_code
                    }
                    data.append(data_point)
                    idx += 1
                
        i += 1
    
    print(f"Parsed {len(data)} code entries from CGD file", file=sys.stderr)
    
    # Count target distribution
    target_0_count = sum(1 for item in data if item['target'] == 0)
    target_1_count = sum(1 for item in data if item['target'] == 1)
    print(f"Target 0 (non-vulnerable): {target_0_count}", file=sys.stderr)
    print(f"Target 1 (vulnerable): {target_1_count}", file=sys.stderr)
    
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
    base_name = file_path.rsplit('.', 1)[0]
    
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
        print(f"{split_name} - Target 0 (non-vulnerable): {target_0}, Target 1 (vulnerable): {target_1}")
    
    print("\nTarget distribution:")
    print_target_dist(train_data, "Train")
    print_target_dist(test_data, "Test")
    print_target_dist(val_data, "Validation")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_cgd_file> [train_ratio] [test_ratio] [val_ratio]")
        print("Default ratios: train=0.7, test=0.2, val=0.1")
        print("\nExample:")
        print("  python script.py vuldeepecker_data.cgd")
        print("  python script.py vuldeepecker_data.cgd 0.8 0.15 0.05")
        exit(1)
    
    cgd_file = sys.argv[1]
    
    # Parse ratios if provided
    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
        val_ratio = float(sys.argv[4])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
    
    parse_cgd_file(cgd_file, train_ratio, test_ratio, val_ratio)
