import os
import json
import sys
import random
import re

def extract_functions_from_file(file_path):
    """
    Extract individual functions from a C/C++ file
    Returns list of tuples: (function_name, function_code, target)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []
    
        
    functions = []
    
    # More sophisticated function extraction for Juliet format
    # Look for function definitions with proper braces matching
    lines = content.split('\n')

    if file_path == "./100761-v1.0.0/src/testcases/CWE401_Memory_Leak/s03/CWE401_Memory_Leak__wchar_t_calloc_83_bad.cpp":
        print(lines)
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for function definitions (simplified pattern)
        # Match: return_type function_name(parameters) { ... }
        if (line and 
            not line.startswith('//') and 
            not line.startswith('/*') and 
            not line.startswith('#') and
            '(' in line and 
            ')' in line and
            '{' in line):
            
            # This might be a function definition
            func_start = i
            brace_count = line.count('{') - line.count('}')
            func_lines = [lines[i]]
            
            # If braces are balanced on this line, it's a single-line function (unlikely but possible)
            if brace_count == 0 and '{' in line:
                # Look for function name
                func_name = extract_function_name(line)
                if func_name:
                    target = determine_target_from_function_name(func_name)
                    if target is not None:
                        functions.append((func_name, line, target))
                i += 1
                continue
            
            # Multi-line function - collect until braces are balanced
            i += 1
            while i < len(lines) and brace_count > 0:
                current_line = lines[i]
                func_lines.append(current_line)
                brace_count += current_line.count('{') - current_line.count('}')
                i += 1
            
            # Extract function name and determine target
            full_function = '\n'.join(func_lines)
            func_name = extract_function_name(func_lines[0])
            
            if func_name:
                target = determine_target_from_function_name(func_name)
                if target is not None:
                    functions.append((func_name, full_function, target))
        else:
            i += 1
    

    if file_path == "./100761-v1.0.0/src/testcases/CWE401_Memory_Leak/s03/CWE401_Memory_Leak__wchar_t_calloc_83_bad.cpp":
        print(functions)
    
    return functions

def extract_function_name(line):
    """
    Extract function name from a function definition line
    """
    # Remove common prefixes and find the function name
    # Pattern: [static] [return_type] function_name(
    line = line.strip()
    if '(' not in line:
        return None
    
    # Split by '(' and take the left part
    before_paren = line.split('(')[0]
    
    # Split by whitespace and take the last word as function name
    parts = before_paren.strip().split()
    if parts:
        func_name = parts[-1]
        # Remove any pointer indicators
        func_name = func_name.strip('*')
        return func_name
    
    return None

def determine_target_from_function_name(func_name):
    """
    Determine target based on Juliet function naming convention
    """
    func_lower = func_name.lower()
    
    if '_bad' in func_lower or func_lower.endswith('bad'):
        return 1  # vulnerable
    elif '_good' in func_lower or func_lower.endswith('good') or 'goodg2b' in func_lower or 'goodb2g' in func_lower:
        return 0  # safe
    else:
        return None  # skip unclear functions

def convert_juliet_directory(root_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Convert Juliet directory structure to unified JSONL format
    Each file contains both good and bad functions - extract them separately
    """
    data = []
    idx = 0
    
    # File extensions to process
    code_extensions = {'.c', '.cpp', '.cc', '.cxx'}
    
    print(f"Scanning directory: {root_dir}", file=sys.stderr)
    
    processed_files = 0
    skipped_files = 0
    
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(root_dir):
#        print(root)
#        print(dirs)
#        print(files)
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file == "CWE401_Memory_Leak__wchar_t_calloc_83_bad.cpp":
                print(file)
                print(file_path)
            
            # Only process C/C++ files, skip header files and support files
            if file_ext not in code_extensions:
                continue
                
            # Skip support files
            if 'std_testcase' in file or 'io.c' in file:
                continue
            
            processed_files += 1
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} files...", file=sys.stderr)
            
            # Extract functions from file
            functions = extract_functions_from_file(file_path)
            
            if not functions:
                skipped_files += 1
                continue
            
            for func_name, func_code, target in functions:
                if func_code.strip():  # Only add non-empty functions
                    data_point = {
                        "idx": idx,
                        "target": target,
                        "func": func_code.strip()
                    }
                    data.append(data_point)
                    idx += 1
    
    print(f"Processed {processed_files} files, skipped {skipped_files} files", file=sys.stderr)
    print(f"Extracted {len(data)} functions from Juliet dataset", file=sys.stderr)
    
    # Count target distribution
    target_0_count = sum(1 for item in data if item['target'] == 0)
    target_1_count = sum(1 for item in data if item['target'] == 1)
    print(f"Target 0 (safe/good): {target_0_count}", file=sys.stderr)
    print(f"Target 1 (vulnerable/bad): {target_1_count}", file=sys.stderr)
    
    if len(data) == 0:
        print("No data found! Check directory structure and file naming.", file=sys.stderr)
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
    
    # Determine base name from directory
    base_name = os.path.basename(root_dir.rstrip('/\\')) or "juliet"
    
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
        print(f"{split_name} - Target 0 (safe): {target_0}, Target 1 (vulnerable): {target_1}")
    
    print("\nTarget distribution:")
    print_target_dist(train_data, "Train")
    print_target_dist(test_data, "Test")
    print_target_dist(val_data, "Validation")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <juliet_root_directory> [train_ratio] [test_ratio] [val_ratio]")
        print("Default ratios: train=0.7, test=0.2, val=0.1")
        print("\nThe script will:")
        print("- Recursively scan for .c/.cpp/.h/.hpp files") 
        print("- Files with 'bad' in path/name -> target=1 (vulnerable)")
        print("- Files with 'good' in path/name -> target=0 (safe)")
        print("- Extract functions and convert to JSONL format")
        print("\nExamples:")
        print("  python script.py /path/to/juliet-test-suite")
        print("  python script.py ./C_testcases 0.8 0.15 0.05")
        exit(1)
    
    root_directory = sys.argv[1]
    
    if not os.path.isdir(root_directory):
        print(f"Error: {root_directory} is not a valid directory")
        exit(1)
    
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
    
    convert_juliet_directory(root_directory, train_ratio, test_ratio, val_ratio)
