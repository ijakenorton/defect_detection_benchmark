import os
import json
import sys
import random
import re

def extract_functions_from_file(file_path):
    """
    Extract individual functions from a C/C++ file
    Returns list of tuples: (function_name, function_code, target)
    Handles both C functions and C++ class methods/constructors
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []
    
    functions = []
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines, comments, and preprocessor directives
        if (not line or 
            line.startswith('//') or 
            line.startswith('/*') or 
            line.startswith('#') or
            line.startswith('*') or
            line == '}' or
            line.startswith('typedef') or
            line.startswith('struct')):
            i += 1
            continue
        
        # Look for function patterns:
        # 1. Function signature might span multiple lines
        # 2. Look for patterns like: void functionName() or int functionName(params)
        # 3. The opening brace { might be on the next line
        
        potential_func_lines = []
        func_start_idx = i
        
        # Collect lines that might be part of a function signature
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Skip empty lines and comments in function signature
            if (not current_line or 
                current_line.startswith('//') or 
                current_line.startswith('/*') or
                current_line.startswith('*')):
                i += 1
                continue
            
            potential_func_lines.append(current_line)
            
            # If we hit an opening brace, this might be a function
            if '{' in current_line:
                break
            
            # If we hit certain keywords, this is not a function
            if (current_line.startswith('#') or
                current_line.startswith('typedef') or
                current_line.startswith('struct') or
                current_line.endswith(';')):  # Declaration, not definition
                break
                
            i += 1
        
        # Check if we found a potential function
        if potential_func_lines and any('{' in line for line in potential_func_lines):
            # Reconstruct the function signature
            func_signature = ' '.join(potential_func_lines)
            
            # Check if this looks like a function definition
            if (('(' in func_signature and ')' in func_signature and '{' in func_signature) or
                ('::' in func_signature and '{' in func_signature)):
                
                # Extract function name
                func_name = extract_function_name_improved(func_signature)

                print(func_name)
                
                if func_name:
                    target = determine_target_from_function_name(func_name)
                    
                    if target is not None:
                        # Now collect the complete function body
                        brace_count = 0
                        func_lines = []
                        
                        # Start from the beginning of the function
                        j = func_start_idx
                        while j <= i:
                            func_lines.append(lines[j])
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            j += 1
                        
                        # Continue collecting until braces are balanced
                        while j < len(lines) and brace_count > 0:
                            func_lines.append(lines[j])
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            j += 1
                        
                        # Update i to continue after this function
                        i = j
                        
                        full_function = '\n'.join(func_lines)
                        print(full_function)
                        exit(1)
                        functions.append((func_name, full_function, target))
                        continue
        
        i += 1

    
    return functions

def extract_function_name_improved(func_signature):
    """
    Extract function name from a function signature (might be multi-line)
    Handles C functions, C++ methods, constructors, and destructors
    """
    func_signature = func_signature.strip()
    
    # Handle C++ scope resolution (ClassName::method)
    if '::' in func_signature and '(' in func_signature:
        parts = func_signature.split('::')
        if len(parts) >= 2:
            after_scope = parts[-1]
            if '(' in after_scope:
                method_part = after_scope.split('(')[0].strip()
                method_name = method_part.strip('~')
                class_name = parts[-2].split()[-1] if len(parts) >= 2 else ''
                if method_name and class_name:
                    return f"{class_name}::{method_name}"
    
    # Handle regular C functions
    if '(' in func_signature:
        # Split on '(' and take everything before it
        before_paren = func_signature.split('(')[0]
        
        # Look for function name - it's typically the last word before (
        # Remove common return types and modifiers
        words = before_paren.strip().split()
        
        # Filter out common keywords that aren't function names
        keywords_to_skip = {'void', 'int', 'char', 'float', 'double', 'long', 'short', 
                           'unsigned', 'signed', 'const', 'static', 'inline', 'extern'}
        
        # Find the last word that's not a keyword
        for word in reversed(words):
            clean_word = word.strip('*&')  # Remove pointer/reference indicators
            if clean_word and clean_word.lower() not in keywords_to_skip:
                return clean_word
    
    return None

def determine_target_from_function_name(func_name):
    """
    Determine target based on Juliet function naming convention
    """
    func_lower = func_name.lower()
    
    if ('_bad' in func_lower or 
        func_lower.endswith('bad') or
        '::bad' in func_lower or
        '_bad::' in func_lower):
        return 1  # vulnerable
    elif ('_good' in func_lower or 
          func_lower.endswith('good') or 
          'goodg2b' in func_lower or 
          'goodb2g' in func_lower or
          '::good' in func_lower or
          '_good::' in func_lower):
        return 0  # safe
    else:
        return None  # skip unclear functions

def convert_juliet_directory(root_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
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
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
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
    base_name = "juliet"
    
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
    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    if len(sys.argv) >= 5:
        train_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
        val_ratio = float(sys.argv[4])
        
        # Validate ratios sum to 1
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            print("Error: Ratios must sum to 1.0")
            exit(1)
    
    convert_juliet_directory(root_directory, train_ratio, test_ratio, val_ratio)
