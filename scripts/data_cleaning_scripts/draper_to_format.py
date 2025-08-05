import h5py
import json
import sys
import numpy as np

def inspect_hdf5_structure(file_path):
    """
    Inspect the structure of the HDF5 file to understand the data organization
    """
    print("Inspecting HDF5 file structure...")
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                if obj.size < 10:  # Print small datasets
                    print(f"  Data: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        f.visititems(print_structure)

def convert_hdf5_to_jsonl(file_path, label_strategy='any_cwe'):
    """
    Convert Draper HDF5 data to unified JSONL format
    
    Args:
        file_path: Path to HDF5 file
        label_strategy: How to handle multiple CWE labels
            - 'any_cwe': label=1 if any CWE is True, label=0 if all False
            - 'cwe_119': use only CWE-119 labels
            - 'cwe_120': use only CWE-120 labels
            - 'cwe_476': use only CWE-476 labels
            - 'cwe_469': use only CWE-469 labels
            - 'cwe_other': use only CWE-other labels
    """
    with h5py.File(file_path, 'r') as f:
        # Load the function source code
        functions = f['functionSource'][:]
        
        # Load all CWE labels
        cwe_119 = f['CWE-119'][:]
        cwe_120 = f['CWE-120'][:]
        cwe_469 = f['CWE-469'][:]
        cwe_476 = f['CWE-476'][:]
        cwe_other = f['CWE-other'][:]
        
        print(f"Loaded {len(functions)} functions", file=sys.stderr)
        print(f"CWE-119: {sum(cwe_119)} positives", file=sys.stderr)
        print(f"CWE-120: {sum(cwe_120)} positives", file=sys.stderr)
        print(f"CWE-469: {sum(cwe_469)} positives", file=sys.stderr)
        print(f"CWE-476: {sum(cwe_476)} positives", file=sys.stderr)
        print(f"CWE-other: {sum(cwe_other)} positives", file=sys.stderr)
    
        data = []
        for idx in range(len(functions)):
            # Get function source
            func = functions[idx]
            if isinstance(func, bytes):
                func = func.decode('utf-8')
            elif hasattr(func, 'decode'):  # Handle numpy bytes
                func = func.decode('utf-8')
            
            # Determine label based on strategy
            if label_strategy == 'any_cwe':
                # Label=1 if any CWE is present, 0 if no vulnerabilities
                label = int(cwe_119[idx] or cwe_120[idx] or cwe_469[idx] or cwe_476[idx] or cwe_other[idx])
            elif label_strategy == 'cwe_119':
                label = int(cwe_119[idx])
            elif label_strategy == 'cwe_120':
                label = int(cwe_120[idx])
            elif label_strategy == 'cwe_469':
                label = int(cwe_469[idx])
            elif label_strategy == 'cwe_476':
                label = int(cwe_476[idx])
            elif label_strategy == 'cwe_other':
                label = int(cwe_other[idx])
            else:
                raise ValueError(f"Unknown label strategy: {label_strategy}")
            
            data_point = {
                "idx": idx,
                "target": label,
                "func": func
            }
            data.append(data_point)

        

        base_name = file_path.rsplit('.', 1)[0]
        with open(f"{base_name}.jsonl", 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_hdf5_file> [--inspect] [--strategy STRATEGY]")
        print("\nLabel strategies:")
        print("  any_cwe (default): label=1 if any CWE is True")
        print("  cwe_119: use only CWE-119 labels")
        print("  cwe_120: use only CWE-120 labels") 
        print("  cwe_469: use only CWE-469 labels")
        print("  cwe_476: use only CWE-476 labels")
        print("  cwe_other: use only CWE-other labels")
        print("\nExamples:")
        print("  python script.py data.h5 --inspect")
        print("  python script.py data.h5 --strategy cwe_119")
        exit(1)
    
    file_path = sys.argv[1]
    
    # Parse arguments
    inspect_mode = '--inspect' in sys.argv
    
    # Get strategy
    strategy = 'any_cwe'  # default
    if '--strategy' in sys.argv:
        strategy_idx = sys.argv.index('--strategy')
        if strategy_idx + 1 < len(sys.argv):
            strategy = sys.argv[strategy_idx + 1]
    
    if inspect_mode:
        inspect_hdf5_structure(file_path)
    else:
        convert_hdf5_to_jsonl(file_path, strategy)
