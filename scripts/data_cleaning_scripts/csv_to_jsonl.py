import pandas as pd
import json
import sys
import argparse

def csv_to_jsonl(input_file, output_file, code_column, label_column, label_mapping=None):
    """
    Convert CSV to JSONL format compatible with vulnerability detection models.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSONL file
        code_column: Name of the column containing code
        label_column: Name of the column containing labels
        label_mapping: Optional dict to map label values to 0/1 (e.g., {'safe': 0, 'vulnerable': 1})
    """
    # Read CSV file
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='latin-1')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Validate columns exist
    if code_column not in df.columns:
        raise ValueError(f"Code column '{code_column}' not found. Available columns: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(df.columns)}")

    # Convert to JSONL
    all_data = []
    skipped = 0

    for idx, row in df.iterrows():
        # Get label value
        label_value = row[label_column]

        # Map label to 0/1
        if label_mapping:
            # Use provided mapping
            label_str = str(label_value).strip().lower()
            if label_str in label_mapping:
                target = label_mapping[label_str]
            else:
                print(f"Warning: Skipping row {idx} with unknown label value: {label_value}", file=sys.stderr)
                skipped += 1
                continue
        else:
            # Try to convert directly to int (assumes 0/1 already)
            try:
                target = int(label_value)
                if target not in [0, 1]:
                    print(f"Warning: Skipping row {idx} with invalid target value: {target} (must be 0 or 1)", file=sys.stderr)
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                # Try boolean conversion
                label_str = str(label_value).strip().lower()
                if label_str in ['true', '1', 'yes', 'vulnerable', 'buggy']:
                    target = 1
                elif label_str in ['false', '0', 'no', 'safe', 'clean']:
                    target = 0
                else:
                    print(f"Warning: Skipping row {idx} with unclear label value: {label_value}", file=sys.stderr)
                    skipped += 1
                    continue

        data_point = {
            "idx": idx,
            "target": target,
            "func": str(row[code_column])
        }
        all_data.append(data_point)

    # Write to JSONL file
    with open(output_file, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

    print(f"Conversion complete:")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Converted: {len(all_data)}")
    print(f"  Skipped: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert CSV to JSONL format for vulnerability detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Simple conversion (assumes columns are named 'code' and 'target')
  python csv_to_jsonl.py input.csv output.jsonl

  # Specify column names
  python csv_to_jsonl.py input.csv output.jsonl --code-column Function --label-column Vulnerable

  # With custom label mapping
  python csv_to_jsonl.py input.csv output.jsonl --code-column code --label-column status --map "safe:0,vulnerable:1"

  # Boolean labels (before_change example)
  python csv_to_jsonl.py input.csv output.jsonl --code-column code --label-column before_change --map "true:1,false:0"
        '''
    )

    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output JSONL file')
    parser.add_argument('--code-column', default='code', help='Name of column containing code (default: code)')
    parser.add_argument('--label-column', default='target', help='Name of column containing labels (default: target)')
    parser.add_argument('--map', help='Label mapping in format "value1:0,value2:1" (e.g., "safe:0,vulnerable:1")')

    args = parser.parse_args()

    # Parse label mapping if provided
    label_mapping = None
    if args.map:
        label_mapping = {}
        for pair in args.map.split(','):
            key, value = pair.split(':')
            label_mapping[key.strip().lower()] = int(value.strip())

    csv_to_jsonl(args.input, args.output, args.code_column, args.label_column, label_mapping)
