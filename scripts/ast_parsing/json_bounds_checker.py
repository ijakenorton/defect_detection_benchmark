#!/usr/bin/env python3
import json
import os
import argparse
import tree_sitter
from tree_sitter import Language, Parser
from pathlib import Path

from error_analysis_script import (
    load_predictions,
    load_ground_truth,
    # load_code_samples,
    categorize_predictions,
    setup_treesitter,
)


def get_node_text(node, source_code):
    """Extract text from a node."""
    if isinstance(source_code, bytes):
        return source_code[node.start_byte : node.end_byte].decode("utf8")
    else:
        return source_code[node.start_byte : node.end_byte]


def collect_array_accesses(node, source_code, results=None):
    """Find all array accesses in the code."""
    if results is None:
        results = []

    # Check for subscript expressions
    if node.type == "subscript_expression":
        # Get the array identifier and index expression
        arg_node = node.child_by_field_name("argument")
        index_node = node.child_by_field_name("index")

        # Only collect if we have both parts
        if arg_node and index_node:
            access_info = {
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "line": node.start_point[0] + 1,
                "column": node.start_point[1],
                "code": get_node_text(node, source_code),
            }

            # Extract array name if it's a simple identifier
            if arg_node.type == "identifier":
                access_info["array"] = get_node_text(arg_node, source_code)
                access_info["is_static_array"] = True  # Assume static for simplicity

            # Get the index expression
            access_info["index_expr"] = get_node_text(index_node, source_code)

            results.append(access_info)

    # Recursively check children
    for child in node.children:
        collect_array_accesses(child, source_code, results)

    return results


def add_bounds_checks(source_code, array_accesses):
    """Add bounds checks for static arrays."""
    # Sort accesses by position in reverse order
    sorted_accesses = sorted(
        array_accesses, key=lambda x: x["start_byte"], reverse=True
    )

    # Create a mutable list from the source code
    modified_code = list(source_code)

    seen = set()
    # Process each array access
    for access in sorted_accesses:
        # Skip if we don't have the array name
        if "array" not in access or not access.get("is_static_array", False):
            continue

        array_name = access["array"]
        index_expr = access.get("index_expr", "")
        if (array_name in seen):
            continue
        else:
            seen.add(array_name)

        # Create a simple bounds check using sizeof
        bounds_check = f"CHECKBOUNDS({index_expr}{array_name})"

        # Calculate the indentation
        line_start = source_code.rfind("\n", 0, access["start_byte"]) + 1
        # Handle case where we're at the beginning of the file
        if line_start < 0:
            line_start = 0

        # Calculate indentation based on the position of the array name
        indent = " " * (access["start_byte"] - line_start - len(access["array"]))

        # Insert the bounds check before the array access
        insertion_point = source_code.rfind("\n", 0, access["start_byte"]) + 1
        if insertion_point < 0:
            insertion_point = 0

        bounds_check_with_indent = indent + bounds_check

        # Insert at the correct position
        modified_code[insertion_point:insertion_point] = bounds_check_with_indent

    # Reconstruct the source code
    result = "".join(modified_code)

    return result


def load_code_samples(data_file):
    """Load code samples from a JSONL file."""
    code_samples = {}
    with open(data_file, "r") as f:
        for line in f:
            try:
                js = json.loads(line.strip())
                code_samples[int(js["idx"])] = {
                    "func": js["func"],
                    "project": js.get("project", ""),
                    "commit_id": js.get("commit_id", ""),
                    "target": js.get("target", None),
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in JSON: {e}")
                continue

    print(f"Loaded {len(code_samples)} code samples")
    return code_samples


def process_all_samples(code_samples, parser):
    """Process all code samples and add bounds checks."""
    results = {}

    for idx, sample_data in code_samples.items():
        source_code = sample_data["func"]

        # Parse the source code
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node

        # Find array accesses
        array_accesses = collect_array_accesses(root_node, source_code)

        if array_accesses:
            # print(f"  Found {len(array_accesses)} array accesses")
            # Add bounds checks
            modified_code = add_bounds_checks(source_code, array_accesses)

            # Store the modified code
            results[idx] = {
                "original": source_code,
                "modified": modified_code,
                "array_accesses": len(array_accesses),
                "metadata": {
                    "project": sample_data.get("project", ""),
                    "commit_id": sample_data.get("commit_id", ""),
                    "target": sample_data.get("target", None),
                },
            }

            # # Save to output directory if specified
            # if output_dir:
            #     output_path = os.path.join(output_dir, f"sample_{idx}.c")
            #     with open(output_path, "w") as f:
            #         f.write(modified_code)
        else:
            results[idx] = {
                "original": source_code,
                "modified": source_code,
                "array_accesses": len(array_accesses),
                "metadata": {
                    "project": sample_data.get("project", ""),
                    "commit_id": sample_data.get("commit_id", ""),
                    "target": sample_data.get("target", None),
                },
            }

    return results


def save_results_as_jsonl(results, output_file):
    """Save results back in JSONL format."""
    with open(output_file, "w") as f:
        for idx, data in results.items():
            # Create a JSON object like the original but with modified code
            output_obj = {
                "idx": idx,
                "func": data["modified"],
                "project": data["metadata"]["project"],
                "commit_id": data["metadata"]["commit_id"],
                "target": data["metadata"]["target"],
            }
            f.write(json.dumps(output_obj) + "\n")

    print(f"Saved modified code to {output_file}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add bounds checks to C code samples")
    parser.add_argument(
        "--input", required=True, help="Input JSONL file with code samples"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file for modified code"
    )
    parser.add_argument("--summary", help="File to save summary statistics")
    args = parser.parse_args()

    # Set up Tree-sitter
    ts_parser = setup_treesitter()

    # Load code samples
    code_samples = load_code_samples(args.input)

    # Process all samples
    results = process_all_samples(code_samples, ts_parser)

    # Save results
    save_results_as_jsonl(results, args.output)

    # Save summary statistics if requested
    if args.summary:
        total_samples = len(code_samples)
        samples_with_arrays = len(results)
        total_array_accesses = sum(data["array_accesses"] for data in results.values())

        summary = {
            "total_samples": total_samples,
            "samples_with_array_accesses": samples_with_arrays,
            "percentage_with_array_accesses": (
                round(samples_with_arrays / total_samples * 100, 2)
                if total_samples > 0
                else 0
            ),
            "total_array_accesses": total_array_accesses,
            "average_array_accesses_per_sample": (
                round(total_array_accesses / samples_with_arrays, 2)
                if samples_with_arrays > 0
                else 0
            ),
        }

        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary statistics saved to {args.summary}")

    print(
        f"Done! Processed {len(code_samples)} samples, modified {len(results)} with array accesses"
    )


if __name__ == "__main__":
    main()
