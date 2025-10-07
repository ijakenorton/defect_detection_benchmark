#!/usr/bin/env python3
import json
import os
import argparse
import tree_sitter
from tree_sitter import Language, Parser

from error_analysis_script import (
    load_predictions,
    load_ground_truth,
    load_code_samples,
    categorize_predictions,
    setup_treesitter,
)


def get_node_text(node, source_code):
    """
    Extract text from a node regardless of whether source_code is bytes or string.

    Args:
        node: Tree-sitter node
        source_code: Source code as string or bytes

    Returns:
        String content of the node
    """
    if isinstance(source_code, bytes):
        return source_code[node.start_byte : node.end_byte].decode("utf8")
    else:
        return source_code[node.start_byte : node.end_byte]


def debug_node(node, source_code, indent=0):
    """
    Print detailed information about a node for debugging.

    Args:
        node: Tree-sitter node to debug
        source_code: Source code as string or bytes
        indent: Indentation level for pretty printing
    """
    node_text = get_node_text(node, source_code)
    print(" " * indent + f"Node type: {node.type}")
    print(" " * indent + f"Text: {node_text}")
    print(" " * indent + f"Position: {node.start_point} -> {node.end_point}")

    print(" " * indent + "Fields:")
    for field in node.fields:
        field_node = node.child_by_field_name(field)
        if field_node:
            field_text = get_node_text(field_node, source_code)
            print(" " * (indent + 2) + f"{field}: {field_node.type} - {field_text}")

    print(" " * indent + "Children:")
    for i, child in enumerate(node.children):
        child_text = get_node_text(child, source_code)
        print(" " * (indent + 2) + f"Child {i}: {child.type} - {child_text[:20]}...")

    print("")


def traverse_tree(node, source_code, results=None):
    """
    Traverse the tree-sitter syntax tree and collect subscript expressions info.

    Args:
        node: Tree-sitter node to traverse
        source_code: Source code as string or bytes
        results: Dictionary to collect results (created if None)

    Returns:
        Dictionary with collected information about array accesses
    """
    if results is None:
        results = {"array_accesses": []}

    # Check for subscript expressions
    if node.type == "subscript_expression":
        # Get the argument (array/container being accessed)
        arg_node = node.child_by_field_name("argument")
        # Get the index expression
        index_node = node.child_by_field_name("index")

        access_info = {
            "line": node.start_point[0] + 1,  # 1-indexed line number
            "column": node.start_point[1],
            "code": get_node_text(node, source_code),
        }

        # Extract identifier names
        if arg_node and arg_node.type == "identifier":
            access_info["array"] = get_node_text(arg_node, source_code)
            access_info["array_line"] = arg_node.start_point[0] + 1

            # Print information (you can remove this if you only want to collect data)
            print(f"Array: {access_info['array']} at line {access_info['array_line']}")

        # If index is a binary expression, extract the operands
        if index_node:
            access_info["index_type"] = index_node.type

            if index_node.type == "binary_expression":
                left_node = index_node.child_by_field_name("left")
                right_node = index_node.child_by_field_name("right")

                # Try to find the operator
                for child in index_node.children:
                    if child.type != "identifier" and (
                        child != left_node and child != right_node
                    ):
                        access_info["operator"] = get_node_text(child, source_code)
                        break

                if left_node and left_node.type == "identifier":
                    left_name = get_node_text(left_node, source_code)
                    left_line = left_node.start_point[0] + 1

                    access_info["left_operand"] = {
                        "type": left_node.type,
                        "text": left_name,
                        "line": left_line,
                    }

                    # Print information
                    print(f"Left operand: {left_name} at line {left_line}")

                if right_node and right_node.type == "identifier":
                    right_name = get_node_text(right_node, source_code)
                    right_line = right_node.start_point[0] + 1

                    access_info["right_operand"] = {
                        "type": right_node.type,
                        "text": right_name,
                        "line": right_line,
                    }

                    # Print information
                    print(f"Right operand: {right_name} at line {right_line}")
            else:
                # Handle simple indexes (like array[5])
                access_info["simple_index"] = get_node_text(index_node, source_code)

        results["array_accesses"].append(access_info)

    # Recursively check children
    for child in node.children:
        traverse_tree(child, source_code, results)

    return results


def parse(code_samples, parser):
    """
    Analyze code for subscript expressions using tree-sitter.

    Args:
        code_samples: List of code segments to analyze
        parser: Configured Tree-sitter parser

    Returns:
        List of results for each code segment
    """
    all_results = []

    for i, code_segment in enumerate(code_samples):
        if not code_segment.strip():
            continue  # Skip empty segments

        print(f"\n--- Analyzing code segment {i+1} ---")

        # Parse with tree-sitter - convert to bytes for parsing
        code_bytes = bytes(code_segment, "utf8")
        tree = parser.parse(code_bytes)
        root_node = tree.root_node

        # Collect information about array accesses using the string version
        segment_results = traverse_tree(root_node, code_segment)
        segment_results["segment_id"] = i

        all_results.append(segment_results)

    return all_results


def arg_parse():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="C Code Array Access Analyzer")
    parser.add_argument(
        "--data", required=True, help="Path to the C code file to analyze"
    )
    parser.add_argument(
        "--output", help="Path to save detailed analysis results (JSON)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    return args


def main():
    """Main function to run the analysis."""
    args = arg_parse()

    # Set up Tree-sitter
    ts_parser = setup_treesitter()

    # Load and split code
    try:
        with open(args.data, "r") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {args.data}: {e}")
        exit(1)

    # Split by separator (if any) or treat as a single segment
    if "*********************************************" in code:
        code_samples = [
            segment.strip()
            for segment in code.split("*********************************************")
            if segment.strip()
        ]
    else:
        code_samples = [code]

    print(f"Found {len(code_samples)} code segments to analyze")

    # Perform analysis
    results = parse(code_samples, ts_parser)

    # Print summary
    total_accesses = sum(len(res["array_accesses"]) for res in results)
    print(f"\n--- Analysis Summary ---")
    print(f"Total array accesses found: {total_accesses}")

    # Save detailed results if output path is provided
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Detailed analysis saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to {args.output}: {e}")


if __name__ == "__main__":
    main()
