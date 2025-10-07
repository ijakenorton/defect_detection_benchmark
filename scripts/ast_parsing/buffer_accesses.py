#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from collections import defaultdict, Counter
import tree_sitter
from tree_sitter import Language, Parser
import re

# Import functions from our basic analysis script
from error_analysis_script import (
    load_predictions,
    load_ground_truth,
    load_code_samples,
    categorize_predictions,
    setup_treesitter,
)

node_types = set()


def print_buffer_operations(code, parser):
    """Detect common vulnerability patterns in code."""
    patterns = {
        "buffer_operations": 0,
    }

    # Parse the code with tree-sitter
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # Initialize counters
    buffer_ops = 0

    # Function to traverse the AST and find patterns
    def traverse_tree(node):
        global node_types

        node_types.add(node.type)
        nonlocal buffer_ops

        # if node.type == "function_definition":
        #     print(node.keys())
        #     exit(1)

        if node.type == "function_definition":
            print("function_definition")
            print(node)
            print("*********************************************")
        if node.type == "preproc_function_def":
            print("preproc_function_def")
            print(node)
            print("*********************************************")
        if node.type == "abstract_function_declarator":
            print("abstract_function_declarator")
            print(node)
            print("*********************************************")
        if node.type == "function_declarator":
            print("function_declarator")
            print(node)
            print("*********************************************")

        # Check for buffer operations
        if node.type == "subscript_expression":
            buffer_ops += 1

        # Recursively check children
        for child in node.children:
            traverse_tree(child)

    # Start traversal
    traverse_tree(root_node)

    # Update patterns dictionary
    patterns["buffer_operations"] = buffer_ops

    # if patterns["buffer_operations"] > 0:
    #     print("*********************************************")
    #     print(code)
    #     print("*********************************************")

    return patterns


def analyze_samples(code_samples, categories, parser, category_type):
    """Perform detailed analysis on code samples by category."""
    results = {}

    for category, indices in categories.items():
        if category != category_type:
            continue

        for idx in indices:
            if idx in code_samples:
                code = code_samples[idx]
                try:
                    print_buffer_operations(code, parser)
                except Exception as e:
                    print(f"Error analyzing sample {idx}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced vulnerability pattern analysis"
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to the predictions file"
    )
    parser.add_argument("--data", required=True, help="Path to the test data file")
    parser.add_argument("--output", help="Path to save detailed analysis results")
    parser.add_argument(
        "--type",
        choices=["TP", "FP", "TN", "FN"],
        help="Type of data you want TP FP TN FN",
    )

    args = parser.parse_args()

    # Load data
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.data)
    code_samples = load_code_samples(args.data)

    # Categorize predictions
    categories = categorize_predictions(predictions, ground_truth)

    # Set up Tree-sitter
    ts_parser = setup_treesitter()

    analyze_samples(code_samples, categories, ts_parser, args.type)
    print(node_types)


if __name__ == "__main__":
    main()
