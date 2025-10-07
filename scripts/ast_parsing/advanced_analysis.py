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


def detect_common_vulnerability_patterns(code, parser):
    """Detect common vulnerability patterns in code."""
    patterns = {
        "buffer_operations": 0,
        "pointer_arithmetic": 0,
        "memory_allocations": 0,
        "unsafe_functions": 0,
        "integer_operations": 0,
        "unchecked_returns": 0,
        "global_variables": 0,
    }

    # Detect unsafe functions via regex
    unsafe_functions = [
        "strcpy",
        "strcat",
        "sprintf",
        "gets",
        "scanf",
        "memcpy",
        "malloc",
        "free",
        "realloc",
        "calloc",
    ]

    for func in unsafe_functions:
        if re.search(r"\b" + func + r"\s*\(", code):
            patterns["unsafe_functions"] += 1

    # Parse the code with tree-sitter
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # Initialize counters
    buffer_ops = 0
    pointer_arith = 0
    memory_allocs = 0
    integer_ops = 0
    unchecked_returns = 0
    global_vars = 0

    # Function to traverse the AST and find patterns
    def traverse_tree(node):
        nonlocal buffer_ops, pointer_arith, memory_allocs, integer_ops, unchecked_returns, global_vars

        # Check for buffer operations
        if node.type == "subscript_expression":
            buffer_ops += 1

        # Check for pointer arithmetic
        if node.type == "binary_expression":
            left_type = node.child_by_field_name("left")
            if left_type and "*" in code[left_type.start_byte : left_type.end_byte]:
                pointer_arith += 1

        # Check for memory allocations
        if node.type == "call_expression":
            func_name = node.child_by_field_name("function")
            if func_name:
                func_name_str = code[func_name.start_byte : func_name.end_byte]
                if func_name_str in ["malloc", "calloc", "realloc"]:
                    memory_allocs += 1

        # Check for integer operations that could overflow
        if node.type == "binary_expression":
            op = node.child_by_field_name("operator")
            if op and op.type in ["+", "-", "*", "/"]:
                integer_ops += 1

        # Check for function calls with unchecked returns
        if node.type == "call_expression":
            parent = node.parent
            if (
                parent
                and parent.type != "if_statement"
                and parent.type != "assignment_expression"
            ):
                unchecked_returns += 1

        # Check for global variables
        if node.type == "declaration" and node.parent == root_node:
            global_vars += 1

        # Recursively check children
        for child in node.children:
            traverse_tree(child)

    # Start traversal
    traverse_tree(root_node)

    # Update patterns dictionary
    patterns["buffer_operations"] = buffer_ops
    patterns["pointer_arithmetic"] = pointer_arith
    patterns["memory_allocations"] = memory_allocs
    patterns["integer_operations"] = integer_ops
    patterns["unchecked_returns"] = unchecked_returns
    patterns["global_variables"] = global_vars


    return patterns


def calculate_entropy(code):
    """Calculate Shannon entropy of the code as a measure of complexity."""
    # Count character frequencies
    char_counts = Counter(code)
    total_chars = len(code)

    # Calculate entropy
    entropy = 0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * np.log2(probability)

    return entropy


def analyze_code_complexity(code, parser):
    """Analyze code complexity metrics."""
    metrics = {
        "entropy": calculate_entropy(code),
        "avg_line_length": np.mean([len(line) for line in code.split("\n")]),
        "unique_variables": 0,
        "function_parameters": 0,
        "cyclomatic_complexity": 0,
    }

    # Parse with tree-sitter
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # Count unique variables
    variable_names = set()

    def find_variables(node):
        if node.type == "identifier":
            parent = node.parent
            if parent and parent.type in ["declaration", "parameter_declaration"]:
                variable_names.add(code[node.start_byte : node.end_byte])

        for child in node.children:
            find_variables(child)

    find_variables(root_node)
    metrics["unique_variables"] = len(variable_names)

    # Count function parameters
    parameter_count = 0
    for node in root_node.children:
        if node.type == "function_definition":
            param_list = node.child_by_field_name("parameters")
            if param_list:
                parameter_count += len(
                    [
                        c
                        for c in param_list.children
                        if c.type == "parameter_declaration"
                    ]
                )

    metrics["function_parameters"] = parameter_count

    # Calculate cyclomatic complexity (number of decisions + 1)
    decision_nodes = 0

    def count_decisions(node):
        nonlocal decision_nodes
        if node.type in [
            "if_statement",
            "for_statement",
            "while_statement",
            "case_statement",
            "&&",
            "||",
        ]:
            decision_nodes += 1

        for child in node.children:
            count_decisions(child)

    count_decisions(root_node)
    metrics["cyclomatic_complexity"] = decision_nodes + 1

    return metrics


def analyze_samples(code_samples, categories, parser):
    """Perform detailed analysis on code samples by category."""
    results = {}

    for category, indices in categories.items():
        vulnerability_patterns = []
        complexity_metrics = []

        for idx in indices:
            if idx in code_samples:
                code = code_samples[idx]
                try:
                    patterns = detect_common_vulnerability_patterns(code, parser)
                    complexity = analyze_code_complexity(code, parser)

                    vulnerability_patterns.append(patterns)
                    complexity_metrics.append(complexity)
                except Exception as e:
                    print(f"Error analyzing sample {idx}: {e}")

        # Calculate averages
        if vulnerability_patterns:
            avg_patterns = {
                pattern: np.mean([vp[pattern] for vp in vulnerability_patterns])
                for pattern in vulnerability_patterns[0].keys()
            }

            avg_complexity = {
                metric: np.mean([cm[metric] for cm in complexity_metrics])
                for metric in complexity_metrics[0].keys()
            }

            results[category] = {
                "vulnerability_patterns": avg_patterns,
                "complexity_metrics": avg_complexity,
                "count": len(vulnerability_patterns),
            }

    return results


def print_advanced_analysis(results):
    """Print the advanced analysis results."""
    print("\n===== ADVANCED ANALYSIS RESULTS =====")

    # Print number of samples in each category
    print("\nSample Counts:")
    for category, data in results.items():
        print(f"{category}: {data['count']} samples")

    # Print vulnerability patterns
    print("\nVulnerability Patterns (average occurrences per sample):")
    if results:
        patterns = list(next(iter(results.values()))["vulnerability_patterns"].keys())

        for pattern in patterns:
            print(f"\n{pattern.upper().replace('_', ' ')}:")
            for category in ["TP", "TN", "FP", "FN"]:
                if category in results:
                    value = results[category]["vulnerability_patterns"][pattern]
                    print(f"  {category}: {value:.2f}")

    # Print complexity metrics
    print("\nCode Complexity Metrics (average per sample):")
    if results:
        metrics = list(next(iter(results.values()))["complexity_metrics"].keys())

        for metric in metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            for category in ["TP", "TN", "FP", "FN"]:
                if category in results:
                    value = results[category]["complexity_metrics"][metric]
                    print(f"  {category}: {value:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced vulnerability pattern analysis"
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to the predictions file"
    )
    parser.add_argument("--data", required=True, help="Path to the test data file")
    parser.add_argument("--output", help="Path to save detailed analysis results")

    args = parser.parse_args()

    # Load data
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.data)
    code_samples = load_code_samples(args.data)

    # Categorize predictions
    categories = categorize_predictions(predictions, ground_truth)

    # Set up Tree-sitter
    ts_parser = setup_treesitter()

    # Perform advanced analysis
    results = analyze_samples(code_samples, categories, ts_parser)

    # Print analysis results
    print_advanced_analysis(results)

    # Save detailed results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed analysis saved to {args.output}")


if __name__ == "__main__":
    main()
