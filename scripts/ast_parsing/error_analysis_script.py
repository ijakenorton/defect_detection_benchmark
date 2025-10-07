#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from collections import defaultdict
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_c


def load_predictions(predictions_file):
    """Load model predictions."""
    predictions = {}
    with open(predictions_file, "r") as f:
        for line in f:
            idx, label = line.strip().split()
            predictions[int(idx)] = int(label)
    return predictions


def load_ground_truth(data_file):
    """Load ground truth labels from the dataset."""
    ground_truth = {}
    with open(data_file, "r") as f:
        for line in f:
            js = json.loads(line.strip())
            ground_truth[int(js["idx"])] = js["target"]
    return ground_truth


def load_code_samples(data_file):
    """Load the actual code samples from the dataset."""
    code_samples = {}
    with open(data_file, "r") as f:
        for line in f:
            js = json.loads(line.strip())
            code_samples[int(js["idx"])] = js["func"]
    return code_samples


def categorize_predictions(predictions, ground_truth):
    """Categorize predictions into TP, TN, FP, FN."""
    categories = {
        "TP": [],  # True Positives
        "TN": [],  # True Negatives
        "FP": [],  # False Positives
        "FN": [],  # False Negatives
    }

    for idx, pred in predictions.items():
        if idx not in ground_truth:
            print(f"Warning: Sample {idx} not found in ground truth")
            continue

        true_label = ground_truth[idx]

        if pred == 1 and true_label == 1:
            categories["TP"].append(idx)
        elif pred == 0 and true_label == 0:
            categories["TN"].append(idx)
        elif pred == 1 and true_label == 0:
            categories["FP"].append(idx)
        elif pred == 0 and true_label == 1:
            categories["FN"].append(idx)

    return categories


def analyze_code_structure(code, parser):
    """Analyze code structure using Tree-sitter without using TreeCursor."""
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        # Initialize analysis metrics
        metrics = {
            "loc": len(code.split("\n")),
            "max_nesting_depth": 0,
            "num_if_statements": 0,
            "num_loops": 0,
            "num_function_calls": 0,
        }

        # Use a recursive approach instead of TreeCursor
        def count_nodes(node):
            if_statements = 0
            loops = 0
            function_calls = 0

            # Count this node if it matches what we're looking for
            if node.type == "if_statement":
                if_statements += 1
            elif node.type in ["for_statement", "while_statement", "do_statement"]:
                loops += 1
            elif node.type == "call_expression":
                function_calls += 1

            # Recursively count all children
            for child in node.children:
                child_ifs, child_loops, child_calls = count_nodes(child)
                if_statements += child_ifs
                loops += child_loops
                function_calls += child_calls

            return if_statements, loops, function_calls

        # Calculate nesting depth
        def calculate_max_depth(node, current_depth=0):
            max_depth = current_depth

            if node.type in [
                "compound_statement",
                "if_statement",
                "for_statement",
                "while_statement",
                "do_statement",
            ]:
                current_depth += 1
                max_depth = max(max_depth, current_depth)

            for child in node.children:
                child_depth = calculate_max_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)

            return max_depth

        # Count nodes and update metrics
        ifs, loops, calls = count_nodes(root_node)
        metrics["num_if_statements"] = ifs
        metrics["num_loops"] = loops
        metrics["num_function_calls"] = calls

        # Calculate max nesting depth
        metrics["max_nesting_depth"] = calculate_max_depth(root_node)

        return metrics

    except Exception as e:
        print(f"Error in analyze_code_structure: {e}")
        # Return default metrics if parsing fails
        return {
            "loc": len(code.split("\n")),
            "max_nesting_depth": 0,
            "num_if_statements": 0,
            "num_loops": 0,
            "num_function_calls": 0,
        }


def analyze_code_by_category(code_samples, categories, parser):
    """Analyze code structure for each category."""
    category_metrics = {}

    for category, indices in categories.items():
        metrics_list = []

        for idx in indices:
            if idx in code_samples:
                code = code_samples[idx]
                try:
                    metrics = analyze_code_structure(code, parser)
                    metrics_list.append(metrics)
                except Exception as e:
                    print(f"Error analyzing sample {idx}: {e}")

        # Calculate average metrics for this category
        if metrics_list:
            avg_metrics = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0].keys()
            }
            category_metrics[category] = {
                "avg": avg_metrics,
                "count": len(metrics_list),
            }

    return category_metrics


def print_analysis_results(category_metrics):
    """Print analysis results."""
    print("\n===== ANALYSIS RESULTS =====")

    # Print number of samples in each category
    print("\nSample Counts:")
    for category, data in category_metrics.items():
        print(f"{category}: {data['count']} samples")

    # Print average metrics for each category
    print("\nAverage Metrics by Category:")
    metrics = list(next(iter(category_metrics.values()))["avg"].keys())

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for category in ["TP", "TN", "FP", "FN"]:
            if category in category_metrics:
                value = category_metrics[category]["avg"][metric]
                print(f"  {category}: {value:.2f}")


def setup_treesitter():

    from tree_sitter_language_pack import get_language, get_parser

    language = get_language("c")
    parser = get_parser("c")

    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model errors using Tree-sitter"
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

    # Analyze code structure
    category_metrics = analyze_code_by_category(code_samples, categories, ts_parser)

    # Print analysis results
    print_analysis_results(category_metrics)

    # Save detailed results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(category_metrics, f, indent=2)
        print(f"\nDetailed analysis saved to {args.output}")


if __name__ == "__main__":
    main()
