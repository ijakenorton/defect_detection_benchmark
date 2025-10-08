#!/usr/bin/env python3
"""
Anonymize C/C++ code by:
1. Replacing function names with func_N
2. Removing all comments
3. Optionally anonymizing variable names

Works on code snippets - doesn't require compilable code.
"""

import argparse
import sys
from pathlib import Path
from tree_sitter_language_pack import get_language, get_parser

# Increase recursion limit for deeply nested ASTs
sys.setrecursionlimit(10000)


def collect_function_names(root_node, source_code):
    """Collect all function definition names and their byte ranges."""
    functions = []

    def traverse(node):
        if node.type == "function_definition":
            # Get the declarator which contains the function name
            declarator = node.child_by_field_name("declarator")
            if declarator:
                # Handle direct function_declarator or pointer_declarator
                func_declarator = declarator
                if declarator.type == "pointer_declarator":
                    # Navigate through pointer declarators to find function_declarator
                    for child in declarator.children:
                        if child.type == "function_declarator":
                            func_declarator = child
                            break

                if func_declarator.type == "function_declarator":
                    # Try to get the function name via the "declarator" field first
                    name_node = func_declarator.child_by_field_name("declarator")

                    if name_node:
                        # Handle parenthesized declarators
                        while name_node.type == "parenthesized_declarator":
                            name_node = name_node.child_by_field_name("declarator")

                        if name_node.type in ["identifier", "field_identifier"]:
                            func_name = source_code[name_node.start_byte:name_node.end_byte]
                            functions.append({
                                "name": func_name,
                                "start": name_node.start_byte,
                                "end": name_node.end_byte
                            })
                    else:
                        # If no "declarator" field, check direct children for identifier/field_identifier
                        for child in func_declarator.children:
                            if child.type in ["identifier", "field_identifier"]:
                                func_name = source_code[child.start_byte:child.end_byte]
                                functions.append({
                                    "name": func_name,
                                    "start": child.start_byte,
                                    "end": child.end_byte
                                })
                                break

        for child in node.children:
            traverse(child)

    traverse(root_node)
    return functions


def collect_comments(root_node):
    """Collect all comment nodes and their byte ranges."""
    comments = []

    def traverse(node):
        if node.type == "comment":
            comments.append({
                "start": node.start_byte,
                "end": node.end_byte
            })

        for child in node.children:
            traverse(child)

    traverse(root_node)
    return comments


def collect_variables(root_node, source_code, exclude_functions=None):
    """Collect all variable names and their byte ranges."""
    if exclude_functions is None:
        exclude_functions = set()

    variables = {}  # {var_name: [list of byte ranges]}

    def traverse(node):
        if node.type == "identifier":
            var_name = source_code[node.start_byte:node.end_byte]

            # Skip if it's a function name we're already anonymizing
            if var_name in exclude_functions:
                return

            # Check if it's a variable declaration or usage
            parent = node.parent
            if parent and parent.type in [
                "declaration",
                "parameter_declaration",
                "init_declarator",
                "assignment_expression",
                "binary_expression",
                "call_expression"
            ]:
                if var_name not in variables:
                    variables[var_name] = []
                variables[var_name].append({
                    "start": node.start_byte,
                    "end": node.end_byte
                })

        for child in node.children:
            traverse(child)

    traverse(root_node)
    return variables


def apply_replacements(source_code, replacements):
    """
    Apply replacements to source code.
    Replacements should be a list of dicts with 'start', 'end', and 'new_text'.
    They will be sorted in reverse order to avoid offset issues.
    """
    # Sort by start position in reverse order
    replacements = sorted(replacements, key=lambda x: x["start"], reverse=True)

    result = source_code
    for repl in replacements:
        result = result[:repl["start"]] + repl["new_text"] + result[repl["end"]:]

    return result


def anonymize_code(source_code, remove_comments=True, anonymize_functions=True,
                   anonymize_variables=False, language_type="c"):
    """
    Anonymize C/C++ source code.

    Args:
        source_code: String containing C/C++ code
        remove_comments: Remove all comments
        anonymize_functions: Replace function names with func_N
        anonymize_variables: Replace variable names with var_N
        language_type: "c" or "cpp" for parser selection

    Returns:
        Anonymized source code as string
    """
    language = get_language(language_type)
    parser = get_parser(language_type)

    # Parse the code
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node

    # Check if parsing succeeded
    if root_node is None:
        # Parser failed, return original code
        return source_code

    replacements = []

    # Collect and anonymize function names
    function_names = set()
    if anonymize_functions:
        functions = collect_function_names(root_node, source_code)
        for i, func in enumerate(functions):
            function_names.add(func["name"])
            replacements.append({
                "start": func["start"],
                "end": func["end"],
                "new_text": f"func_{i}"
            })

    # Remove comments
    if remove_comments:
        comments = collect_comments(root_node)
        for comment in comments:
            # Replace comment with empty string (or single space to avoid token merging)
            replacements.append({
                "start": comment["start"],
                "end": comment["end"],
                "new_text": ""
            })

    # Anonymize variables
    if anonymize_variables:
        variables = collect_variables(root_node, source_code, exclude_functions=function_names)
        var_counter = 0
        var_mapping = {}

        for var_name, occurrences in variables.items():
            new_name = f"var_{var_counter}"
            var_mapping[var_name] = new_name
            var_counter += 1

            for occurrence in occurrences:
                replacements.append({
                    "start": occurrence["start"],
                    "end": occurrence["end"],
                    "new_text": new_name
                })

    # Apply all replacements
    result = apply_replacements(source_code, replacements)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize C/C++ code by replacing function names and removing comments"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (if not provided, writes to stdout)"
    )
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Keep comments in the output"
    )
    parser.add_argument(
        "--no-functions",
        action="store_true",
        help="Don't anonymize function names"
    )
    parser.add_argument(
        "--anonymize-vars",
        action="store_true",
        help="Also anonymize variable names"
    )
    parser.add_argument(
        "--lang",
        choices=["c", "cpp"],
        help="Language type (auto-detected from file extension if not specified)"
    )

    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, "r") as f:
            source_code = f.read()
    else:
        source_code = sys.stdin.read()

    # Auto-detect language from file extension if not specified
    language_type = args.lang
    if not language_type and args.input:
        ext = Path(args.input).suffix.lower()
        if ext in ['.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.h++']:
            language_type = "cpp"
        else:
            language_type = "c"
    elif not language_type:
        language_type = "c"  # Default to C

    # Anonymize
    result = anonymize_code(
        source_code,
        remove_comments=not args.keep_comments,
        anonymize_functions=not args.no_functions,
        anonymize_variables=args.anonymize_vars,
        language_type=language_type
    )

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
