"""
Post-processing script to aggregate results across multiple seeds
Usage: python aggregate_results.py --results_dir ../models --output results_summary.csv
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re

def extract_results_from_directory(results_dir):
    """
    Extract results from all experiment directories
    
    Expected structure:
    models/
    ├── codebert/
    │   ├── diversevul_seed123/
    │   │   ├── experiment_summary.txt
    │   │   ├── threshold_comparison.txt
    │   │   └── predictions.txt
    │   ├── diversevul_seed456/
    │   └── icvul_seed123/
    ├── natgen/
    └── ...
    """
    all_results = []
    
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # Parse experiment directory name: "dataset_seed123" or "dataset_pos2.0_seed123"
            exp_name = exp_dir.name
            
            # Extract dataset, seed, and other parameters
            if '_seed' in exp_name:
                parts = exp_name.split('_seed')
                dataset_part = parts[0]
                seed = int(parts[1])
                
                # Check if there are additional parameters (like pos_weight)
                if '_pos' in dataset_part:
                    dataset_parts = dataset_part.split('_pos')
                    dataset = dataset_parts[0]
                    pos_weight = float(dataset_parts[1])
                else:
                    dataset = dataset_part
                    pos_weight = 1.0
            else:
                continue  # Skip directories that don't match expected pattern
            
            # Look for results files
            threshold_file = exp_dir / "threshold_comparison.txt"
            summary_file = exp_dir / "experiment_summary.txt"
            
            if threshold_file.exists():
                results = parse_threshold_comparison(threshold_file)
                if results:
                    results.update({
                        'model': model_name,
                        'dataset': dataset,
                        'seed': seed,
                        'pos_weight': pos_weight,
                        'exp_dir': str(exp_dir)
                    })
                    all_results.append(results)
                    print(f"  ✓ {exp_name}: F1={results.get('optimal_f1', 'N/A'):.4f}")
                else:
                    print(f"  ✗ {exp_name}: Could not parse results")
            else:
                print(f"  ✗ {exp_name}: No threshold_comparison.txt found")
    
    return all_results

def parse_threshold_comparison(threshold_file):
    """Parse the threshold_comparison.txt file to extract metrics"""
    try:
        with open(threshold_file, 'r') as f:
            content = f.read()
        
        results = {}
        
        # Extract default threshold results
        default_match = re.search(r'Default Threshold \(0\.5\):\s*\n(.*?)(?=\n\nOptimal|$)', content, re.DOTALL)
        if default_match:
            default_section = default_match.group(1)
            results['default_accuracy'] = extract_metric(default_section, 'accuracy')
            results['default_precision'] = extract_metric(default_section, 'precision') 
            results['default_recall'] = extract_metric(default_section, 'recall')
            results['default_f1'] = extract_metric(default_section, 'f1')
        
        # Extract optimal threshold results
        optimal_match = re.search(r'Optimal Threshold \(([\d.]+)\):\s*\n(.*?)(?=\n\nImprovement|$)', content, re.DOTALL)
        if optimal_match:
            optimal_threshold = float(optimal_match.group(1))
            optimal_section = optimal_match.group(2)
            results['optimal_threshold'] = optimal_threshold
            results['optimal_accuracy'] = extract_metric(optimal_section, 'accuracy')
            results['optimal_precision'] = extract_metric(optimal_section, 'precision')
            results['optimal_recall'] = extract_metric(optimal_section, 'recall') 
            results['optimal_f1'] = extract_metric(optimal_section, 'f1')
        
        # Extract improvement
        improvement_match = re.search(r'Improvement: F1 ([+-][\d.]+)', content)
        if improvement_match:
            results['f1_improvement'] = float(improvement_match.group(1))
        
        return results
    
    except Exception as e:
        print(f"Error parsing {threshold_file}: {e}")
        return None

def extract_metric(text, metric_name):
    """Extract a specific metric value from text"""
    pattern = rf'{metric_name}:\s*([\d.]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None

def aggregate_results(all_results):
    """Aggregate results by model, dataset, and pos_weight"""
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results found!")
        return pd.DataFrame()
    
    # Group by model, dataset, pos_weight
    grouped = df.groupby(['model', 'dataset', 'pos_weight'])
    
    aggregated = []
    
    for (model, dataset, pos_weight), group in grouped:
        if len(group) < 2:
            print(f"Warning: Only {len(group)} run(s) for {model}/{dataset}/pos{pos_weight}")
        
        # Calculate mean and std for key metrics
        metrics = ['optimal_f1', 'optimal_accuracy', 'optimal_precision', 'optimal_recall', 
                  'default_f1', 'default_accuracy', 'f1_improvement']
        
        agg_result = {
            'model': model,
            'dataset': dataset, 
            'pos_weight': pos_weight,
            'n_seeds': len(group),
            'seeds': sorted(group['seed'].tolist())
        }
        
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    agg_result[f'{metric}_mean'] = values.mean()
                    agg_result[f'{metric}_std'] = values.std() if len(values) > 1 else 0.0
                    agg_result[f'{metric}_min'] = values.min()
                    agg_result[f'{metric}_max'] = values.max()
        
        aggregated.append(agg_result)
    
    return pd.DataFrame(aggregated)

def format_results_table(df):
    """Format results for publication"""
    if df.empty:
        return df
    
    # Create publication-ready table
    pub_df = df.copy()
    
    # Format mean ± std columns
    for metric in ['optimal_f1', 'optimal_accuracy', 'optimal_precision', 'optimal_recall']:
        if f'{metric}_mean' in df.columns and f'{metric}_std' in df.columns:
            pub_df[f'{metric}_formatted'] = pub_df.apply(
                lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}", 
                axis=1
            )
    
    # Select and reorder columns for publication
    pub_columns = ['model', 'dataset', 'pos_weight', 'n_seeds', 
                   'optimal_f1_formatted', 'optimal_accuracy_formatted', 
                   'optimal_precision_formatted', 'optimal_recall_formatted']
    
    pub_df = pub_df[pub_columns].copy()
    pub_df.columns = ['Model', 'Dataset', 'Pos Weight', 'Seeds', 
                      'F1', 'Accuracy', 'Precision', 'Recall']
    
    return pub_df

def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results across seeds')
    parser.add_argument('--results_dir', default='../models', help='Directory containing model results')
    parser.add_argument('--output', default='results_summary.csv', help='Output CSV file')
    parser.add_argument('--min_seeds', default=3, type=int, help='Minimum seeds required for inclusion')
    
    args = parser.parse_args()
    
    print(f"Scanning results directory: {args.results_dir}")
    all_results = extract_results_from_directory(args.results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"\nFound {len(all_results)} individual experiment results")
    
    # Aggregate results
    aggregated_df = aggregate_results(all_results)
    
    # Filter by minimum seeds
    aggregated_df = aggregated_df[aggregated_df['n_seeds'] >= args.min_seeds]
    
    print(f"Aggregated into {len(aggregated_df)} model/dataset combinations")
    
    # Save detailed results
    detailed_output = args.output.replace('.csv', '_detailed.csv')
    aggregated_df.to_csv(detailed_output, index=False)
    print(f"Saved detailed results to: {detailed_output}")
    
    # Create publication table
    pub_df = format_results_table(aggregated_df)
    pub_df.to_csv(args.output, index=False)
    print(f"Saved publication table to: {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(pub_df.to_string(index=False))
    
    # Identify best performing model per dataset
    print("\n" + "="*80)
    print("BEST MODELS PER DATASET")
    print("="*80)
    
    for dataset in aggregated_df['dataset'].unique():
        dataset_results = aggregated_df[aggregated_df['dataset'] == dataset]
        best_idx = dataset_results['optimal_f1_mean'].idxmax()
        best = dataset_results.loc[best_idx]
        
        print(f"{dataset:15s}: {best['model']:12s} "
              f"F1={best['optimal_f1_mean']:.3f}±{best['optimal_f1_std']:.3f} "
              f"({best['n_seeds']} seeds)")

if __name__ == "__main__":
    main()
