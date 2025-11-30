#!/usr/bin/env python3
"""
Script to create a comprehensive convergence rate comparison table.
Exports results to CSV and creates formatted markdown tables.
"""

import os
import csv
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Base results directory
RESULTS_DIR = "/home/sujit_2021cs35/Github/FedSat/RESULTS/results"

def extract_method_name(folder_name):
    """Extract method name from folder name."""
    parts = folder_name.split('_')
    return parts[0].upper()

def find_round_for_accuracy(csv_path, target_accuracy):
    """Find the first round where global_accuracy >= target_accuracy."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('global_accuracy') and row.get('global_accuracy').strip():
                    try:
                        acc = float(row['global_accuracy'])
                        if acc >= target_accuracy:
                            return int(row['round'])
                    except ValueError:
                        continue
        return None
    except Exception as e:
        return None

def main():
    # Dictionary to store results
    results = defaultdict(lambda: defaultdict(dict))
    
    # Target accuracies
    targets = [0.20, 0.40, 0.60]
    
    # Iterate through all dataset directories
    for dataset_dir in sorted(os.listdir(RESULTS_DIR)):
        dataset_path = os.path.join(RESULTS_DIR, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        # Iterate through all method directories
        for method_dir in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method_dir)
            if not os.path.isdir(method_path):
                continue
            
            csv_file = os.path.join(method_path, "test_stats.csv")
            if not os.path.exists(csv_file):
                continue
            
            method_name = extract_method_name(method_dir)
            
            # Extract convergence information
            convergence_info = {}
            for target in targets:
                round_num = find_round_for_accuracy(csv_file, target)
                convergence_info[target] = round_num
            
            results[dataset_dir][method_name] = convergence_info
    
    # Create CSV files and markdown table
    csv_file = "/home/sujit_2021cs35/Github/FedSat/convergence_comparison_table.csv"
    markdown_file = "/home/sujit_2021cs35/Github/FedSat/convergence_comparison_table.md"
    json_file = "/home/sujit_2021cs35/Github/FedSat/convergence_comparison_table.json"
    
    # Export to JSON
    with open(json_file, 'w') as f:
        json.dump(dict(results), f, indent=2)
    
    # Create CSV with all data
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset Setting', 'Method', '20% Acc (Round)', '40% Acc (Round)', '60% Acc (Round)'])
        
        for dataset_setting in sorted(results.keys()):
            methods_data = results[dataset_setting]
            for method in sorted(methods_data.keys()):
                convergence = methods_data[method]
                r20 = convergence.get(0.20)
                r40 = convergence.get(0.40)
                r60 = convergence.get(0.60)
                
                writer.writerow([
                    dataset_setting,
                    method,
                    r20 if r20 is not None else 'Not reached',
                    r40 if r40 is not None else 'Not reached',
                    r60 if r60 is not None else 'Not reached'
                ])
    
    # Create markdown table
    with open(markdown_file, 'w') as f:
        f.write("# Convergence Rate Comparison Table\n\n")
        f.write("This table shows the round at which each method reaches 20%, 40%, and 60% global accuracy.\n\n")
        
        for dataset_setting in sorted(results.keys()):
            f.write(f"## Dataset Setting: {dataset_setting}\n\n")
            
            methods_data = results[dataset_setting]
            
            # Create markdown table
            f.write("| Method | 20% Acc (Round) | 40% Acc (Round) | 60% Acc (Round) |\n")
            f.write("|--------|-----------------|-----------------|-----------------|>\n")
            
            for method in sorted(methods_data.keys()):
                convergence = methods_data[method]
                r20 = convergence.get(0.20)
                r40 = convergence.get(0.40)
                r60 = convergence.get(0.60)
                
                r20_str = str(r20) if r20 is not None else '✗ Not reached'
                r40_str = str(r40) if r40 is not None else '✗ Not reached'
                r60_str = str(r60) if r60 is not None else '✗ Not reached'
                
                f.write(f"| {method:<30} | {r20_str:<15} | {r40_str:<15} | {r60_str:<15} |\n")
            
            f.write("\n")
    
    print(f"\n✓ CSV file saved to: {csv_file}")
    print(f"✓ Markdown file saved to: {markdown_file}")
    print(f"✓ JSON file saved to: {json_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    all_methods = set()
    for methods in results.values():
        all_methods.update(methods.keys())
    
    print(f"\nTotal datasets analyzed: {len(results)}")
    print(f"Total unique methods: {len(all_methods)}")
    print(f"Methods: {', '.join(sorted(all_methods))}")
    
    # Calculate average convergence rounds per method
    print("\n" + "-"*80)
    print("Average rounds to reach target accuracy (across all datasets):")
    print("-"*80)
    
    method_stats = defaultdict(lambda: {'20': [], '40': [], '60': []})
    
    for dataset_setting, methods_data in results.items():
        for method, convergence in methods_data.items():
            for target, round_num in convergence.items():
                if round_num is not None:
                    key = str(int(target * 100))
                    method_stats[method][key].append(round_num)
    
    print(f"{'Method':<15} {'Avg to 20%':<15} {'Avg to 40%':<15} {'Avg to 60%':<15}")
    print("-"*80)
    
    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        avg_20 = f"{sum(stats['20'])/len(stats['20']):.1f}" if stats['20'] else "N/A"
        avg_40 = f"{sum(stats['40'])/len(stats['40']):.1f}" if stats['40'] else "N/A"
        avg_60 = f"{sum(stats['60'])/len(stats['60']):.1f}" if stats['60'] else "N/A"
        
        print(f"{method:<15} {avg_20:<15} {avg_40:<15} {avg_60:<15}")

if __name__ == "__main__":
    main()
