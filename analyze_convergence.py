#!/usr/bin/env python3
"""
Script to analyze convergence rates across different methods and dataset settings.
Creates a table showing at which round each method reaches 20%, 40%, and 60% global accuracy.
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

# Base results directory
RESULTS_DIR = "/home/sujit_2021cs35/Github/FedSat/RESULTS/results"

def extract_method_name(folder_name):
    """Extract method name from folder name."""
    # Split by underscore and take the first part before hyperparameters
    parts = folder_name.split('_')
    return parts[0].upper()

def extract_dataset_setting(dataset_dir):
    """Extract dataset setting from directory name."""
    # Format: dataset_distribution_beta_k{num_clients}
    return dataset_dir

def find_round_for_accuracy(csv_path, target_accuracy):
    """
    Find the first round where global_accuracy >= target_accuracy.
    Returns the round number or None if never reached.
    """
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
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    # Dictionary to store results: {dataset_setting: {method: {accuracy_threshold: round}}}
    results = defaultdict(lambda: defaultdict(dict))
    
    # Target accuracies
    targets = [0.20, 0.40, 0.60]
    
    # Iterate through all dataset directories
    for dataset_dir in sorted(os.listdir(RESULTS_DIR)):
        dataset_path = os.path.join(RESULTS_DIR, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"Processing dataset: {dataset_dir}")
        
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
    
    # Generate output table
    output_file = "/home/sujit_2021cs35/Github/FedSat/convergence_comparison_table.json"
    
    with open(output_file, 'w') as f:
        json.dump(dict(results), f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary table to console
    print("\n" + "="*120)
    print("CONVERGENCE RATE COMPARISON TABLE")
    print("(Round at which methods reach target global accuracy)")
    print("="*120)
    
    for dataset_setting in sorted(results.keys()):
        print(f"\nDataset Setting: {dataset_setting}")
        print("-" * 120)
        print(f"{'Method':<20} {'20% Acc (Round)':<20} {'40% Acc (Round)':<20} {'60% Acc (Round)':<20}")
        print("-" * 120)
        
        methods_data = results[dataset_setting]
        for method in sorted(methods_data.keys()):
            convergence = methods_data[method]
            r20 = convergence.get(0.20, '-')
            r40 = convergence.get(0.40, '-')
            r60 = convergence.get(0.60, '-')
            
            # Format output
            r20_str = str(r20) if r20 is not None else 'Not reached'
            r40_str = str(r40) if r40 is not None else 'Not reached'
            r60_str = str(r60) if r60 is not None else 'Not reached'
            
            print(f"{method:<20} {r20_str:<20} {r40_str:<20} {r60_str:<20}")

if __name__ == "__main__":
    main()
