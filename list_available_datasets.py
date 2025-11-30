#!/usr/bin/env python3
"""
Interactive utility to generate distribution figures for any available dataset/beta combination.

Usage:
    python list_available_datasets.py
    
This will:
1. Scan the DATA directory for available datasets
2. Show which datasets and beta values are available
3. Generate figures for any available combination
"""

import os
import glob
import re

DATASET_DIR = "./DATA"


def find_all_datasets():
    """Find all available dataset configurations."""
    results = {}
    
    for dataset in os.listdir(DATASET_DIR):
        dataset_path = os.path.join(DATASET_DIR, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        # Find all noiid_lbldir configurations
        pattern = os.path.join(dataset_path, "noiid_lbldir_b*_k*")
        matches = glob.glob(pattern)
        
        if matches:
            results[dataset] = []
            for match in matches:
                dir_name = os.path.basename(match)
                # Extract beta and k values using regex
                m = re.search(r'b([\d_]+)_k(\d+)', dir_name)
                if m:
                    beta_str = m.group(1)
                    k = m.group(2)
                    beta = float(beta_str.replace('_', '.'))
                    
                    # Count client files
                    client_files = glob.glob(os.path.join(match, "*.pkl"))
                    num_clients = len([f for f in client_files if os.path.basename(f)[:-4].isdigit()])
                    
                    results[dataset].append({
                        'beta': beta,
                        'k': int(k),
                        'num_clients': num_clients,
                        'path': match
                    })
            
            # Sort by beta
            results[dataset] = sorted(results[dataset], key=lambda x: x['beta'])
    
    return results


def print_available_datasets():
    """Print all available datasets in a formatted table."""
    datasets = find_all_datasets()
    
    if not datasets:
        print("❌ No datasets found in ./DATA/")
        return
    
    print("\n" + "="*80)
    print("AVAILABLE DATASETS AND CONFIGURATIONS")
    print("="*80)
    
    for dataset in sorted(datasets.keys()):
        configs = datasets[dataset]
        print(f"\n📊 {dataset.upper()}:")
        print(f"   {'Beta':<10} {'K':<10} {'Clients':<15} {'Path':<45}")
        print("   " + "-"*80)
        
        for config in configs:
            beta_str = f"{config['beta']:.2f}"
            print(f"   {beta_str:<10} {config['k']:<10} {config['num_clients']:<15} "
                  f"{os.path.basename(config['path']):<45}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Datasets: {len(datasets)}")
    total_configs = sum(len(configs) for configs in datasets.values())
    print(f"Total Configurations: {total_configs}")
    
    print("\n💡 To generate figures for a specific configuration, use:")
    print("   python generate_distribution_figures.py")
    print("\n📝 Edit the BETA_VALUES and DATASETS lists in generate_distribution_figures.py")
    print("   to customize which figures to generate.")
    print()


if __name__ == "__main__":
    print_available_datasets()
