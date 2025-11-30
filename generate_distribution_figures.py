#!/usr/bin/env python3
"""
Generate dataset distribution figures for different datasets and beta values.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "./DATA"
RESULTS_DIR = "./RESULTS/figures"
DATASETS = ["fmnist", "cifar", "cifar100"]
BETA_VALUES = [0.5, 0.3, 0.1, 0.05]
NUM_CLIENTS = 100

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_client_file_sizes(dataset, beta, num_clients=100):
    """Get file sizes as proxy for sample counts."""
    beta_str_options = [
        f"{beta:.2f}".replace(".", "_"),
        f"{beta:.1f}".replace(".", "_") if beta >= 0.1 else None,
    ]
    
    data_path = None
    for beta_str in beta_str_options:
        if beta_str is None:
            continue
        dataset_type = f"noiid_lbldir_b{beta_str}_k{num_clients}"
        test_path = os.path.join(DATASET_DIR, dataset, dataset_type)
        if os.path.exists(test_path):
            data_path = test_path
            break
    
    if data_path is None:
        return None, None, False
    
    file_sizes = []
    for client_id in range(num_clients):
        pkl_file = os.path.join(data_path, f"{client_id}.pkl")
        if os.path.exists(pkl_file):
            try:
                file_sizes.append(os.path.getsize(pkl_file))
            except:
                file_sizes.append(0)
        else:
            file_sizes.append(0)
    
    if all(s == 0 for s in file_sizes):
        return None, None, False
    
    avg_bytes_per_sample = 700
    sample_counts = [max(1, int(s / avg_bytes_per_sample)) for s in file_sizes]
    
    return sample_counts, data_path, True


def plot_distribution(dataset, beta, sample_counts, save_path):
    """Create a bar plot of sample distribution."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    client_ids = list(range(len(sample_counts)))
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    bar_colors = [colors[i % len(colors)] for i in range(len(sample_counts))]
    
    ax.bar(client_ids, sample_counts, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Client ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("Estimated Number of Samples", fontsize=12, fontweight='bold')
    ax.set_title(f"Data Distribution: {dataset.upper()} (β={beta})", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(0, len(sample_counts), 10))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_distribution_subplots(datasets, beta_values):
    """Create subplot figure with available datasets."""
    fig, axes = plt.subplots(len(datasets), len(beta_values), figsize=(18, 12))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Data Distribution Across Clients - Non-IID (Label Dirichlet)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for row, dataset in enumerate(datasets):
        for col, beta in enumerate(beta_values):
            ax = axes[row, col]
            
            sample_counts, data_path, found = get_client_file_sizes(dataset, beta, NUM_CLIENTS)
            
            if not found or sample_counts is None:
                ax.text(0.5, 0.5, f"Data not available\n{dataset}, β={beta}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            
            client_ids = list(range(len(sample_counts)))
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            bar_colors = [colors[i % len(colors)] for i in range(len(sample_counts))]
            
            ax.bar(client_ids, sample_counts, color=bar_colors, edgecolor='black', linewidth=0.3)
            ax.set_title(f'{dataset.upper()} (β={beta})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Client ID', fontsize=10)
            ax.set_ylabel('Est. Samples', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticks(range(0, NUM_CLIENTS, 20))
            
            sample_counts_arr = np.array(sample_counts)
            mean_samples = np.mean(sample_counts_arr)
            std_samples = np.std(sample_counts_arr)
            min_samples = np.min(sample_counts_arr)
            max_samples = np.max(sample_counts_arr)
            
            stats_text = f'μ={mean_samples:.0f}\nσ={std_samples:.0f}\nmin={min_samples}\nmax={max_samples}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "data_distribution_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined figure: {save_path}")
    plt.close()


def generate_statistics(datasets, beta_values):
    """Generate statistics for all combinations."""
    print("\n" + "="*100)
    print("DATA DISTRIBUTION STATISTICS (Estimated from File Sizes)")
    print("="*100)
    
    stats_table = []
    
    for dataset in datasets:
        for beta in beta_values:
            sample_counts, data_path, found = get_client_file_sizes(dataset, beta, NUM_CLIENTS)
            
            if not found or sample_counts is None:
                continue
            
            sample_counts = np.array(sample_counts)
            
            mean = np.mean(sample_counts)
            std = np.std(sample_counts)
            min_val = np.min(sample_counts)
            max_val = np.max(sample_counts)
            total = np.sum(sample_counts)
            
            stats_table.append({
                'Dataset': dataset.upper(),
                'Beta': f'{beta:.2f}',
                'Clients': len(sample_counts),
                'Mean': f'{mean:.1f}',
                'Std': f'{std:.1f}',
                'Min': int(min_val),
                'Max': int(max_val),
                'Total': int(total),
            })
            
            print(f"\n{dataset.upper()} (β={beta}):")
            print(f"  Data Path:         {data_path}")
            print(f"  Total Clients:     {len(sample_counts)}")
            print(f"  Mean Samples:      {mean:.2f}")
            print(f"  Std Dev:           {std:.2f}")
            print(f"  Min Samples:       {int(min_val)}")
            print(f"  Max Samples:       {int(max_val)}")
            print(f"  Total Samples:     {int(total)}")
            if min_val > 0:
                print(f"  Imbalance Ratio:   {max_val/min_val:.2f}x")
    
    if stats_table:
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        print(f"{'Dataset':<12} {'Beta':<8} {'Clients':<10} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Total':<10}")
        print("-" * 100)
        for row in stats_table:
            print(f"{row['Dataset']:<12} {row['Beta']:<8} {row['Clients']:<10} {row['Mean']:<10} "
                  f"{row['Std']:<10} {row['Min']:<8} {row['Max']:<8} {row['Total']:<10}")


def main():
    """Generate all distribution figures."""
    print("Generating dataset distribution figures...")
    print(f"Datasets: {DATASETS}")
    print(f"Beta values: {BETA_VALUES}")
    print(f"Number of clients: {NUM_CLIENTS}\n")
    
    print("Generating individual plots...")
    for dataset in DATASETS:
        for beta in BETA_VALUES:
            sample_counts, data_path, found = get_client_file_sizes(dataset, beta, NUM_CLIENTS)
            
            if found and sample_counts is not None:
                beta_str = f"{beta:.2f}".replace(".", "_")
                filename = f"distribution_{dataset}_b{beta_str}.png"
                save_path = os.path.join(RESULTS_DIR, filename)
                plot_distribution(dataset, beta, sample_counts, save_path)
    
    print("\nGenerating combined grid plot...")
    plot_distribution_subplots(DATASETS, BETA_VALUES)
    
    generate_statistics(DATASETS, BETA_VALUES)
    
    print("\n" + "="*100)
    print("✓ All figures generated successfully!")
    print(f"  Saved to: {RESULTS_DIR}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
