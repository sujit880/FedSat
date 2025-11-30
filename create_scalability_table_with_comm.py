"""
Create scalability table with communication cost metrics.
Combines accuracy, convergence rounds, and per-round communication cost.
"""

import os
import json
import pandas as pd
from collections import defaultdict


def get_model_size_mb(dataset, model_name='tresnet18p'):
    """Get the size of the model in megabytes (MB)."""
    model_sizes = {
        'tresnet18p': {
            'cifar': 11.2,  # ~2.9M parameters * 4 bytes
            'cifar10': 11.2,
            'cifar100': 11.5,
            'fmnist': 11.2,
        },
    }
    
    if model_name in model_sizes and dataset in model_sizes[model_name]:
        return model_sizes[model_name][dataset]
    else:
        return 11.2


def compute_communication_cost(model_size_mb, clients_per_round, method='standard'):
    """
    Compute per-round communication cost in MB.
    
    Communication = Upload + Download
    Upload = model_size * clients_per_round
    Download = model_size * clients_per_round
    """
    method_lower = method.lower()
    
    # Ditto maintains both global and personalized models
    if method_lower == 'ditto':
        upload_cost = model_size_mb * clients_per_round * 2
        download_cost = model_size_mb * clients_per_round
        total_cost = upload_cost + download_cost
    else:
        # Standard FL: upload model updates, download global model
        upload_cost = model_size_mb * clients_per_round
        download_cost = model_size_mb * clients_per_round
        total_cost = upload_cost + download_cost
    
    return total_cost


def get_max_accuracy(csv_path):
    """Find the maximum accuracy achieved and the round it was reached"""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None
        max_idx = df['global_accuracy'].idxmax()
        max_acc = df.loc[max_idx, 'global_accuracy']
        max_round = int(df.loc[max_idx, 'round'])
        return max_acc, max_round
    except Exception as e:
        return None, None


def collect_scalability_data_with_comm(results_dir, dataset='cifar', model_name='tresnet18p'):
    """
    Collect scalability data including accuracy, rounds, and communication cost.
    
    Returns:
        Dictionary: {beta: {num_clients: {method: (accuracy, round, comm_cost)}}}
    """
    model_size_mb = get_model_size_mb(dataset, model_name)
    print(f"Model size for {model_name} on {dataset}: {model_size_mb:.4f} MB\n")
    
    data_b03 = defaultdict(lambda: defaultdict(dict))
    data_b01 = defaultdict(lambda: defaultdict(dict))
    
    # Iterate through all experiment directories
    for exp_folder in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
        
        # Skip if not noiid_lbldir setting
        if 'noiid_lbldir' not in exp_folder:
            continue
        
        # Extract dataset
        exp_dataset = None
        if 'cifar100' in exp_folder:
            exp_dataset = 'cifar100'
        elif 'fmnist' in exp_folder:
            exp_dataset = 'fmnist'
        elif 'cifar' in exp_folder and 'cifar100' not in exp_folder:
            exp_dataset = 'cifar'
        
        # Skip if not target dataset
        if exp_dataset != dataset:
            continue
        
        # Extract beta value
        beta = None
        if '_b0_3_' in exp_folder:
            beta = '0.3'
            data = data_b03
        elif '_b0_1_' in exp_folder:
            beta = '0.1'
            data = data_b01
        else:
            continue
        
        # Extract number of clients
        import re
        match = re.search(r'_k(\d+)_', exp_folder)
        if not match:
            continue
        num_clients = int(match.group(1))
        
        # Process each method
        for method_folder in sorted(os.listdir(exp_path)):
            method_path = os.path.join(exp_path, method_folder)
            if not os.path.isdir(method_path):
                continue
            
            test_stats_path = os.path.join(method_path, 'test_stats.csv')
            metadata_path = os.path.join(method_path, 'metadata.json')
            
            if not os.path.exists(test_stats_path) or not os.path.exists(metadata_path):
                continue
            
            # Get accuracy and round
            max_acc, max_round = get_max_accuracy(test_stats_path)
            if max_acc is None or max_round is None:
                continue
            
            # Load metadata for clients_per_round
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            clients_per_round = metadata.get('clients_per_round', num_clients // 2)
            
            # Extract method name
            method = method_folder.split('_')[0]
            
            # Handle fedavg variants (FedSat)
            if method == 'fedavg':
                if 'CACS' in method_folder or 'CALC' in method_folder:
                    if '_A_fedsat' in method_folder or '_A_fedsatc' in method_folder:
                        method = 'fedsat'
            
            # Compute communication cost
            comm_cost = compute_communication_cost(model_size_mb, clients_per_round, method)
            
            # Store results
            data[num_clients][method] = (max_acc, max_round, comm_cost)
            
            print(f"✓ β={beta} | k={num_clients:3} | {method:12} | Acc: {max_acc:5.1%} @ Round {max_round:3} | Comm: {comm_cost:6.1f} MB/round")
    
    return data_b03, data_b01, model_size_mb


def generate_latex_table_with_comm(data, beta, dataset='cifar'):
    """Generate LaTeX table with accuracy, rounds, and communication cost."""
    
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{Scalability analysis on {dataset.upper()}-10 ($\\beta = {beta}$) with per-round communication cost. FedSat achieves superior accuracy with competitive communication overhead.}}")
    latex_lines.append(f"\\label{{tab:scalability_{dataset}_b{beta.replace('.', '')}_comm}}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{")
    latex_lines.append("\\begin{tabular}{lccccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multirow{2}{*}{Method} & \\multicolumn{3}{c}{$k=20$} & \\multicolumn{3}{c}{$k=50$} & \\multicolumn{3}{c}{$k=100$} \\\\")
    latex_lines.append("\\cmidrule(lr){2-10}")
    latex_lines.append(" & Acc. & Round & Comm. & Acc. & Round & Comm. & Acc. & Round & Comm. \\\\")
    latex_lines.append(" & (\\%) & & (MB) & (\\%) & & (MB) & (\\%) & & (MB) \\\\")
    latex_lines.append("\\midrule")
    
    # Get all methods
    all_methods = set()
    for num_clients in data:
        all_methods.update(data[num_clients].keys())
    
    # Specific methods from your table in order
    method_order = ['ccvr', 'ditto', 'fedavg', 'fedfa', 'fedprox', 'fedsam', 'moon', 'fedsat']
    
    # Filter to only methods we have data for and sort
    methods_to_show = [m for m in method_order if m in all_methods]
    
    for method in methods_to_show:
        is_fedsat = (method == 'fedsat')
        method_display = '\\textbf{fedsat (ours)}' if is_fedsat else method
        
        row = f"{method_display}"
        
        for k in [20, 50, 100]:
            if k in data and method in data[k]:
                acc, rounds, comm = data[k][method]
                acc_pct = acc * 100  # Convert to percentage
                
                if is_fedsat:
                    row += f" & \\textbf{{{acc_pct:.1f}}} & \\textbf{{{rounds}}} & \\textbf{{{comm:.1f}}}"
                else:
                    row += f" & {acc_pct:.1f} & {rounds} & {comm:.1f}"
            else:
                row += " & -- & -- & --"
        
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    """Main function."""
    results_dir = '/home/sujit_2021cs35/Github/FedSat/RESULTS/results'
    
    print("="*80)
    print("SCALABILITY ANALYSIS WITH COMMUNICATION COST")
    print("="*80)
    print()
    
    # Collect data for CIFAR-10
    data_b03, data_b01, model_size = collect_scalability_data_with_comm(
        results_dir, 
        dataset='cifar', 
        model_name='tresnet18p'
    )
    
    print("\n" + "="*80)
    print("GENERATING LATEX TABLES")
    print("="*80)
    
    # Generate LaTeX tables
    for beta, data in [('0.3', data_b03), ('0.1', data_b01)]:
        if not data:
            continue
        
        latex_code = generate_latex_table_with_comm(data, beta, 'cifar')
        
        print(f"\n% LaTeX Table for β={beta}")
        print(latex_code)
        
        # Save to file
        filename = f'scalability_table_b{beta.replace(".", "")}_with_comm.tex'
        with open(filename, 'w') as f:
            f.write(latex_code)
        print(f"\nSaved to: {filename}")
    
    # Save raw data to JSON
    output_data = {
        'model_size_mb': model_size,
        'beta_0.3': dict(data_b03),
        'beta_0.1': dict(data_b01)
    }
    
    with open('scalability_with_comm.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\n✓ Raw data saved to: scalability_with_comm.json")


if __name__ == '__main__':
    main()
