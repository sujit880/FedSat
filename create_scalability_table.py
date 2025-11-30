import os
import pandas as pd
from collections import defaultdict

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

def main():
    results_dir = '/home/sujit_2021cs35/Github/FedSat/RESULTS/results'
    
    # Collect data: {beta: {num_clients: {dataset: {method: (max_accuracy, round)}}}}
    convergence_data_b03 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    convergence_data_b01 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Iterate through all experiment directories
    for exp_folder in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
        
        # Skip if not noiid_lbldir setting
        if 'noiid_lbldir' not in exp_folder:
            continue
        
        # Extract dataset
        dataset = None
        beta = None
        num_clients = None
        
        if 'cifar100' in exp_folder:
            dataset = 'cifar100'
        elif 'fmnist' in exp_folder:
            dataset = 'fmnist'
        elif 'cifar' in exp_folder:
            dataset = 'cifar'
        
        # Skip unwanted datasets
        if dataset in ['femnist', 'emnist'] or dataset is None:
            continue
        
        # Extract beta value
        if '_b0_3_' in exp_folder:
            beta = '0.3'
            convergence_data = convergence_data_b03
        elif '_b0_1_' in exp_folder:
            beta = '0.1'
            convergence_data = convergence_data_b01
        else:
            continue
        
        # Extract number of clients
        import re
        match = re.search(r'_k(\d+)_', exp_folder)
        if match:
            num_clients = int(match.group(1))
        
        if dataset and beta and num_clients:
            # Iterate through method folders
            for method_folder in sorted(os.listdir(exp_path)):
                method_path = os.path.join(exp_path, method_folder)
                if not os.path.isdir(method_path):
                    continue
                
                test_stats_path = os.path.join(method_path, 'test_stats.csv')
                if not os.path.exists(test_stats_path):
                    continue
                
                # Extract method name and distinguish fedavg variants
                method = method_folder.split('_')[0]
                
                # For fedavg, identify the loss function and if it's FedSat
                if method == 'fedavg':
                    if 'CACS' in method_folder or 'CALC' in method_folder:
                        # Check if it's FedSat variant
                        if '_A_fedsat' in method_folder or '_A_fedsatc' in method_folder:
                            method = 'FedSat'
                        else:
                            # CACS or CALC variants
                            if 'CACS' in method_folder:
                                method = 'FedAvg+CACS'
                            else:
                                method = 'FedAvg+CALC'
                    elif '_CE_' in method_folder or '_CE()_' in method_folder:
                        method = 'FedAvg+CE'
                
                # Get max accuracy and round
                max_acc, max_round = get_max_accuracy(test_stats_path)
                if max_acc is not None and max_round is not None:
                    convergence_data[num_clients][dataset][method] = (max_acc, max_round)
                    print(f"✓ {beta} | {dataset:12} | {num_clients:3} clients | {method:20} | Acc: {max_acc:.1%} @ Round {max_round}")
    
    # Generate LaTeX tables
    generate_scalability_table(convergence_data_b03, "0.3")
    generate_scalability_table(convergence_data_b01, "0.1")

def generate_scalability_table(convergence_data, beta):
    """Generate scalability LaTeX table showing max accuracy and round achieved"""
    
    latex_output = []
    
    # Sort client counts
    client_counts = sorted(convergence_data.keys())
    
    # Create LaTeX table for each dataset
    for dataset in ['fmnist', 'cifar', 'cifar100']:
        if not any(dataset in convergence_data[k] for k in convergence_data):
            continue
        
        latex_output.append(f"\n% Scalability table for {dataset.upper()} (β = {beta})")
        latex_output.append("\\begin{table}[H]")
        latex_output.append("\\centering")
        latex_output.append(f"\\caption{{Scalability analysis: Maximum accuracy achieved with varying number of clients ({dataset.upper()}, β={beta})}}")
        
        # Create column structure: Method | k=20 (Acc% @ Round) | k=50 | k=100
        latex_output.append("\\begin{tabular}{l" + "cc" * len(client_counts) + "}")
        latex_output.append("\\toprule")
        
        # First header row
        header1 = "\\multirow{2}{*}{Method}"
        for k in client_counts:
            header1 += f" & \\multicolumn{{2}}{{c}}{{$k={k}$}}"
        header1 += " \\\\"
        latex_output.append(header1)
        
        # Second header row
        latex_output.append("\\cmidrule(lr){2-" + str(2 + 2*len(client_counts)-1) + "}")
        header2 = ""
        for k in client_counts:
            header2 += " & Acc. & Round"
        header2 += " \\\\"
        latex_output.append(header2)
        latex_output.append("\\midrule")
        
        # Data rows - only include methods that have data for this dataset
        dataset_methods = set()
        for num_clients in convergence_data:
            if dataset in convergence_data[num_clients]:
                dataset_methods.update(convergence_data[num_clients][dataset].keys())
        
        for method in sorted(dataset_methods):
            row = method
            for k in client_counts:
                if dataset in convergence_data[k] and method in convergence_data[k][dataset]:
                    max_acc, max_round = convergence_data[k][dataset][method]
                    row += f" & {max_acc:.1%} & {max_round}"
                else:
                    row += " & -- & --"
            row += " \\\\"
            latex_output.append(row)
        
        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append(f"\\label{{tab:scalability_{dataset}_b{beta.replace('.', '')}}}")
        latex_output.append("\\end{table}")
    
    # Print LaTeX code
    latex_code = "\n".join(latex_output)
    print("\n" + "="*130)
    print(f"SCALABILITY LATEX CODE (β={beta}):")
    print("="*130)
    print(latex_code)
    print("="*130)
    
    # Save to file
    filename = f'/home/sujit_2021cs35/Github/FedSat/scalability_table_b{beta.replace(".", "")}.tex'
    with open(filename, 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX code saved to: scalability_table_b{beta.replace('.', '')}.tex")

if __name__ == "__main__":
    main()
