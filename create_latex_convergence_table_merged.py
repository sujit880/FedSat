import os
import json
import pandas as pd
from collections import defaultdict

def find_convergence_round(csv_path, target_accuracy):
    """Find the round where global accuracy reaches target accuracy"""
    try:
        df = pd.read_csv(csv_path)
        # Find first round where global_accuracy >= target_accuracy
        matching_rows = df[df['global_accuracy'] >= target_accuracy]
        if not matching_rows.empty:
            return int(matching_rows.iloc[0]['round'])
        return None  # Never reached
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    results_dir = '/home/sujit_2021cs35/Github/FedSat/RESULTS/results'
    
    # Define target accuracies for each dataset
    targets = {
        'fmnist': [0.40, 0.60, 0.80],
        'cifar': [0.20, 0.40, 0.60],
        'cifar100': [0.20, 0.30, 0.40]
    }
    
    # Collect data: {beta: {dataset: {method: {accuracy_target: round}}}}
    convergence_data_b03 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    convergence_data_b01 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Iterate through all experiment directories
    for exp_folder in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
        
        # Only process k100 (100 clients) directories
        if '_k100_' not in exp_folder and not exp_folder.endswith('_k100'):
            continue
        
        # Extract dataset
        dataset = None
        beta = None
        
        if 'cifar100' in exp_folder:
            dataset = 'cifar100'
        elif 'femnist' in exp_folder:
            dataset = 'femnist'
        elif 'fmnist' in exp_folder:
            dataset = 'fmnist'
        elif 'cifar' in exp_folder:
            dataset = 'cifar'
        elif 'emnist' in exp_folder:
            dataset = 'emnist'
        
        # Skip unwanted datasets
        if dataset in ['femnist', 'emnist']:
            continue
        
        # Extract beta value
        if '_b0_3_' in exp_folder:
            beta = '0.3'
            convergence_data = convergence_data_b03[dataset]
        elif '_b0_1_' in exp_folder:
            beta = '0.1'
            convergence_data = convergence_data_b01[dataset]
        else:
            continue
        
        if dataset and beta:
            # Get target accuracies for this dataset
            dataset_targets = targets.get(dataset, [])
            
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
                
                # Get convergence rounds
                for target_acc in dataset_targets:
                    round_num = find_convergence_round(test_stats_path, target_acc)
                    convergence_data[method][target_acc] = round_num
                
                print(f"✓ {beta} | {dataset:12} | {method:20}")
    
    # Generate merged LaTeX tables
    generate_merged_latex_table(convergence_data_b03, targets, "0.3")
    generate_merged_latex_table(convergence_data_b01, targets, "0.1")

def generate_merged_latex_table(convergence_data, targets, beta):
    """Generate merged LaTeX table from convergence data"""
    
    latex_output = []
    
    # Collect all methods across all datasets
    all_methods = set()
    for dataset in convergence_data:
        all_methods.update(convergence_data[dataset].keys())
    all_methods = sorted(all_methods)
    
    # Create merged table
    latex_output.append(f"\n% Merged table for all datasets (k=100, β = {beta})")
    latex_output.append("\\begin{table}[H]")
    latex_output.append("\\centering")
    latex_output.append(f"\\caption{{Convergence comparison across datasets (100 clients, β={beta})}}")
    
    # Determine number of columns
    # Method | FMNIST(40%,60%,80%) | CIFAR(20%,40%,60%) | CIFAR100(20%,30%,40%)
    num_cols = 1 + 3 + 3 + 3  # method + fmnist + cifar + cifar100
    
    latex_output.append("\\begin{tabular}{l" + "r" * (num_cols - 1) + "}")
    latex_output.append("\\toprule")
    
    # Header rows
    latex_output.append("\\multirow{2}{*}{Method} & \\multicolumn{3}{c}{FMNIST} & \\multicolumn{3}{c}{CIFAR} & \\multicolumn{3}{c}{CIFAR100} \\\\")
    latex_output.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
    header = " & 40\\% & 60\\% & 80\\% & 20\\% & 40\\% & 60\\% & 20\\% & 30\\% & 40\\% \\\\"
    latex_output.append(header)
    latex_output.append("\\midrule")
    
    # Data rows
    for method in all_methods:
        row = method
        
        # FMNIST data (40%, 60%, 80%)
        for target_acc in [0.40, 0.60, 0.80]:
            if 'fmnist' in convergence_data and method in convergence_data['fmnist']:
                round_num = convergence_data['fmnist'][method].get(target_acc, None)
            else:
                round_num = None
            
            if round_num is not None:
                row += f" & {round_num}"
            else:
                row += " & --"
        
        # CIFAR data (20%, 40%, 60%)
        for target_acc in [0.20, 0.40, 0.60]:
            if 'cifar' in convergence_data and method in convergence_data['cifar']:
                round_num = convergence_data['cifar'][method].get(target_acc, None)
            else:
                round_num = None
            
            if round_num is not None:
                row += f" & {round_num}"
            else:
                row += " & --"
        
        # CIFAR100 data (20%, 30%, 40%)
        for target_acc in [0.20, 0.30, 0.40]:
            if 'cifar100' in convergence_data and method in convergence_data['cifar100']:
                round_num = convergence_data['cifar100'][method].get(target_acc, None)
            else:
                round_num = None
            
            if round_num is not None:
                row += f" & {round_num}"
            else:
                row += " & --"
        
        row += " \\\\"
        latex_output.append(row)
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\label{tab:convergence_merged_k100_b" + beta.replace(".", "") + "}")
    latex_output.append("\\end{table}")
    
    # Print LaTeX code
    latex_code = "\n".join(latex_output)
    print("\n" + "="*120)
    print(f"MERGED LATEX CODE (β={beta}):")
    print("="*120)
    print(latex_code)
    print("="*120)
    
    # Save to file
    filename = f'/home/sujit_2021cs35/Github/FedSat/convergence_table_merged_k100_b{beta.replace(".", "")}.tex'
    with open(filename, 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX code saved to: convergence_table_merged_k100_b{beta.replace('.', '')}.tex")

if __name__ == "__main__":
    main()
