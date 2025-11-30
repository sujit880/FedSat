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
    
    # Collect data: {(dataset, beta, method): {accuracy_target: round}}
    convergence_data_b03 = defaultdict(lambda: defaultdict(dict))
    convergence_data_b01 = defaultdict(lambda: defaultdict(dict))
    
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
            convergence_data = convergence_data_b03
        elif '_b0_1_' in exp_folder:
            beta = '0.1'
            convergence_data = convergence_data_b01
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
                key = dataset
                for target_acc in dataset_targets:
                    round_num = find_convergence_round(test_stats_path, target_acc)
                    convergence_data[key][method][target_acc] = round_num
                
                print(f"✓ {beta} | {dataset:12} | {method:20}")
    
    # Generate LaTeX tables
    generate_latex_table(convergence_data_b03, targets, "0.3")
    generate_latex_table(convergence_data_b01, targets, "0.1")

def generate_latex_table(convergence_data, targets, beta):
    """Generate LaTeX table from convergence data"""
    
    latex_output = []
    
    for dataset in ['fmnist', 'cifar', 'cifar100']:
        if dataset not in convergence_data:
            continue
        
        methods_data = convergence_data[dataset]
        target_accs = targets[dataset]
        
        # Create LaTeX table
        latex_output.append(f"\n% Table for {dataset.upper()} (k=100, β = {beta})")
        latex_output.append("\\begin{table}[H]")
        latex_output.append("\\centering")
        latex_output.append(f"\\caption{{Convergence rounds for {dataset.upper()} (100 clients, β={beta})}}")
        
        # Create column specification
        num_cols = 1 + len(target_accs)  # Method column + accuracy columns
        latex_output.append("\\begin{tabular}{" + "l" * num_cols + "}")
        latex_output.append("\\toprule")
        
        # Header row
        header = "Method"
        for acc in target_accs:
            header += f" & {int(acc*100)}\\%"
        header += " \\\\"
        latex_output.append(header)
        latex_output.append("\\midrule")
        
        # Data rows - sorted by method name
        for method in sorted(methods_data.keys()):
            acc_rounds = methods_data[method]
            row = method
            for target_acc in target_accs:
                round_num = acc_rounds.get(target_acc, None)
                if round_num is not None:
                    row += f" & {round_num}"
                else:
                    row += " & --"  # Not reached
            row += " \\\\"
            latex_output.append(row)
        
        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\label{tab:convergence_" + dataset + "_k100_b" + beta.replace(".", "") + "}")
        latex_output.append("\\end{table}")
    
    # Print LaTeX code
    latex_code = "\n".join(latex_output)
    print("\n" + "="*80)
    print(f"LATEX CODE (β={beta}):")
    print("="*80)
    print(latex_code)
    print("="*80)
    
    # Save to file
    filename = f'/home/sujit_2021cs35/Github/FedSat/convergence_table_k100_b{beta.replace(".", "")}.tex'
    with open(filename, 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX code saved to: convergence_table_k100_b{beta.replace('.', '')}.tex")

if __name__ == "__main__":
    main()
