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

def parse_experiment_name(exp_name):
    """Parse experiment name to extract method, model, etc."""
    parts = exp_name.split('_')
    method = parts[0]  # e.g., 'fedavg'
    if len(parts) > 1:
        model = parts[1]  # e.g., 'tresnet18p'
    else:
        model = 'unknown'
    return method, model

def main():
    results_dir = '/home/sujit_2021cs35/Github/FedSat/RESULTS/results'
    
    # Define target accuracies for each dataset
    targets = {
        'fmnist': [0.60, 0.80, 0.90],
        'cifar': [0.20, 0.40, 0.60],
        'cifar100': [0.20, 0.30, 0.40]
    }
    
    # Collect data: {(dataset, dataset_setting, method): {accuracy_target: round}}
    convergence_data = defaultdict(lambda: defaultdict(dict))
    
    # Iterate through all experiment directories
    for exp_folder in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
        
        # Extract dataset and beta from folder name
        dataset = None
        beta = None
        dataset_setting = None
        
        # Only process b0_3 (beta=0.3) directories
        if '_b0_3_' not in exp_folder and not exp_folder.endswith('_b0_3'):
            continue
        
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
        
        # Skip unwanted datasets or duplicates
        if dataset in ['femnist', 'emnist']:
            continue  # Only keep the main datasets
        
        # Extract full dataset setting (e.g., "noiid_lbldir_b0_3_k100")
        if dataset:
            # Extract setting part
            start_idx = exp_folder.find(dataset) + len(dataset) + 1
            end_idx = exp_folder.rfind('_Global') if '_Global' in exp_folder else exp_folder.rfind('_Local')
            if end_idx > start_idx:
                dataset_setting = exp_folder[start_idx:end_idx]
        
        # Extract beta value
        if '_b0_3_' in exp_folder:
            beta = '0.3'
        elif exp_folder.endswith('_b0_3'):
            beta = '0.3'
        
        if dataset and beta == '0.3' and dataset_setting:
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
                
                # Extract method name (e.g., 'fedavg' from 'fedavg_lenet5_CE_...')
                method = method_folder.split('_')[0]
                
                # Get convergence rounds
                key = (dataset, dataset_setting, method)
                for target_acc in dataset_targets:
                    round_num = find_convergence_round(test_stats_path, target_acc)
                    convergence_data[key][target_acc] = round_num
                
                print(f"✓ {dataset:12} | {dataset_setting:30} | {method:15} | {test_stats_path}")
    
    # Generate LaTeX table
    generate_latex_table(convergence_data, targets)

def generate_latex_table(convergence_data, targets):
    """Generate LaTeX table from convergence data"""
    
    # Organize data by dataset and setting
    datasets_settings_methods = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for (dataset, dataset_setting, method), acc_rounds in convergence_data.items():
        datasets_settings_methods[dataset][dataset_setting][method] = acc_rounds
    
    # Generate LaTeX for each dataset
    latex_output = []
    
    for dataset in ['fmnist', 'cifar', 'cifar100']:
        if dataset not in datasets_settings_methods:
            continue
        
        settings_data = datasets_settings_methods[dataset]
        target_accs = targets[dataset]
        
        for setting in sorted(settings_data.keys()):
            methods_data = settings_data[setting]
            
            # Create LaTeX table
            latex_output.append(f"\n% Table for {dataset.upper()} - {setting} (β = 0.3)")
            latex_output.append("\\begin{table}[H]")
            latex_output.append("\\centering")
            latex_output.append(f"\\caption{{Convergence rounds for {dataset.upper()} ({setting}, β=0.3)}}")
            
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
            
            # Data rows
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
            latex_output.append("\\label{tab:convergence_" + dataset + "_" + setting.replace("_", "") + "}")
            latex_output.append("\\end{table}")
    
    # Print LaTeX code
    latex_code = "\n".join(latex_output)
    print("\n" + "="*80)
    print("LATEX CODE:")
    print("="*80)
    print(latex_code)
    print("="*80)
    
    # Save to file
    with open('/home/sujit_2021cs35/Github/FedSat/convergence_table.tex', 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX code saved to: convergence_table.tex")

if __name__ == "__main__":
    main()
