"""
YAML-based Experiment Runner for FedSat Project
Reads configurations from YAML files and executes experiments systematically.
"""

import yaml
import subprocess
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from collections import defaultdict


class YAMLExperimentRunner:
    """Run experiments based on YAML configuration"""
    
    def __init__(self, config_path: str = "configs/experiments.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.global_settings = self.config.get('global', {})
        self.datasets = self.config.get('datasets', {})
        self.methods = self.config.get('methods', {})
        self.losses = self.config.get('losses', {})
        self.non_iid = self.config.get('non_iid', {})
        self.experiments = self.config.get('experiments', {})
        self.paper_experiments = self.config.get('paper_experiments', {})
    
    def load_config(self) -> Dict:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build_command(self, params: Dict[str, Any]) -> List[str]:
        """Build command line arguments from parameters"""
        cmd = ['python', 'main.py']
        
        for key, value in params.items():
            if value is None or value == '':
                continue  # Skip None or empty values
            
            # Handle boolean flags - only add flag if True, omit if False
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    cmd.append(f'--{key}')
                # If False, don't add anything (let argparse use default)
            else:
                # For non-boolean values, add key-value pair
                cmd.append(f'--{key}')
                cmd.append(str(value))
        
        return cmd
    
    def run_single_experiment(self, params: Dict[str, Any], dry_run: bool = False) -> bool:
        """Run a single experiment"""
        cmd = self.build_command(params)
        
        print(f"\n{'='*70}")
        print(f"Method: {params.get('trainer', 'N/A')}")
        print(f"Dataset: {params.get('dataset', 'N/A')}")
        print(f"Non-IID: {params.get('non_iid_level', 'N/A')} (beta={params.get('beta', 'N/A')})")
        print(f"Loss: {params.get('loss', 'CE')}")
        print(f"Seed: {params.get('seed', 'N/A')}")
        print(f"Rounds: {params.get('num_rounds', 'N/A')}")
        print(f"{'='*70}")
        
        if dry_run:
            print("Command:", ' '.join(cmd))
            print()
            return True
        
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return False
    
    def run_experiment_suite(self, suite_name: str, dry_run: bool = False):
        """Run a predefined experiment suite"""
        if suite_name in self.experiments:
            suite_config = self.experiments[suite_name]
        elif suite_name in self.paper_experiments:
            suite_config = self.paper_experiments[suite_name]
        else:
            raise ValueError(f"Unknown experiment suite: {suite_name}")
        
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT SUITE: {suite_name}")
        print(f"# {suite_config.get('description', 'No description')}")
        print(f"{'#'*70}\n")
        
        # Generate experiment configurations
        if 'configurations' in suite_config:
            # Ablation study format
            experiments = self._generate_ablation_experiments(suite_config)
        elif 'grid' in suite_config:
            # Hyperparameter search format
            experiments = self._generate_grid_search_experiments(suite_config)
        elif 'client_configs' in suite_config:
            # Scalability study format
            experiments = self._generate_scalability_experiments(suite_config)
        elif 'local_epoch_configs' in suite_config:
            # Communication study format
            experiments = self._generate_communication_experiments(suite_config)
        else:
            # Standard format
            experiments = self._generate_standard_experiments(suite_config)
        
        total = len(experiments)
        print(f"📊 Total experiments to run: {total}\n")
        
        if not dry_run and total > 10:
            response = input("❓ Continue? (y/n): ")
            if response.lower() != 'y':
                print("❌ Aborted.")
                return
        
        successful = 0
        failed = 0
        
        for i, params in enumerate(experiments, 1):
            print(f"\n[{i}/{total}]")
            if self.run_single_experiment(params, dry_run):
                successful += 1
            else:
                failed += 1
                if not dry_run:
                    response = input("\n⚠️  Continue after error? (y/n): ")
                    if response.lower() != 'y':
                        break
        
        print(f"\n{'='*70}")
        print(f"✅ Completed: {successful}/{total} experiments successful")
        if failed > 0:
            print(f"❌ Failed: {failed}/{total} experiments")
        print(f"{'='*70}\n")
    
    def _generate_standard_experiments(self, suite_config: Dict) -> List[Dict[str, Any]]:
        """Generate standard experiment configurations"""
        experiments = []
        
        datasets = suite_config.get('datasets', [])
        methods = suite_config.get('methods', [])
        non_iid_levels = suite_config.get('non_iid_levels', ['moderate'])
        losses = suite_config.get('losses', ['CE'])
        seeds = suite_config.get('seeds', self.global_settings['seeds'])
        
        # Get method-specific losses if specified
        method_specific_loss = suite_config.get('method_specific_loss', {})
        
        for dataset, method, non_iid_level, seed in itertools.product(
            datasets, methods, non_iid_levels, seeds
        ):
            # Determine loss function
            if method in method_specific_loss:
                loss_list = [method_specific_loss[method]]
            else:
                loss_list = losses
            
            for loss in loss_list:
                params = self._build_params(
                    dataset, method, non_iid_level, loss, seed, suite_config
                )
                experiments.append(params)
        
        return experiments
    
    def _generate_ablation_experiments(self, suite_config: Dict) -> List[Dict[str, Any]]:
        """Generate ablation study experiments"""
        experiments = []
        
        datasets = suite_config.get('datasets', [])
        non_iid_levels = suite_config.get('non_iid_levels', ['moderate'])
        seeds = suite_config.get('seeds', self.global_settings['seeds'])
        configurations = suite_config.get('configurations', [])
        
        for dataset, config, non_iid_level, seed in itertools.product(
            datasets, configurations, non_iid_levels, seeds
        ):
            params = self._build_params(
                dataset, config['method'], non_iid_level, config['loss'], seed, suite_config
            )
            params['experiment_name'] = config.get('name', config['method'])
            experiments.append(params)
        
        return experiments
    
    def _generate_grid_search_experiments(self, suite_config: Dict) -> List[Dict[str, Any]]:
        """Generate hyperparameter grid search experiments"""
        experiments = []
        
        datasets = suite_config.get('datasets', [])
        non_iid_levels = suite_config.get('non_iid_levels', ['moderate'])
        seeds = suite_config.get('seeds', self.global_settings['seeds'])
        grid = suite_config.get('grid', {})
        
        for method, param_grid in grid.items():
            # Get all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for dataset, non_iid_level, seed, param_combo in itertools.product(
                datasets, non_iid_levels, seeds, itertools.product(*param_values)
            ):
                params = self._build_params(
                    dataset, method, non_iid_level, 'CE', seed, suite_config
                )
                
                # Add hyperparameters
                for param_name, param_value in zip(param_names, param_combo):
                    params[param_name] = param_value
                
                experiments.append(params)
        
        return experiments
    
    def _generate_scalability_experiments(self, suite_config: Dict) -> List[Dict[str, Any]]:
        """Generate scalability study experiments"""
        experiments = []
        
        datasets = suite_config.get('datasets', [])
        methods = suite_config.get('methods', [])
        non_iid_levels = suite_config.get('non_iid_levels', ['moderate'])
        losses = suite_config.get('losses', ['CE'])
        seeds = suite_config.get('seeds', self.global_settings['seeds'])
        client_configs = suite_config.get('client_configs', [])
        
        for dataset, method, non_iid_level, loss, seed, client_config in itertools.product(
            datasets, methods, non_iid_levels, losses, seeds, client_configs
        ):
            params = self._build_params(
                dataset, method, non_iid_level, loss, seed, suite_config
            )
            params['num_clients'] = client_config['num_clients']
            params['clients_per_round'] = client_config['clients_per_round']
            
            # Note: dataset_type should be specified in overrides if using different num_clients
            # The dataset must already exist in DATA/{dataset}/ directory
            
            experiments.append(params)
        
        return experiments
    
    def _generate_communication_experiments(self, suite_config: Dict) -> List[Dict[str, Any]]:
        """Generate communication efficiency experiments"""
        experiments = []
        
        datasets = suite_config.get('datasets', [])
        methods = suite_config.get('methods', [])
        non_iid_levels = suite_config.get('non_iid_levels', ['moderate'])
        losses = suite_config.get('losses', ['CE'])
        seeds = suite_config.get('seeds', self.global_settings['seeds'])
        local_epoch_configs = suite_config.get('local_epoch_configs', [])
        
        for dataset, method, non_iid_level, loss, seed, epoch_config in itertools.product(
            datasets, methods, non_iid_levels, losses, seeds, local_epoch_configs
        ):
            params = self._build_params(
                dataset, method, non_iid_level, loss, seed, suite_config
            )
            params['num_epochs'] = epoch_config['num_epochs']
            params['num_rounds'] = epoch_config['num_rounds']
            
            experiments.append(params)
        
        return experiments
    
    def _build_params(self, dataset: str, method: str, non_iid_level: str, 
                     loss: str, seed: int, suite_config: Dict) -> Dict[str, Any]:
        """Build parameter dictionary for an experiment"""
        # Start with global defaults
        params = self.global_settings['defaults'].copy()
        
        # Add basic parameters
        params['trainer'] = self.methods[method]['trainer']
        params['dataset'] = self.datasets[dataset]['name']  # Get actual dataset name
        params['loss'] = loss
        params['seed'] = seed
        
        # Only add GPU/CUDA if explicitly set to non-default values
        gpu = self.global_settings.get('gpu')
        # gpu default is True, so only add if explicitly set to False
        if gpu is not None and gpu is False:
            params['gpu'] = gpu
        
        cuda = self.global_settings.get('cuda')
        # cuda default is -1, so only add if explicitly set to something else
        if cuda is not None and cuda != -1:
            params['cuda'] = cuda
        
        # Handle dataset_type: only add if explicitly specified in config
        # Otherwise, main.py will use its default or handle it
        dataset_config = self.datasets.get(dataset, {})
        if 'dataset_type' in dataset_config:
            params['dataset_type'] = dataset_config['dataset_type']
        
        # Add beta value (used by some methods and for dataset generation)
        params['beta'] = self.non_iid[non_iid_level]['beta']
        
        # Add method-specific hyperparameters (only valid main.py parameters)
        method_config = self.methods.get(method, {})
        if 'hyperparameters' in method_config:
            # Only add hyperparameters that are valid command-line arguments
            # Valid params: agg, mu, model, etc. from config_main.py
            for key, value in method_config['hyperparameters'].items():
                if key in ['agg', 'mu', 'model']:  # Only known valid params
                    params[key] = value
        
        # Apply suite-specific overrides (can override dataset_type here too)
        if 'overrides' in suite_config:
            params.update(suite_config['overrides'])
        
        return params
    
    def list_configurations(self):
        """List all available configurations"""
        print("\n" + "="*70)
        print("AVAILABLE CONFIGURATIONS")
        print("="*70)
        
        print("\n📦 DATASETS:")
        for name, config in self.datasets.items():
            print(f"  • {name}: {config['num_classes']} classes, model={config['model']}")
        
        print("\n📊 NON-IID LEVELS:")
        for name, config in self.non_iid.items():
            print(f"  • {name}: β={config['beta']} - {config['description']}")
        
        print("\n🔧 METHODS:")
        for category in ['baseline', 'adaptive', 'class_imbalance', 'distillation', 'proposed']:
            methods = [(m, info) for m, info in self.methods.items() 
                      if info.get('category') == category]
            if methods:
                print(f"\n  {category.upper()}:")
                for method, info in methods:
                    print(f"    • {method}: {info['description']}")
        
        print("\n📝 LOSS FUNCTIONS:")
        for name, config in self.losses.items():
            print(f"  • {name}: {config['description']}")
        
        print("\n🧪 EXPERIMENT SUITES:")
        for name, config in self.experiments.items():
            print(f"  • {name}: {config.get('description', 'No description')}")
        
        print("\n📄 PAPER EXPERIMENTS:")
        for name, config in self.paper_experiments.items():
            print(f"  • {name}: {config.get('description', 'No description')}")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='YAML-based Experiment Runner for FedSat',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all configurations
  python run_yaml_experiments.py --list
  
  # Run quick test
  python run_yaml_experiments.py quick_test
  
  # Dry run to see commands
  python run_yaml_experiments.py full_comparison --dry-run
  
  # Run paper experiments
  python run_yaml_experiments.py main_results
        """
    )
    
    parser.add_argument(
        'suite',
        nargs='?',
        help='Experiment suite to run (use --list to see available suites)'
    )
    parser.add_argument(
        '--config',
        default='configs/experiments.yaml',
        help='Path to YAML configuration file (default: configs/experiments.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available configurations and experiment suites'
    )
    
    args = parser.parse_args()
    
    try:
        runner = YAMLExperimentRunner(args.config)
        
        if args.list:
            runner.list_configurations()
            return
        
        if not args.suite:
            parser.print_help()
            print("\n💡 Tip: Use --list to see available experiment suites")
            return
        
        runner.run_experiment_suite(args.suite, dry_run=args.dry_run)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the config file exists at: configs/experiments.yaml")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
