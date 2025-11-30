"""
Experiment Configuration File for FedSat Project
This file defines all experimental settings and provides utilities to run experiments systematically.
"""

import os
import subprocess
import itertools
from pathlib import Path
from typing import Dict, List, Any
import json

# Base directory
BASE_DIR = Path(__file__).parent

#############################################
# EXPERIMENT CONFIGURATIONS
#############################################

class ExperimentConfig:
    """Base configuration for experiments"""
    
    # Dataset configurations
    DATASETS = {
        'cifar10': {
            'name': 'cifar10',
            'num_classes': 10,
            'model': 'resnet18',
        },
        'cifar100': {
            'name': 'cifar100',
            'num_classes': 100,
            'model': 'resnet18',
        },
        'fmnist': {
            'name': 'fmnist',
            'num_classes': 10,
            'model': 'lenet5',
        },
        'emnist': {
            'name': 'emnist',
            'num_classes': 47,
            'model': 'lenet5',
        },
    }
    
    # Non-IID configurations
    NON_IID_SETTINGS = {
        'mild': {'beta': 0.5, 'description': 'Mild non-IID'},
        'moderate': {'beta': 0.3, 'description': 'Moderate non-IID'},
        'severe': {'beta': 0.1, 'description': 'Severe non-IID'},
        'extreme': {'beta': 0.05, 'description': 'Extreme non-IID'},
    }
    
    # Federated Learning methods
    METHODS = {
        # Standard baselines
        'fedavg': {
            'trainer': 'fedavg',
            'description': 'Standard FedAvg',
            'category': 'baseline',
        },
        'fedprox': {
            'trainer': 'fedprox',
            'description': 'FedProx with proximal term',
            'category': 'baseline',
            'mu': 0.01,  # proximal term coefficient
        },
        'scaffold': {
            'trainer': 'scaffold',
            'description': 'SCAFFOLD variance reduction',
            'category': 'baseline',
        },
        'moon': {
            'trainer': 'moon',
            'description': 'MOON contrastive learning',
            'category': 'baseline',
        },
        
        # Adaptive optimization methods
        'fedadagrad': {
            'trainer': 'fedadagrad',
            'description': 'Adaptive with Adagrad',
            'category': 'adaptive',
            'server_learning_rate': 0.01,
            'tau': 1e-3,
        },
        'fedyogi': {
            'trainer': 'fedyogi',
            'description': 'Adaptive with Yogi',
            'category': 'adaptive',
            'server_learning_rate': 0.01,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-3,
        },
        'fedadam': {
            'trainer': 'fedadam',
            'description': 'Adaptive with Adam',
            'category': 'adaptive',
            'server_learning_rate': 0.01,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-3,
        },
        
        # Class imbalance methods
        'fedrs': {
            'trainer': 'fedrs',
            'description': 'Restricted softmax for class imbalance',
            'category': 'class_imbalance',
        },
        'fedsam': {
            'trainer': 'fedsam',
            'description': 'Sharpness-aware minimization',
            'category': 'class_imbalance',
            'rho': 0.05,
        },
        
        # Knowledge distillation methods
        'fedntd': {
            'trainer': 'fedntd',
            'description': 'Not-true distillation',
            'category': 'distillation',
            'beta_ntd': 1.0,
            'tau_ntd': 1.0,
        },
        'fedproto': {
            'trainer': 'fedproto',
            'description': 'Prototype-based learning',
            'category': 'distillation',
        },
        
        # Your methods
        'fedsat': {
            'trainer': 'fedsat',
            'description': 'FedSat struggle-aware aggregation',
            'category': 'proposed',
        },
    }
    
    # Loss functions
    LOSSES = {
        'CE': {'name': 'CE', 'description': 'Cross Entropy'},
        'FL': {'name': 'FL', 'description': 'Focal Loss'},
        'CB': {'name': 'CB', 'description': 'Class Balanced Loss'},
        'CALC': {'name': 'CALC', 'description': 'Confusion-Aware Cost-Sensitive with Label Calibration'},
        'CACS': {'name': 'CACS', 'description': 'Confusion-Aware Cost-Sensitive'},
    }
    
    # Default training hyperparameters
    DEFAULT_PARAMS = {
        'num_rounds': 200,
        'clients_per_round': 10,
        'num_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.01,
        'num_clients': 100,
        'gpu': 0,
        'cuda': 'True',
    }
    
    # Seeds for reproducibility
    SEEDS = [42, 123, 456]


#############################################
# EXPERIMENT SUITES
#############################################

class ExperimentSuite:
    """Defines different experiment suites"""
    
    @staticmethod
    def quick_test():
        """Quick test with minimal settings"""
        return {
            'name': 'quick_test',
            'datasets': ['cifar10'],
            'methods': ['fedavg', 'fedyogi', 'fedsat'],
            'non_iid': ['moderate'],
            'losses': ['CE'],
            'seeds': [42],
            'num_rounds': 10,
        }
    
    @staticmethod
    def baseline_comparison():
        """Compare standard baseline methods"""
        return {
            'name': 'baseline_comparison',
            'datasets': ['cifar10', 'cifar100', 'fmnist'],
            'methods': ['fedavg', 'fedprox', 'scaffold', 'moon'],
            'non_iid': ['moderate', 'severe'],
            'losses': ['CE'],
            'seeds': [42, 123],
        }
    
    @staticmethod
    def adaptive_methods():
        """Compare adaptive optimization methods"""
        return {
            'name': 'adaptive_comparison',
            'datasets': ['cifar10', 'cifar100', 'fmnist'],
            'methods': ['fedavg', 'fedadagrad', 'fedyogi', 'fedadam'],
            'non_iid': ['mild', 'moderate', 'severe'],
            'losses': ['CE'],
            'seeds': [42, 123, 456],
        }
    
    @staticmethod
    def class_imbalance_study():
        """Study methods for class imbalance"""
        return {
            'name': 'class_imbalance',
            'datasets': ['cifar10', 'cifar100'],
            'methods': ['fedavg', 'fedrs', 'fedsam', 'fedsat'],
            'non_iid': ['severe', 'extreme'],
            'losses': ['CE', 'FL', 'CB', 'CALC'],
            'seeds': [42, 123, 456],
        }
    
    @staticmethod
    def ablation_study():
        """Ablation study for FedSat + CALC"""
        return {
            'name': 'ablation',
            'datasets': ['cifar10', 'cifar100'],
            'experiments': [
                {'method': 'fedavg', 'loss': 'CE', 'description': 'Baseline'},
                {'method': 'fedavg', 'loss': 'CALC', 'description': 'CALC loss only'},
                {'method': 'fedsat', 'loss': 'CE', 'description': 'FedSat aggregation only'},
                {'method': 'fedsat', 'loss': 'CALC', 'description': 'Full method'},
            ],
            'non_iid': ['moderate', 'severe'],
            'seeds': [42, 123, 456],
        }
    
    @staticmethod
    def full_comparison():
        """Full experimental comparison"""
        return {
            'name': 'full_comparison',
            'datasets': ['cifar10', 'cifar100', 'fmnist'],
            'methods': [
                'fedavg', 'fedyogi', 'fedrs', 'fedsam', 'fedntd', 'fedsat'
            ],
            'non_iid': ['mild', 'moderate', 'severe'],
            'losses': ['CE', 'CALC'],
            'seeds': [42, 123, 456],
        }


#############################################
# EXPERIMENT RUNNER
#############################################

class ExperimentRunner:
    """Runs experiments based on configurations"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results_dir = BASE_DIR / 'RESULTS'
        self.results_dir.mkdir(exist_ok=True)
    
    def build_command(self, params: Dict[str, Any]) -> List[str]:
        """Build command line arguments from parameters"""
        cmd = ['python', 'main.py']
        
        for key, value in params.items():
            if value is not None:
                cmd.append(f'--{key}')
                cmd.append(str(value))
        
        return cmd
    
    def run_single_experiment(self, params: Dict[str, Any], dry_run: bool = False):
        """Run a single experiment"""
        cmd = self.build_command(params)
        
        print(f"\n{'='*60}")
        print(f"Running: {params.get('trainer', 'unknown')} on {params.get('dataset', 'unknown')}")
        print(f"Beta: {params.get('beta', 'N/A')}, Loss: {params.get('loss', 'CE')}, Seed: {params.get('seed', 'N/A')}")
        print(f"{'='*60}")
        
        if dry_run:
            print("Command:", ' '.join(cmd))
            return None
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            return False
    
    def run_suite(self, suite_config: Dict[str, Any], dry_run: bool = False):
        """Run an experiment suite"""
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT SUITE: {suite_config.get('name', 'unnamed')}")
        print(f"{'#'*60}\n")
        
        # Handle different suite types
        if 'experiments' in suite_config:
            # Ablation study format
            experiments = self._generate_ablation_experiments(suite_config)
        else:
            # Standard format
            experiments = self._generate_standard_experiments(suite_config)
        
        total = len(experiments)
        print(f"Total experiments to run: {total}\n")
        
        if not dry_run:
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        
        successful = 0
        for i, params in enumerate(experiments, 1):
            print(f"\n[{i}/{total}]")
            if self.run_single_experiment(params, dry_run):
                successful += 1
        
        print(f"\n{'='*60}")
        print(f"Completed: {successful}/{total} experiments successful")
        print(f"{'='*60}\n")
    
    def _generate_standard_experiments(self, suite_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experiment configurations for standard suite"""
        experiments = []
        
        datasets = suite_config.get('datasets', ['cifar10'])
        methods = suite_config.get('methods', ['fedavg'])
        non_iid_levels = suite_config.get('non_iid', ['moderate'])
        losses = suite_config.get('losses', ['CE'])
        seeds = suite_config.get('seeds', self.config.SEEDS)
        
        for dataset, method, non_iid, loss, seed in itertools.product(
            datasets, methods, non_iid_levels, losses, seeds
        ):
            beta = self.config.NON_IID_SETTINGS[non_iid]['beta']
            dataset_type = f"noiid_lbldir_b{str(beta).replace('.', '_')}_k{self.config.DEFAULT_PARAMS['num_clients']}"
            
            params = {
                'trainer': self.config.METHODS[method]['trainer'],
                'dataset': dataset,
                'dataset_type': dataset_type,
                'loss': loss,
                'seed': seed,
                **self.config.DEFAULT_PARAMS,
            }
            
            # Override num_rounds if specified in suite
            if 'num_rounds' in suite_config:
                params['num_rounds'] = suite_config['num_rounds']
            
            # Add method-specific parameters
            method_params = {k: v for k, v in self.config.METHODS[method].items() 
                           if k not in ['trainer', 'description', 'category']}
            params.update(method_params)
            
            experiments.append(params)
        
        return experiments
    
    def _generate_ablation_experiments(self, suite_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experiment configurations for ablation study"""
        experiments = []
        
        datasets = suite_config.get('datasets', ['cifar10'])
        exp_configs = suite_config.get('experiments', [])
        non_iid_levels = suite_config.get('non_iid', ['moderate'])
        seeds = suite_config.get('seeds', self.config.SEEDS)
        
        for dataset, exp_config, non_iid, seed in itertools.product(
            datasets, exp_configs, non_iid_levels, seeds
        ):
            beta = self.config.NON_IID_SETTINGS[non_iid]['beta']
            dataset_type = f"noiid_lbldir_b{str(beta).replace('.', '_')}_k{self.config.DEFAULT_PARAMS['num_clients']}"
            
            params = {
                'trainer': exp_config['method'],
                'dataset': dataset,
                'dataset_type': dataset_type,
                'loss': exp_config['loss'],
                'seed': seed,
                **self.config.DEFAULT_PARAMS,
            }
            
            experiments.append(params)
        
        return experiments


#############################################
# MAIN INTERFACE
#############################################

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FedSat Experiment Runner')
    parser.add_argument('suite', nargs='?', 
                       choices=['quick', 'baseline', 'adaptive', 'imbalance', 'ablation', 'full'],
                       help='Experiment suite to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--list', action='store_true',
                       help='List available methods and configurations')
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    if args.list:
        print("\n=== Available Methods ===")
        for category in ['baseline', 'adaptive', 'class_imbalance', 'distillation', 'proposed']:
            methods = [m for m, info in config.METHODS.items() if info.get('category') == category]
            if methods:
                print(f"\n{category.upper()}:")
                for m in methods:
                    print(f"  - {m}: {config.METHODS[m]['description']}")
        
        print("\n=== Available Datasets ===")
        for name, info in config.DATASETS.items():
            print(f"  - {name}: {info['num_classes']} classes")
        
        print("\n=== Non-IID Settings ===")
        for name, info in config.NON_IID_SETTINGS.items():
            print(f"  - {name}: beta={info['beta']} ({info['description']})")
        
        return
    
    if not args.suite:
        print("Usage: python experiment_config.py [suite] [--dry-run] [--list]")
        print("\nAvailable suites: quick, baseline, adaptive, imbalance, ablation, full")
        return
    
    # Run the selected suite
    suite_map = {
        'quick': ExperimentSuite.quick_test(),
        'baseline': ExperimentSuite.baseline_comparison(),
        'adaptive': ExperimentSuite.adaptive_methods(),
        'imbalance': ExperimentSuite.class_imbalance_study(),
        'ablation': ExperimentSuite.ablation_study(),
        'full': ExperimentSuite.full_comparison(),
    }
    
    suite = suite_map[args.suite]
    runner.run_suite(suite, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
