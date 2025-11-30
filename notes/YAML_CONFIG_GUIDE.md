# YAML Configuration Guide for FedSat Experiments

Complete guide for using the YAML-based experiment configuration system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration Structure](#configuration-structure)
3. [Running Experiments](#running-experiments)
4. [Available Experiment Suites](#available-experiment-suites)
5. [Customizing Experiments](#customizing-experiments)
6. [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# Install PyYAML if not already installed
pip install pyyaml

# List all available configurations
python run_yaml_experiments.py --list

# Run a quick test
python run_yaml_experiments.py quick_test

# Preview commands without executing
python run_yaml_experiments.py baseline_comparison --dry-run

# Run paper experiments
python run_yaml_experiments.py main_results
```

## Configuration Structure

The configuration file `configs/experiments.yaml` is organized hierarchically:

```yaml
global:              # Global settings
  gpu: 0            # GPU device ID
  cuda: true        # Enable CUDA
  seeds: [1, 2, 3]  # Random seeds for reproducibility
  defaults:         # Default parameters for all experiments
    num_rounds: 200
    num_epochs: 10
    batch_size: 32
    learning_rate: 0.01
    num_clients: 100
    clients_per_round: 10

datasets:            # Dataset definitions
  cifar10:
    num_classes: 10
    model: cnn
    
non_iid:             # Non-IID configurations
  moderate:
    beta: 0.3
    description: "Moderate heterogeneity"

methods:             # FL methods
  fedavg:
    category: baseline
    trainer: fedavg
    description: "Federated Averaging (McMahan et al., 2017)"
    
  fedyogi:
    category: adaptive
    trainer: fedyogi
    description: "FedYogi optimizer (Reddi et al., 2021)"
    hyperparameters:
      server_learning_rate: 0.01
      tau: 0.001
      beta_1: 0.9
      beta_2: 0.99

losses:              # Loss functions
  CE:
    description: "Cross-Entropy Loss"
  CALC:
    description: "Class-Aware Loss Correction"

experiments:         # Experiment suites
  quick_test:
    description: "Quick validation test"
    datasets: [cifar10]
    methods: [fedavg, fedyogi, fedsat]
    non_iid_levels: [moderate]
    losses: [CE]
    seeds: [1]
    overrides:
      num_rounds: 10
```

## Running Experiments

### Basic Usage

```bash
# Run a predefined experiment suite
python run_yaml_experiments.py <suite_name>

# Examples:
python run_yaml_experiments.py quick_test
python run_yaml_experiments.py baseline_comparison
python run_yaml_experiments.py adaptive_methods
```

### Available Options

```bash
# Specify custom config file
python run_yaml_experiments.py quick_test --config my_config.yaml

# Dry run (show commands without executing)
python run_yaml_experiments.py baseline_comparison --dry-run

# List all available configurations
python run_yaml_experiments.py --list
```

## Available Experiment Suites

### Standard Experiments

1. **quick_test** - Fast validation (10 rounds, 1 seed)
   ```bash
   python run_yaml_experiments.py quick_test
   ```

2. **baseline_comparison** - Compare all baseline FL methods
   ```bash
   python run_yaml_experiments.py baseline_comparison
   ```

3. **adaptive_methods** - Compare adaptive optimization methods
   ```bash
   python run_yaml_experiments.py adaptive_methods
   ```

4. **class_imbalance** - Evaluate methods for class imbalance
   ```bash
   python run_yaml_experiments.py class_imbalance
   ```

5. **ablation_study** - Ablation on FedSat components
   ```bash
   python run_yaml_experiments.py ablation_study
   ```

6. **full_comparison** - Complete experimental evaluation
   ```bash
   python run_yaml_experiments.py full_comparison
   ```

7. **hyperparameter_search** - Grid search for optimal hyperparameters
   ```bash
   python run_yaml_experiments.py hyperparameter_search
   ```

8. **scalability** - Scalability study with varying client numbers
   ```bash
   python run_yaml_experiments.py scalability
   ```

### Paper-Specific Experiments

1. **main_results** - Main results table for paper
   ```bash
   python run_yaml_experiments.py main_results
   ```

2. **ablation_table** - Ablation study table
   ```bash
   python run_yaml_experiments.py ablation_table
   ```

3. **loss_comparison** - Compare different loss functions
   ```bash
   python run_yaml_experiments.py loss_comparison
   ```

4. **non_iid_robustness** - Robustness to non-IID levels
   ```bash
   python run_yaml_experiments.py non_iid_robustness
   ```

## Customizing Experiments

### 1. Adding a New Experiment Suite

Edit `configs/experiments.yaml` and add a new entry under `experiments`:

```yaml
experiments:
  my_experiment:
    description: "My custom experiment"
    datasets: [cifar10, cifar100]
    methods: [fedavg, fedyogi, fedsat]
    non_iid_levels: [mild, moderate]
    losses: [CE, FL]
    seeds: [1, 2, 3]
    overrides:
      num_rounds: 100
      batch_size: 64
      learning_rate: 0.01
```

Run with:
```bash
python run_yaml_experiments.py my_experiment
```

### 2. Adding a New FL Method

First, implement the method in code:
- `flearn/clients/mymethod.py` - Client implementation
- `flearn/trainers/mymethod.py` - Server implementation

Then add to YAML configuration:

```yaml
methods:
  mymethod:
    category: baseline  # or: adaptive, class_imbalance, distillation, proposed
    trainer: mymethod
    description: "My custom FL method"
    hyperparameters:
      learning_rate: 0.01
      momentum: 0.9
      custom_param: 0.5
```

### 3. Grid Search Configuration

For hyperparameter tuning:

```yaml
experiments:
  lr_search:
    description: "Learning rate grid search"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    seeds: [1]
    grid:
      fedavg:
        learning_rate: [0.001, 0.01, 0.1, 0.5]
        batch_size: [16, 32, 64]
      fedyogi:
        server_learning_rate: [0.001, 0.01, 0.1]
        tau: [0.0001, 0.001, 0.01]
```

This will run all combinations: 4 × 3 × 3 × 3 = 108 experiments for FedAvg and FedYogi.

### 4. Ablation Study Configuration

For ablation studies with specific configurations:

```yaml
experiments:
  component_ablation:
    description: "Ablate FedSat components"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    seeds: [1, 2, 3]
    configurations:
      - {name: "No SA, No CALC", method: fedavg, loss: CE}
      - {name: "SA only", method: fedrs, loss: CE}
      - {name: "CALC only", method: fedavg, loss: CALC}
      - {name: "SA + CALC (Full)", method: fedsat, loss: CALC}
```

### 5. Scalability Study Configuration

For varying client numbers:

```yaml
experiments:
  client_scaling:
    description: "Scalability with different client counts"
    datasets: [cifar10]
    methods: [fedavg, fedyogi, fedsat]
    non_iid_levels: [moderate]
    losses: [CE]
    seeds: [1, 2, 3]
    client_configs:
      - {num_clients: 20, clients_per_round: 5}
      - {num_clients: 50, clients_per_round: 10}
      - {num_clients: 100, clients_per_round: 10}
      - {num_clients: 200, clients_per_round: 20}
```

## Advanced Usage

### Method-Specific Loss Functions

Some methods require specific loss functions:

```yaml
experiments:
  mixed_losses:
    description: "Different methods with their optimal losses"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    methods: [fedavg, fedrs, fedsat]
    seeds: [1, 2, 3]
    method_specific_loss:
      fedavg: CE
      fedrs: CE
      fedsat: CALC
```

### Communication Efficiency Study

Vary local epochs vs communication rounds:

```yaml
experiments:
  communication_study:
    description: "Trade-off between local epochs and rounds"
    datasets: [cifar10]
    methods: [fedavg, fedyogi]
    non_iid_levels: [moderate]
    losses: [CE]
    seeds: [1, 2, 3]
    local_epoch_configs:
      - {num_epochs: 1, num_rounds: 1000}   # More communication
      - {num_epochs: 5, num_rounds: 200}    # Balanced
      - {num_epochs: 10, num_rounds: 100}   # Less communication
      - {num_epochs: 20, num_rounds: 50}    # Minimal communication
```

### Dataset-Specific Configurations

Define dataset-specific settings:

```yaml
datasets:
  cifar10:
    num_classes: 10
    model: cnn
    
  mnist:
    num_classes: 10
    model: simple_cnn
    
  cifar100:
    num_classes: 100
    model: resnet18
```

### Non-IID Levels

Define different heterogeneity levels:

```yaml
non_iid:
  iid:
    beta: 1000  # Effectively IID
    description: "IID distribution"
    
  mild:
    beta: 0.5
    description: "Mild heterogeneity"
    
  moderate:
    beta: 0.3
    description: "Moderate heterogeneity"
    
  strong:
    beta: 0.1
    description: "Strong heterogeneity"
    
  extreme:
    beta: 0.05
    description: "Extreme heterogeneity"
```

## Best Practices

### 1. Start Small
Always test with `quick_test` before running full experiments:
```bash
python run_yaml_experiments.py quick_test
```

### 2. Use Dry Run
Preview commands before execution:
```bash
python run_yaml_experiments.py my_experiment --dry-run
```

### 3. Progressive Experimentation
```bash
# 1. Quick validation
python run_yaml_experiments.py quick_test

# 2. Single dataset baseline
python run_yaml_experiments.py baseline_comparison --dry-run
# Edit YAML to use only cifar10, then run

# 3. Full evaluation
python run_yaml_experiments.py full_comparison
```

### 4. Version Control
Always commit your YAML configuration:
```bash
git add configs/experiments.yaml
git commit -m "Add experiment configuration for XYZ study"
```

### 5. Long-Running Experiments
Use `nohup` for overnight experiments:
```bash
nohup python run_yaml_experiments.py full_comparison > full_exp.log 2>&1 &

# Monitor progress
tail -f full_exp.log
```

### 6. Reproducibility
- Always specify `seeds` in configuration
- Document hyperparameters in YAML
- Keep track of experiment suite names

### 7. Organization
Structure your configurations:
```yaml
experiments:
  # Development experiments
  dev_quick_test: ...
  dev_ablation: ...
  
  # Production experiments
  paper_main_results: ...
  paper_ablation: ...
  
  # Exploratory experiments
  explore_hyperparams: ...
  explore_architectures: ...
```

## Validation

### Check Configuration Validity

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/experiments.yaml'))"

# List all configurations (checks for errors)
python run_yaml_experiments.py --list

# Dry run to validate parameters
python run_yaml_experiments.py quick_test --dry-run
```

### Common Errors

**1. YAML Syntax Error**
```
yaml.scanner.ScannerError: while scanning a simple key
```
Solution: Check indentation, quotes, and colons.

**2. Unknown Experiment Suite**
```
ValueError: Unknown experiment suite: my_experimen
```
Solution: Check spelling, use `--list` to see available suites.

**3. Method Not Found**
```
KeyError: 'mymethod'
```
Solution: Ensure method is defined in `methods` section of YAML.

**4. Dataset Not Generated**
```
FileNotFoundError: Dataset directory not found
```
Solution: Generate dataset first:
```bash
python generate_clients_dataset.py --dataset cifar10 --beta 0.3
```

## Examples

### Example 1: Quick Method Comparison

```yaml
experiments:
  compare_three:
    description: "Compare three methods quickly"
    datasets: [cifar10]
    methods: [fedavg, fedyogi, fedsat]
    non_iid_levels: [moderate]
    losses: [CE]
    seeds: [1]
    overrides:
      num_rounds: 50
      num_epochs: 5
```

Run:
```bash
python run_yaml_experiments.py compare_three
```

### Example 2: Comprehensive Non-IID Study

```yaml
experiments:
  non_iid_comprehensive:
    description: "Study all non-IID levels"
    datasets: [cifar10, cifar100]
    methods: [fedavg, fedprox, fedyogi, fedsat]
    non_iid_levels: [iid, mild, moderate, strong, extreme]
    losses: [CE]
    seeds: [1, 2, 3, 4, 5]
    overrides:
      num_rounds: 200
```

### Example 3: Loss Function Ablation

```yaml
experiments:
  loss_ablation:
    description: "Ablation on loss functions with FedSat"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    seeds: [1, 2, 3]
    configurations:
      - {name: "CE", method: fedsat, loss: CE}
      - {name: "Focal", method: fedsat, loss: FL}
      - {name: "ClassBalance", method: fedsat, loss: CB}
      - {name: "CALC", method: fedsat, loss: CALC}
      - {name: "CACS", method: fedsat, loss: CACS}
```

## Tips

1. **Parallel Execution**: Run multiple experiments on different GPUs:
   ```bash
   # Terminal 1 (GPU 0)
   python run_yaml_experiments.py baseline_comparison --config gpu0_config.yaml
   
   # Terminal 2 (GPU 1)
   python run_yaml_experiments.py adaptive_methods --config gpu1_config.yaml
   ```

2. **Experiment Naming**: Use descriptive names:
   ```yaml
   experiments:
     paper_table1_main_results: ...
     paper_figure2_convergence: ...
     supp_ablation_components: ...
   ```

3. **Documentation**: Add descriptions to all experiments:
   ```yaml
   experiments:
     my_experiment:
       description: "Evaluates XYZ under ABC conditions for paper Section 4.3"
   ```

4. **Iterative Development**: Start with small seeds, expand later:
   ```yaml
   # Development
   seeds: [1]
   
   # Production
   seeds: [1, 2, 3, 4, 5]
   ```

## Summary

The YAML configuration system provides:
- ✅ **Readability**: Clean, hierarchical structure
- ✅ **Flexibility**: Supports all experiment types
- ✅ **Reproducibility**: Version-controlled configurations
- ✅ **Scalability**: Easy to add new experiments
- ✅ **Maintainability**: Single source of truth for all experiments

For more information:
- See `configs/experiments.yaml` for complete configuration example
- See `EXPERIMENT_GUIDE.md` for detailed experiment documentation
- See `baseline_papers/` for method-specific details
