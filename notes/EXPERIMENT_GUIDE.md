# Experiment Configuration Guide

This directory contains three experiment configuration systems for running systematic FL experiments.

## Quick Start

### Option 1: YAML Configuration (Recommended) ⭐

```bash
# List all available configurations
python run_yaml_experiments.py --list

# Run experiment suites
python run_yaml_experiments.py quick_test              # Quick test (10 rounds)
python run_yaml_experiments.py baseline_comparison     # Compare baseline methods
python run_yaml_experiments.py adaptive_methods        # Compare adaptive optimization
python run_yaml_experiments.py class_imbalance         # Class imbalance study
python run_yaml_experiments.py ablation_study          # Ablation study
python run_yaml_experiments.py full_comparison         # Full comparison

# Run paper experiments
python run_yaml_experiments.py main_results            # Main results table
python run_yaml_experiments.py ablation_table          # Ablation study table
python run_yaml_experiments.py loss_comparison         # Loss function comparison
python run_yaml_experiments.py non_iid_robustness      # Non-IID robustness study

# Dry run (see commands without executing)
python run_yaml_experiments.py quick_test --dry-run

# Use custom config file
python run_yaml_experiments.py quick_test --config my_config.yaml
```

**Why YAML?**
- ✅ Clean, readable configuration format
- ✅ Easy to modify experimental parameters
- ✅ Hierarchical structure for complex experiments
- ✅ Supports all experiment types (grid search, ablation, scalability)
- ✅ Version control friendly

### Option 2: Bash Script (Simple)

```bash
# Interactive menu
./run_experiments.sh

# Or run specific modes directly
./run_experiments.sh quick          # Quick test
./run_experiments.sh baseline       # Baseline comparison
./run_experiments.sh adaptive       # Adaptive methods
./run_experiments.sh fedsat         # FedSat with different losses
./run_experiments.sh ablation       # Ablation study
./run_experiments.sh full           # Full experimental suite
```

### Option 3: Python Script (Advanced)

```bash
# List available configurations
python experiment_config.py --list

# Run experiment suites
python experiment_config.py quick          # Quick test (10 rounds)
python experiment_config.py baseline       # Compare baseline methods
python experiment_config.py adaptive       # Compare adaptive optimization
python experiment_config.py imbalance      # Class imbalance study
python experiment_config.py ablation       # Ablation study
python experiment_config.py full           # Full comparison

# Dry run (see commands without executing)
python experiment_config.py quick --dry-run
```

## Experiment Suites

### 1. Quick Test
- **Purpose**: Fast validation that everything works
- **Duration**: ~10 minutes
- **Configuration**:
  - Dataset: CIFAR-10 only
  - Methods: FedAvg, FedYogi, FedSat
  - Rounds: 10
  - Seeds: 1

### 2. Baseline Comparison
- **Purpose**: Compare standard FL methods
- **Methods**: FedAvg, FedProx, SCAFFOLD, MOON
- **Datasets**: CIFAR-10, CIFAR-100, FMNIST
- **Non-IID levels**: Moderate (β=0.3), Severe (β=0.1)
- **Seeds**: 2

### 3. Adaptive Methods
- **Purpose**: Evaluate adaptive server-side optimization
- **Methods**: FedAvg, FedAdagrad, FedYogi, FedAdam
- **Datasets**: CIFAR-10, CIFAR-100, FMNIST
- **Non-IID levels**: Mild (β=0.5), Moderate (β=0.3), Severe (β=0.1)
- **Seeds**: 3

### 4. Class Imbalance Study
- **Purpose**: Test methods for class-imbalanced scenarios
- **Methods**: FedAvg, FedRS, FedSAM, FedSat
- **Datasets**: CIFAR-10, CIFAR-100
- **Non-IID levels**: Severe (β=0.1), Extreme (β=0.05)
- **Losses**: CE, FL, CB, CALC
- **Seeds**: 3

### 5. Ablation Study
- **Purpose**: Analyze contribution of each component
- **Experiments**:
  1. FedAvg + CE (baseline)
  2. FedAvg + CALC (loss contribution)
  3. FedSat + CE (aggregation contribution)
  4. FedSat + CALC (full method)
- **Datasets**: CIFAR-10, CIFAR-100
- **Seeds**: 3

### 6. Full Comparison
- **Purpose**: Comprehensive comparison of all methods
- **Methods**: FedAvg, FedYogi, FedRS, FedSAM, FedNTD, FedSat
- **Datasets**: CIFAR-10, CIFAR-100, FMNIST
- **Non-IID levels**: Mild, Moderate, Severe
- **Losses**: CE, CALC
- **Seeds**: 3
- **Total experiments**: 162

## Configuration Files

### `run_experiments.sh`
- Bash-based experiment runner
- Interactive menu
- Good for simple batch runs
- Easy to customize for specific needs

### `experiment_config.py`
- Python-based experiment configuration
- Programmatic control
- Better for complex experimental designs
- Generates all parameter combinations automatically

## Customizing Experiments

### Modify Bash Script

Edit `run_experiments.sh`:

```bash
# Change datasets
DATASETS=("cifar10" "mnist")

# Change methods to test
ALL_METHODS=("fedavg" "fedyogi" "fedsat")

# Change training parameters
NUM_ROUNDS=100
BATCH_SIZE=64
```

### Modify Python Config

Edit `experiment_config.py`:

```python
# Create custom suite
@staticmethod
def my_custom_suite():
    return {
        'name': 'custom',
        'datasets': ['cifar10'],
        'methods': ['fedavg', 'fedyogi', 'fedsat'],
        'non_iid': ['severe'],
        'losses': ['CE', 'CALC'],
        'seeds': [42],
        'num_rounds': 50,  # Custom rounds
    }
```

## Output Structure

Results are saved to:
```
RESULTS/
├── results/
│   ├── cifar10_noiid_lbldir_b0_3_k100_Global/
│   │   ├── fedavg_resnet18_CE_lr_0.01_...
│   │   ├── fedyogi_resnet18_CE_lr_0.01_...
│   │   └── fedsat_resnet18_CALC_lr_0.01_...
│   └── ...
└── json_dump/
```

## Monitoring Experiments

### Check Progress
```bash
# Watch running experiment
tail -f RESULTS/results/<experiment_name>/log.txt

# Count completed experiments
ls RESULTS/results/**/test_stats.csv | wc -l
```

### Stop/Resume
```bash
# Stop: Ctrl+C in terminal
# Resume: Re-run same command - skips completed experiments (if configured)
```

## Tips for Large Experiments

1. **Start with quick test**
   ```bash
   ./run_experiments.sh quick
   ```

2. **Use dry-run to verify**
   ```bash
   python experiment_config.py full --dry-run
   ```

3. **Run overnight**
   ```bash
   nohup ./run_experiments.sh full > experiment.log 2>&1 &
   ```

4. **Use tmux/screen for persistence**
   ```bash
   tmux new -s experiments
   ./run_experiments.sh full
   # Detach: Ctrl+b, then d
   # Reattach: tmux attach -t experiments
   ```

5. **Generate datasets first**
   ```bash
   ./run_experiments.sh generate
   ```

## Estimated Runtime

Based on CIFAR-10, 200 rounds, 10 epochs:
- **Single experiment**: ~30-60 minutes
- **Quick test**: ~10 minutes
- **Baseline comparison**: ~6-12 hours
- **Full comparison**: ~80-160 hours (3-7 days)

Tip: Use GPU and parallel runs if available!

## YAML Configuration Details

### Structure of `configs/experiments.yaml`

```yaml
global:           # Global settings (GPU, seeds, defaults)
datasets:         # Dataset configurations (cifar10, cifar100, etc.)
non_iid:          # Non-IID levels (iid, mild, moderate, strong, extreme)
methods:          # FL methods with hyperparameters
losses:           # Loss functions (CE, FL, CB, CALC, CACS)
experiments:      # Standard experiment suites
paper_experiments: # Paper-specific experiments
```

### Adding a New Experiment Suite

Edit `configs/experiments.yaml`:

```yaml
experiments:
  my_custom_experiment:
    description: "Custom experiment for testing"
    datasets: [cifar10]
    methods: [fedavg, fedyogi, fedsat]
    non_iid_levels: [moderate]
    losses: [CE]
    seeds: [1, 2, 3]
    overrides:
      num_rounds: 100
      batch_size: 64
```

Then run:
```bash
python run_yaml_experiments.py my_custom_experiment
```

### Adding a New Method

Add method to YAML:

```yaml
methods:
  my_method:
    category: baseline
    trainer: mymethod
    description: "My custom federated learning method"
    hyperparameters:
      learning_rate: 0.01
      mu: 0.1
```

Ensure the trainer is implemented in:
- `flearn/clients/mymethod.py`
- `flearn/trainers/mymethod.py`

### Grid Search Example

```yaml
experiments:
  lr_search:
    description: "Learning rate grid search"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    seeds: [1]
    grid:
      fedavg:
        learning_rate: [0.001, 0.01, 0.1]
        batch_size: [16, 32, 64]
```

### Ablation Study Example

```yaml
experiments:
  loss_ablation:
    description: "Ablation on loss functions"
    datasets: [cifar10]
    non_iid_levels: [moderate]
    seeds: [1, 2, 3]
    configurations:
      - {name: "Baseline", method: fedsat, loss: CE}
      - {name: "Focal", method: fedsat, loss: FL}
      - {name: "ClassBalance", method: fedsat, loss: CB}
      - {name: "CALC", method: fedsat, loss: CALC}
      - {name: "CACS", method: fedsat, loss: CACS}
```

### Checking Configuration

```bash
# List all available components
python run_yaml_experiments.py --list

# Preview commands without running
python run_yaml_experiments.py my_experiment --dry-run

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/experiments.yaml'))"
```

## Troubleshooting

### Dataset not found
```bash
# Generate datasets first
python generate_clients_dataset.py --dataset cifar10 --dataset_type noiid --beta 0.3 --num_clients 100
```

### Memory errors
- Reduce `batch_size`
- Reduce `num_epochs`
- Use smaller model

### Method not found
```bash
# Check available methods
python run_yaml_experiments.py --list
```

### YAML parsing errors
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/experiments.yaml'))"
```

## Example Workflow

### Using YAML (Recommended)

```bash
# 1. List all configurations
python run_yaml_experiments.py --list

# 2. Quick test
python run_yaml_experiments.py quick_test

# 3. Dry run to preview
python run_yaml_experiments.py baseline_comparison --dry-run

# 4. Run ablation study
python run_yaml_experiments.py ablation_study

# 5. Full comparison (overnight)
nohup python run_yaml_experiments.py full_comparison > full_exp.log 2>&1 &

# 6. Check progress
tail -f full_exp.log
```

### Using Python Script

```bash
# 1. List configurations
python experiment_config.py --list

# 2. Quick test
python experiment_config.py quick

# 3. If successful, run ablation study
python experiment_config.py ablation

# 4. Full comparison (overnight)
nohup python experiment_config.py full > full_exp.log 2>&1 &

# 5. Check progress
tail -f full_exp.log
```
