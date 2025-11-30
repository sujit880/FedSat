# FedSat Shell Scripts Reference

Quick reference for all available shell scripts to run experiments.

## 📋 Available Scripts

### 1. `quick_start.sh` - Interactive Menu (Easiest) ⭐

**What it does:** Interactive menu for common experiment tasks

**Usage:**
```bash
./quick_start.sh
```

**Features:**
- User-friendly menu interface
- Confirmation prompts for long experiments
- Time estimates for each option
- No need to remember command syntax

---

### 2. `run_yaml_experiments.sh` - Main Experiment Runner

**What it does:** Runs experiments from YAML configuration

**Usage:**
```bash
# Show help
./run_yaml_experiments.sh help

# List all experiments
./run_yaml_experiments.sh list

# Run specific experiment
./run_yaml_experiments.sh <command> [--dry-run] [--config FILE]
```

**Commands:**
| Command | Suite | Description |
|---------|-------|-------------|
| `quick` | quick_test | Fast validation (10 rounds) |
| `baseline` | baseline_comparison | Compare baseline FL methods |
| `adaptive` | adaptive_methods | Adaptive optimization study |
| `imbalance` | class_imbalance | Class imbalance methods |
| `ablation` | ablation_study | FedSat component ablation |
| `full` | full_comparison | Complete evaluation (LONG!) |
| `paper` | main_results | Main results for paper |

**Examples:**
```bash
# Quick test
./run_yaml_experiments.sh quick

# Preview baseline comparison
./run_yaml_experiments.sh baseline --dry-run

# Run ablation study
./run_yaml_experiments.sh ablation

# Run custom suite
./run_yaml_experiments.sh run loss_comparison

# Use custom config
./run_yaml_experiments.sh quick --config my_config.yaml
```

---

### 3. `generate_datasets.sh` - Dataset Generator

**What it does:** Generate non-IID datasets for experiments

**Usage:**
```bash
./generate_datasets.sh
```

**Features:**
- Interactive menu for dataset selection
- Batch generation of all required datasets
- Custom dataset configuration
- Supports CIFAR-10, CIFAR-100, FMNIST, EMNIST

**Common datasets:**
```bash
# Generate all datasets (recommended before running experiments)
./generate_datasets.sh
# Select option 1: "Generate all datasets for experiments"
```

---

## 🚀 Quick Start Workflow

### First Time Setup

```bash
# 1. Make scripts executable
chmod +x *.sh

# 2. Install dependencies
pip install pyyaml

# 3. Generate datasets
./generate_datasets.sh
# Select: "Generate all datasets for experiments"

# 4. Run quick test
./run_yaml_experiments.sh quick
```

### Running Experiments

```bash
# Interactive menu (easiest)
./quick_start.sh

# OR use direct commands
./run_yaml_experiments.sh quick      # Quick test
./run_yaml_experiments.sh baseline   # Baseline comparison
./run_yaml_experiments.sh paper      # Paper results
```

## 📊 Experiment Suites Overview

### Quick Validation
```bash
./run_yaml_experiments.sh quick
```
- **Time:** ~10 minutes
- **Purpose:** Verify setup works
- **Datasets:** CIFAR-10 only
- **Methods:** FedAvg, FedYogi, FedSat

### Baseline Comparison
```bash
./run_yaml_experiments.sh baseline
```
- **Time:** 6-12 hours
- **Purpose:** Compare standard FL methods
- **Datasets:** CIFAR-10, CIFAR-100, FMNIST
- **Methods:** FedAvg, FedProx, SCAFFOLD, MOON

### Ablation Study
```bash
./run_yaml_experiments.sh ablation
```
- **Time:** 6-12 hours
- **Purpose:** Analyze FedSat components
- **Configurations:**
  - Baseline (FedAvg + CE)
  - CALC loss only (FedAvg + CALC)
  - Full FedSat (FedAvg + fedsat agg + CALC)
  - FedSatC variant (FedAvg + fedsatc agg + CACS)

### Full Comparison
```bash
./run_yaml_experiments.sh full
```
- **Time:** 3-7 DAYS
- **Purpose:** Complete experimental evaluation
- **Datasets:** CIFAR-10, CIFAR-100, FMNIST
- **Methods:** All implemented methods
- **Non-IID:** Mild, Moderate, Severe

### Paper Experiments
```bash
# Main results table
./run_yaml_experiments.sh paper

# Ablation table
./run_yaml_experiments.sh run ablation_table

# Loss comparison
./run_yaml_experiments.sh run loss_comparison

# Non-IID robustness
./run_yaml_experiments.sh run non_iid_robustness
```

## 🔧 Advanced Usage

### Dry Run (Preview Commands)

```bash
# See commands without executing
./run_yaml_experiments.sh baseline --dry-run
./run_yaml_experiments.sh full --dry-run
```

### Background Execution

```bash
# Long experiments should run in background
nohup ./run_yaml_experiments.sh full > full_exp.log 2>&1 &

# Monitor progress
tail -f full_exp.log

# Check if running
ps aux | grep run_yaml_experiments
```

### Using Screen/Tmux

```bash
# Start screen session
screen -S fedsat_exp

# Run experiment
./run_yaml_experiments.sh full

# Detach: Ctrl+A then D
# Reattach: screen -r fedsat_exp
```

### Custom Configuration

```bash
# Edit config file
vim configs/experiments.yaml

# Add your custom experiment
experiments:
  my_experiment:
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    # ... more configuration

# Run it
./run_yaml_experiments.sh run my_experiment
```

## 📁 Directory Structure

```
FedSat/
├── quick_start.sh              # Interactive menu (easiest)
├── run_yaml_experiments.sh     # Main experiment runner
├── generate_datasets.sh        # Dataset generator
├── run_yaml_experiments.py     # Python runner (called by .sh)
├── configs/
│   └── experiments.yaml        # Experiment configurations
├── DATA/                       # Generated datasets
│   ├── cifar10/
│   ├── cifar100/
│   └── fmnist/
└── RESULTS/                    # Experiment results
    ├── json_dump/
    ├── results/
    └── figures/
```

## 📝 Common Tasks

### 1. First Time Setup
```bash
./generate_datasets.sh          # Generate datasets
./run_yaml_experiments.sh quick # Validate setup
```

### 2. Run Paper Experiments
```bash
# Main results
nohup ./run_yaml_experiments.sh paper > paper.log 2>&1 &

# Ablation study
nohup ./run_yaml_experiments.sh run ablation_table > ablation.log 2>&1 &
```

### 3. Test New Method
```bash
# Add method to configs/experiments.yaml
# Then run quick test
./run_yaml_experiments.sh quick --dry-run  # Preview
./run_yaml_experiments.sh quick            # Execute
```

### 4. Generate Single Dataset
```bash
python generate_clients_dataset.py \
  --dataset cifar10 \
  --type noiid_lbldir \
  --clients 100 \
  --beta 0.3
```

### 5. Manual Experiment
```bash
python main.py \
  --trainer=fedavg \
  --agg=fedsat \
  --loss=CALC \
  --dataset=cifar10 \
  --dataset_type=noiid_lbldir_b0_3_k100 \
  --num_rounds=200 \
  --num_epochs=5 \
  --batch_size=64 \
  --seed=42
```

## ⚙️ Configuration

### Experiment Configuration (`configs/experiments.yaml`)

```yaml
experiments:
  my_experiment:
    description: "My custom experiment"
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    non_iid_levels: [moderate]
    seeds: [42]
    method_specific_loss:
      fedavg: CE
      fedsat: CALC
```

### Method Configuration

```yaml
methods:
  fedsat:
    trainer: fedavg          # Use FedAvg trainer
    category: proposed
    hyperparameters:
      agg: fedsat           # FedSat aggregation
    recommended_losses: [CALC, CACS]
```

## 🔍 Troubleshooting

### Scripts not executable
```bash
chmod +x *.sh
```

### PyYAML not found
```bash
pip install pyyaml
```

### Dataset not found
```bash
./generate_datasets.sh
# Select option 1 to generate all
```

### Out of memory
```bash
# Edit configs/experiments.yaml
# Reduce: batch_size, num_epochs, clients_per_round
```

### Check experiment status
```bash
# If running in background
tail -f experiment.log

# Check process
ps aux | grep python | grep main.py

# Check GPU
nvidia-smi
```

## 📚 Documentation

- **RUNNING_EXPERIMENTS.md** - Detailed guide to running experiments
- **YAML_CONFIG_GUIDE.md** - Complete YAML configuration guide
- **FEDSAT_CONFIG.md** - FedSat-specific configuration
- **EXPERIMENT_GUIDE.md** - General experiment guide

## 💡 Tips

1. **Always start with quick test** to validate your setup
2. **Use dry-run** to preview commands before long experiments
3. **Generate all datasets first** using `./generate_datasets.sh`
4. **Run long experiments in background** with nohup or screen
5. **Monitor GPU usage** during experiments
6. **Save logs** for debugging and analysis

## 📞 Getting Help

```bash
# Script help
./run_yaml_experiments.sh help

# List experiments
./run_yaml_experiments.sh list

# Python help
python run_yaml_experiments.py --help
```

## Summary

**Simplest workflow:**
```bash
# 1. Generate datasets
./generate_datasets.sh

# 2. Interactive menu
./quick_start.sh

# 3. Select experiment and run
```

**Command-line workflow:**
```bash
# Test
./run_yaml_experiments.sh quick

# Run experiment
./run_yaml_experiments.sh baseline

# Long experiment (background)
nohup ./run_yaml_experiments.sh full > full.log 2>&1 &
```
