# Running FedSat Experiments

Complete guide to running experiments using the YAML-based configuration system.

## Quick Start

### Interactive Menu (Easiest)

```bash
./quick_start.sh
```

This will show an interactive menu with common experiment options.

### Direct Commands

```bash
# Quick validation test (recommended first step)
./run_yaml_experiments.sh quick

# Baseline comparison
./run_yaml_experiments.sh baseline

# Adaptive optimization methods
./run_yaml_experiments.sh adaptive

# Class imbalance study
./run_yaml_experiments.sh imbalance

# Ablation study
./run_yaml_experiments.sh ablation

# Full comparison (LONG - 3-7 days!)
./run_yaml_experiments.sh full

# Paper experiments
./run_yaml_experiments.sh paper
```

### Preview Before Running (Dry Run)

```bash
# See what commands would be executed without running them
./run_yaml_experiments.sh quick --dry-run
./run_yaml_experiments.sh baseline --dry-run
```

## Installation

### 1. Install PyYAML

```bash
pip install pyyaml
```

### 2. Make Scripts Executable (if needed)

```bash
chmod +x run_yaml_experiments.sh
chmod +x quick_start.sh
```

### 3. Verify Setup

```bash
./run_yaml_experiments.sh list
```

## Available Experiment Suites

### Standard Experiments

| Suite | Command | Description | Estimated Time |
|-------|---------|-------------|----------------|
| **quick_test** | `./run_yaml_experiments.sh quick` | Fast validation (10 rounds) | ~10 min |
| **baseline_comparison** | `./run_yaml_experiments.sh baseline` | Compare FedAvg, FedProx, SCAFFOLD, MOON | 6-12 hours |
| **adaptive_methods** | `./run_yaml_experiments.sh adaptive` | Compare FedAdagrad, FedYogi, FedAdam | 8-16 hours |
| **class_imbalance** | `./run_yaml_experiments.sh imbalance` | FedRS, FedSAM, FedSat for imbalance | 8-16 hours |
| **ablation_study** | `./run_yaml_experiments.sh ablation` | Ablation on FedSat components | 6-12 hours |
| **full_comparison** | `./run_yaml_experiments.sh full` | All methods, all datasets | 3-7 days |

### Paper Experiments

| Suite | Command | Description |
|-------|---------|-------------|
| **main_results** | `./run_yaml_experiments.sh paper` | Main comparison table |
| **ablation_table** | `./run_yaml_experiments.sh run ablation_table` | Ablation study table |
| **loss_comparison** | `./run_yaml_experiments.sh run loss_comparison` | Compare loss functions |
| **non_iid_robustness** | `./run_yaml_experiments.sh run non_iid_robustness` | Non-IID robustness study |

## Advanced Usage

### Using Python Directly

```bash
# List all configurations
python run_yaml_experiments.py --list

# Run specific suite
python run_yaml_experiments.py quick_test

# Dry run
python run_yaml_experiments.py baseline_comparison --dry-run

# Custom config file
python run_yaml_experiments.py quick_test --config my_config.yaml
```

### Running in Background

For long experiments, use `nohup`:

```bash
# Run in background
nohup ./run_yaml_experiments.sh full > full_exp.log 2>&1 &

# Monitor progress
tail -f full_exp.log

# Check if still running
ps aux | grep run_yaml_experiments
```

### Using Screen or Tmux

```bash
# Start screen session
screen -S fedsat_exp

# Run experiment
./run_yaml_experiments.sh full

# Detach: Press Ctrl+A then D
# Reattach later: screen -r fedsat_exp
```

## Manual Execution

### Single Experiment

```bash
# FedSat with CALC on CIFAR-10
python main.py \
  --trainer=fedavg \
  --agg=fedsat \
  --loss=CALC \
  --dataset=cifar10 \
  --dataset_type=noiid_lbldir_b0_3_k100 \
  --num_rounds=200 \
  --num_epochs=5 \
  --batch_size=64 \
  --learning_rate=0.01 \
  --num_clients=100 \
  --clients_per_round=10 \
  --seed=42
```

### Key Parameters for FedSat

```bash
# CORRECT: FedSat with proper configuration
--trainer=fedavg    # Use FedAvg trainer
--agg=fedsat        # FedSat aggregation method
--loss=CALC         # CALC or CACS loss required

# WRONG: This will ERROR!
--trainer=fedavg --agg=fedsat --loss=CE  # fedsat requires CALC/CACS
```

## Configuration File

Edit `configs/experiments.yaml` to customize experiments:

```yaml
experiments:
  my_custom_test:
    description: "My custom experiment"
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    non_iid_levels: [moderate]
    seeds: [42]
    method_specific_loss:
      fedavg: CE
      fedsat: CALC
```

Then run:
```bash
./run_yaml_experiments.sh run my_custom_test
```

## Workflow Examples

### 1. Initial Testing

```bash
# Step 1: Quick test to verify everything works
./run_yaml_experiments.sh quick

# Step 2: If successful, preview a larger experiment
./run_yaml_experiments.sh baseline --dry-run

# Step 3: Run the experiment
./run_yaml_experiments.sh baseline
```

### 2. Paper Results

```bash
# Generate main results table
nohup ./run_yaml_experiments.sh paper > paper_main.log 2>&1 &

# Generate ablation table
nohup ./run_yaml_experiments.sh run ablation_table > paper_ablation.log 2>&1 &

# Loss function comparison
nohup ./run_yaml_experiments.sh run loss_comparison > paper_loss.log 2>&1 &
```

### 3. Custom Dataset Study

Edit `configs/experiments.yaml`:
```yaml
experiments:
  cifar10_only:
    datasets: [cifar10]
    methods: [fedavg, fedprox, fedyogi, fedsat]
    non_iid_levels: [moderate, severe, extreme]
    seeds: [42, 123, 456]
    method_specific_loss:
      fedavg: CE
      fedprox: CE
      fedyogi: CE
      fedsat: CALC
```

Run:
```bash
./run_yaml_experiments.sh run cifar10_only
```

## Troubleshooting

### Dataset Not Found

```bash
# Generate dataset first
python generate_clients_dataset.py \
  --dataset cifar10 \
  --type noiid_lbldir \
  --clients 100 \
  --beta 0.3
```

### PyYAML Not Installed

```bash
pip install pyyaml
```

### Permission Denied

```bash
chmod +x run_yaml_experiments.sh
chmod +x quick_start.sh
```

### Out of Memory

Edit `configs/experiments.yaml` and reduce:
- `batch_size`: 64 → 32
- `num_epochs`: 5 → 3
- `clients_per_round`: 10 → 5

### Check Experiment Progress

```bash
# If running in background with nohup
tail -f full_exp.log

# Check process status
ps aux | grep python | grep main.py

# Check GPU usage
nvidia-smi
```

## Tips

1. **Always start with quick_test** to validate setup
2. **Use dry-run** to preview commands before long experiments
3. **Run long experiments in background** with nohup or screen
4. **Monitor GPU memory** during experiments
5. **Save experiment logs** for debugging and analysis
6. **Use appropriate seeds** for reproducibility (we use 42, 123, 456)

## Results Location

Results are saved in:
```
RESULTS/
├── json_dump/           # Raw results in JSON format
├── results/             # Processed results
└── figures/             # Generated plots
```

## Getting Help

```bash
# Show all available commands
./run_yaml_experiments.sh help

# List all experiment suites
./run_yaml_experiments.sh list

# Show Python script help
python run_yaml_experiments.py --help
```

## Summary

**Simplest approach:**
```bash
./quick_start.sh
```

**Quick command:**
```bash
./run_yaml_experiments.sh quick
```

**Long experiment (background):**
```bash
nohup ./run_yaml_experiments.sh full > full.log 2>&1 &
tail -f full.log
```
