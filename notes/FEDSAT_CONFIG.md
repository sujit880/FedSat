# FedSat Configuration Guide

## Understanding FedSat Components

FedSat is a **two-component** federated learning method that combines:

### 1. Client-Side: Loss Function (CALC or CACS)
- **CALC** (Confusion-Aware Cost-Sensitive with Label Calibration)
- **CACS** (Confusion-Aware Cost-Sensitive)

These loss functions handle class imbalance at the client level during local training.

### 2. Server-Side: Aggregation Strategy (fedsat or fedsatc)
- **fedsat**: Struggle-aware targeted aggregation (works with CALC or CACS)
- **fedsatc**: FedSat with class competence weighting (optimized for CACS)

Both are aggregation methods used with the **FedAvg trainer**.

## Architecture

FedSat uses the **FedAvg trainer** with a custom aggregation method:

```
Trainer: fedavg (base federated averaging trainer)
   ↓
Aggregation: fedsat or fedsatc (defined in aggregator.py)
   ↓
Client Loss: CALC or CACS (confusion-aware losses)
```

## Command Line Usage

### Correct Configuration

FedSat should be run with `--trainer=fedavg`, `--agg=fedsat` (or `fedsatc`), and `--loss=CALC` or `--loss=CACS`:

```bash
# FedSat with CALC loss (recommended)
python main.py \
  --trainer=fedavg \
  --agg=fedsat \
  --loss=CALC \
  --dataset=cifar10 \
  --num_rounds=200 \
  --num_epochs=5 \
  --batch_size=64

# FedSat with CACS loss
python main.py \
  --trainer=fedavg \
  --agg=fedsat \
  --loss=CACS \
  --dataset=cifar10 \
  --num_rounds=200 \
  --num_epochs=5 \
  --batch_size=64

# FedSatC (class competence variant) with CACS
python main.py \
  --trainer=fedavg \
  --agg=fedsatc \
  --loss=CACS \
  --dataset=cifar10 \
  --num_rounds=200 \
  --num_epochs=5 \
  --batch_size=64
```

### What NOT to Do

❌ **Wrong**: Using FedSat aggregation without CALC/CACS loss
```bash
python main.py --trainer=fedavg --agg=fedsat --loss=CE  # Error: requires CALC or CACS!
```

❌ **Wrong**: Using CALC/CACS loss without FedSat aggregation (missing aggregation component)
```bash
python main.py --trainer=fedavg --loss=CALC  # Missing --agg=fedsat!
```

❌ **Wrong**: Using wrong trainer name
```bash
python main.py --trainer=fedsat --loss=CALC  # Wrong! Use --trainer=fedavg --agg=fedsat
```

## YAML Configuration

In `configs/experiments.yaml`, FedSat is properly configured:

```yaml
methods:
  fedsat:
    trainer: fedavg
    category: proposed
    description: "FedSat: Struggle-aware targeted aggregation (requires CALC or CACS loss)"
    hyperparameters:
      agg: fedsat
    recommended_losses: [CALC, CACS]
    
  fedsatc:
    trainer: fedavg
    category: proposed
    description: "FedSatC: FedSat with class competence (requires CACS loss)"
    hyperparameters:
      agg: fedsatc
    recommended_losses: [CACS]
```

### Using in Experiments

When defining experiments, use `method_specific_loss` to ensure FedSat uses the correct loss:

```yaml
experiments:
  my_experiment:
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    non_iid_levels: [moderate]
    seeds: [42]
    method_specific_loss:
      fedavg: CE      # Baseline uses CE
      fedsat: CALC    # FedSat uses CALC
```

Or use the configuration format for ablation studies:

```yaml
experiments:
  ablation:
    datasets: [cifar10]
    non_iid_levels: [severe]
    seeds: [42]
    configurations:
      - {method: fedavg, loss: CE, name: "Baseline"}
      - {method: fedsat, loss: CALC, name: "FedSat-CALC"}
      - {method: fedsatc, loss: CACS, name: "FedSatC-CACS"}
```

## Ablation Study

To understand the contribution of each component:

```yaml
ablation_study:
  configurations:
    - method: fedavg
      loss: CE
      name: "Baseline (standard FedAvg)"
      
    - method: fedavg
      loss: CALC
      name: "CALC loss only (loss contribution)"
      note: "FedAvg trainer with default aggregation but CALC loss"
      
    - method: fedsat
      loss: CE
      name: "FedSat aggregation only"
      note: "FedAvg trainer with FedSat aggregation but CE loss - this will ERROR!"
      
    - method: fedsat
      loss: CALC
      name: "Full FedSat (aggregation + CALC)"
```

**Important Note**: Configuration #3 (FedSat aggregation with CE loss) will **fail** because the code enforces that `fedsat` and `fedsatc` aggregation methods require CALC or CACS loss. This is by design to ensure proper usage.

### Valid Ablation Study

```yaml
ablation_study:
  configurations:
    - {method: fedavg, loss: CE, name: "Baseline"}
    - {method: fedavg, loss: CALC, name: "CALC only"}  
    - {method: fedsat, loss: CALC, name: "Full FedSat"}
```

This helps understand:
- **CALC loss contribution**: Compare rows 1 vs 2
- **Combined effect (synergy)**: Compare row 1 vs 3
- **Is there synergy?**: Check if row 3 > row 2 (suggesting aggregation adds value)

## Running Experiments

```bash
# Quick test with FedSat
python run_yaml_experiments.py quick_test

# Main results (FedSat with CALC)
python run_yaml_experiments.py main_results

# Loss function comparison (test different losses with FedSat)
python run_yaml_experiments.py loss_comparison

# Ablation study
python run_yaml_experiments.py ablation_table
```

## Summary

| Component | Parameter | Values | Description |
|-----------|-----------|--------|-------------|
| **Trainer** | `--trainer` | `fedavg` | Base federated averaging trainer |
| **Aggregation** | `--agg` | `fedsat` or `fedsatc` | Server-side struggle-aware aggregation |
| **Loss Function** | `--loss` | `CALC` or `CACS` | Client-side class-imbalance aware loss |

**Full FedSat** = `--trainer=fedavg` + `--agg=fedsat` + `--loss=CALC` (or `CACS`)

All three components work together to handle data heterogeneity and class imbalance in federated learning.

### Key Difference from Other Methods

Unlike other FL methods where the trainer name determines everything:
- **FedAvg, FedProx, SCAFFOLD**: Use trainer name directly (e.g., `--trainer=fedprox`)
- **FedSat**: Uses `--trainer=fedavg` but specifies custom aggregation via `--agg=fedsat`

This design allows FedAvg trainer to support multiple aggregation strategies (fedavg, fedadam, fedyogi, fedsat, fedsatc, etc.) through the `--agg` parameter.
