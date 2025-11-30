# Dataset Type Configuration Guide

## Overview

`dataset_type` is a **command-line argument** (defined in `flearn/config/config_main.py`) that specifies which pre-generated dataset partition to use. It is NOT automatically generated - you should specify it explicitly.

## Understanding dataset_type

The `dataset_type` parameter tells the system which data partition to load from the `DATA/{dataset}/` directory.

### Available Dataset Types

Check your `DATA/` directory for available datasets:

```bash
ls DATA/cifar/
```

Example output:
```
noiid_lbldir_b0_05_k100/    # Non-IID, beta=0.05, 100 clients
noiid_lbldir_b0_1_k100/     # Non-IID, beta=0.1, 100 clients
noiid_lbldir_b0_3_k100/     # Non-IID, beta=0.3, 100 clients
noiid_lbldir_b0_1_k20/      # Non-IID, beta=0.1, 20 clients
noiid_lbldir_b0_3_k50/      # Non-IID, beta=0.3, 50 clients
qty_lbl_imb_b0_3_k100/      # Quantity + Label imbalance
qty_lbl_imb_b0_3_k100_fc_2/ # Quantity + Label imbalance, 2 feature classes
qty_lbl_imb_b0_3_k100_fc_4/ # Quantity + Label imbalance, 4 feature classes
```

### Dataset Type Naming Convention

Standard non-IID datasets follow this pattern:
```
noiid_lbldir_b{beta}_k{num_clients}
```

Where:
- `beta` = Dirichlet distribution parameter (0.05, 0.1, 0.3, 0.5, etc.)
  - Lower beta → more non-IID
  - Higher beta → more IID-like
- `num_clients` = Number of clients (20, 50, 100, etc.)

Special dataset types:
- `qty_lbl_imb_*` - Combined quantity and label imbalance
- `iid` - Independent and Identically Distributed data

## Three Ways to Specify dataset_type

### 1. Direct Command Line (Recommended for manual runs)

```bash
python main.py \
  --trainer fedavg \
  --dataset cifar \
  --dataset_type "noiid_lbldir_b0_3_k100" \
  --num_clients 100 \
  --num_rounds 100
```

### 2. YAML Configuration - Per Dataset

In `configs/experiments.yaml`, specify in dataset config:

```yaml
datasets:
  cifar10:
    name: cifar
    num_classes: 10
    model: tresnet18p
    dataset_type: "noiid_lbldir_b0_3_k100"  # Fixed dataset type for all experiments
```

### 3. YAML Configuration - Per Experiment Suite

Override for specific experiment suites:

```yaml
experiments:
  my_experiment:
    description: "Test with specific dataset"
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    non_iid_levels: [moderate]
    seeds: [42]
    overrides:
      dataset_type: "qty_lbl_imb_b0_3_k100"  # Use specific pre-generated dataset
      num_clients: 100
```

## Auto-Generation (Convenience Feature)

The YAML experiment runner includes a **convenience function** that auto-generates `dataset_type` if not specified:

```python
def generate_dataset_type(dataset, non_iid_level, num_clients):
    beta = non_iid[non_iid_level]['beta']
    return f"noiid_lbldir_b{beta}_k{num_clients}"
```

**Priority order:**
1. Experiment suite `overrides.dataset_type` (highest priority)
2. Dataset config `dataset_type`
3. Auto-generated from `non_iid_level` + `num_clients` (fallback)

## Examples

### Example 1: Quick Test with Auto-Generation

```yaml
quick_test:
  datasets: [cifar10]
  methods: [fedavg]
  non_iid_levels: [moderate]  # beta=0.3
  seeds: [42]
  # Will auto-generate: dataset_type="noiid_lbldir_b0_3_k100"
  # (using default num_clients=100)
```

### Example 2: Explicit Dataset Type

```yaml
quantity_imbalance_test:
  datasets: [cifar10]
  methods: [fedsat]
  non_iid_levels: [moderate]  # Not used for dataset_type
  seeds: [42]
  overrides:
    dataset_type: "qty_lbl_imb_b0_3_k100"  # Explicit override
    num_clients: 100
```

### Example 3: Different Client Counts

```yaml
scalability_study:
  datasets: [cifar10]
  methods: [fedavg]
  non_iid_levels: [moderate]
  seeds: [42]
  client_configs:
    - num_clients: 20
      clients_per_round: 5
      # Will generate: dataset_type="noiid_lbldir_b0_3_k20"
    
    - num_clients: 100
      clients_per_round: 10
      # Will generate: dataset_type="noiid_lbldir_b0_3_k100"
```

## Checking Available Datasets

Before running experiments, verify the dataset exists:

```bash
# List available datasets for CIFAR
ls -la DATA/cifar/

# List available datasets for all datasets
find DATA/ -maxdepth 2 -type d -name "*noiid*" -o -name "*qty*"
```

## Common Issues

### Issue 1: Dataset Not Found

```
Error: Dataset directory not found: DATA/cifar/noiid_lbldir_b0_3_k100
```

**Solution:** Generate the dataset first:
```bash
python generate_clients_dataset.py \
  --dataset cifar \
  --dataset_type noiid \
  --beta 0.3 \
  --num_clients 100
```

### Issue 2: Mismatched num_clients

```
Warning: num_clients=50 but dataset_type="noiid_lbldir_b0_3_k100"
```

**Solution:** Ensure consistency:
```yaml
overrides:
  dataset_type: "noiid_lbldir_b0_3_k50"  # Match num_clients
  num_clients: 50
```

### Issue 3: Beta Mismatch

Using `non_iid_levels: [severe]` (beta=0.1) but dataset has beta=0.3.

**Solution:** Either:
- Generate dataset with correct beta, OR
- Explicitly specify existing dataset_type in overrides

## Best Practices

1. **List datasets first:** Always check what's available before configuring
2. **Be explicit:** Specify `dataset_type` in overrides for clarity
3. **Match parameters:** Ensure `num_clients` matches the dataset
4. **Document experiments:** Note which dataset_type was used in results
5. **Pre-generate datasets:** Create all needed datasets before batch experiments

## Summary

| Approach | When to Use | Example |
|----------|-------------|---------|
| Auto-generation | Quick tests, standard patterns | `non_iid_levels: [moderate]` |
| Dataset config | All experiments use same dataset | `datasets.cifar10.dataset_type` |
| Experiment override | Specific datasets per experiment | `overrides.dataset_type` |
| Command line | Manual single runs | `--dataset_type "..."` |

**Remember:** `dataset_type` is just a string argument - it should match an existing directory in `DATA/{dataset}/`.
