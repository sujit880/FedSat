# Experiment Setup Corrections - Summary

## What Was Wrong

The original implementation had a conceptual issue with `dataset_type`:

### Previous Implementation ❌
```python
def generate_dataset_type(self, dataset: str, non_iid_level: str, num_clients: int) -> str:
    """Generate dataset type string"""
    beta = self.non_iid[non_iid_level]['beta']
    beta_str = str(beta).replace('.', '_')
    return f"noiid_lbldir_b{beta_str}_k{num_clients}"
```

**The Problem:**
- `dataset_type` is **already a command-line argument** in `config_main.py`
- Auto-generating it limits flexibility
- Ignores pre-existing special datasets (e.g., `qty_lbl_imb_*`)
- Users couldn't specify dataset_type directly

## What Was Fixed ✅

### 1. Updated `_build_params()` Method

**Before:**
```python
# Always auto-generated
params['dataset_type'] = self.generate_dataset_type(
    dataset, non_iid_level, params['num_clients']
)
```

**After:**
```python
# Handle dataset_type with priority order:
# 1. Check suite overrides (highest priority)
# 2. Check dataset config
# 3. Auto-generate as fallback
if 'dataset_type' not in params:
    dataset_config = self.datasets.get(dataset, {})
    if 'dataset_type' in dataset_config:
        params['dataset_type'] = dataset_config['dataset_type']
    else:
        # Auto-generate from non_iid_level
        params['dataset_type'] = self.generate_dataset_type(
            dataset, non_iid_level, params['num_clients']
        )
```

### 2. Updated YAML Configuration

Added documentation and examples:

```yaml
#############################################
# DATASETS
#############################################
# Note: dataset_type can be optionally specified per dataset.
# If not specified, it will be auto-generated from non_iid_level and num_clients.

datasets:
  cifar10:
    name: cifar
    num_classes: 10
    model: tresnet18p
    # Optional: Specify dataset_type explicitly
    # dataset_type: "noiid_lbldir_b0_3_k100"
```

### 3. Added Override Examples

```yaml
experiments:
  # Example: Using specific pre-generated dataset
  quantity_label_imbalance:
    description: "Test with quantity + label imbalance dataset"
    datasets: [cifar10]
    methods: [fedavg, fedsat]
    non_iid_levels: [moderate]
    seeds: [42]
    overrides:
      dataset_type: "qty_lbl_imb_b0_3_k100"  # Explicitly specify
      num_clients: 100
```

### 4. Created Documentation

- `DATASET_TYPE_GUIDE.md` - Comprehensive guide on dataset_type usage

## How It Works Now

### Priority Order for dataset_type

1. **Experiment suite override** (highest priority)
   ```yaml
   overrides:
     dataset_type: "qty_lbl_imb_b0_3_k100"
   ```

2. **Dataset configuration**
   ```yaml
   datasets:
     cifar10:
       dataset_type: "noiid_lbldir_b0_3_k100"
   ```

3. **Auto-generation** (fallback)
   ```python
   # Generated from: non_iid_level + num_clients
   "noiid_lbldir_b{beta}_k{num_clients}"
   ```

### Benefits ✅

1. **Flexible:** Can specify any dataset_type
2. **Backward compatible:** Still auto-generates if not specified
3. **Explicit:** Makes it clear which dataset is being used
4. **Supports special datasets:** Can use `qty_lbl_imb_*` and other variants
5. **Follows config_main.py:** Respects existing argument structure

## Usage Examples

### Example 1: Auto-Generation (Default Behavior)

```bash
bash run_experiments.sh --set baseline_comparison
```

Uses auto-generated `dataset_type` based on `non_iid_level` and `num_clients`.

### Example 2: Explicit Dataset Type

```yaml
my_experiment:
  datasets: [cifar10]
  methods: [fedsat]
  non_iid_levels: [moderate]
  overrides:
    dataset_type: "qty_lbl_imb_b0_3_k100"
    num_clients: 100
```

### Example 3: Command Line

```bash
python main.py \
  --trainer fedavg \
  --dataset cifar \
  --dataset_type "noiid_lbldir_b0_3_k100" \
  --num_clients 100
```

## Files Modified

1. ✅ `run_yaml_experiments.py`
   - Updated `_build_params()` method
   - Added priority-based dataset_type resolution
   - Improved comments

2. ✅ `configs/experiments.yaml`
   - Added documentation comments
   - Added example experiment with explicit dataset_type
   - Added usage notes

3. ✅ `DATASET_TYPE_GUIDE.md` (NEW)
   - Comprehensive guide on dataset_type
   - Examples and best practices
   - Troubleshooting section

## Verification

Check available datasets:
```bash
# CIFAR datasets
ls DATA/cifar/

# Output:
noiid_lbldir_b0_05_k100/
noiid_lbldir_b0_1_k100/
noiid_lbldir_b0_3_k100/
qty_lbl_imb_b0_3_k100/      # ← Now accessible via overrides!
```

Test the corrected setup:
```bash
# List available experiments
bash run_experiments.sh --list

# Dry run to verify commands
bash run_experiments.sh --set quick_test --dry-run
```

## Summary

**Before:** `dataset_type` was always auto-generated, limiting flexibility

**After:** `dataset_type` can be:
- ✅ Explicitly specified in YAML (recommended)
- ✅ Set per dataset
- ✅ Overridden per experiment
- ✅ Auto-generated as fallback (backward compatible)

This aligns with the design in `config_main.py` where `dataset_type` is a command-line argument that users should be able to control.
