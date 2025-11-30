# Quick Reference: Baseline Method Changes

## Summary of Changes

### ❌ Removed
- **FedKS** (fedks) - Non-existent method with no research paper

### ✅ Added  
- **FedNTD** (fedntd) - Not-True Distillation (NeurIPS 2022)
- **FedSAM** (fedsam) - Sharpness-Aware Minimization (ECCV 2022)

---

## Available Trainer Options

Run any of these methods using:
```bash
python main.py --trainer <method_name> [other args...]
```

### Standard FL Baselines
- `local` - Local training only (no federation)
- `fedavg` - FedAvg (McMahan et al., 2017)
- `fedprox` - FedProx with proximal term
- `scaffold` - SCAFFOLD variance reduction
- `moon` - MOON contrastive learning

### Class Imbalance Methods  
- `fedrs` - FedRS restricted softmax
- `fedsam` - **NEW: FedSAM sharpness-aware** ⭐

### Knowledge Distillation
- `fedntd` - **NEW: FedNTD not-true distillation** ⭐
- `fedproto` - FedProto prototype learning

### Personalization
- `ditto` - Ditto personalization
- `pefll` - PEFLL personalized FL

### Your Methods
- `fedsat` - FedSat struggle-aware aggregation
- Multiple variants: fedsatl, fedmap, fedmapd, etc.

### Advanced Methods
- `floco` - FLOCO
- `flute` - FLUTE
- `ccvr` - CCVR
- `fedblo` - FedBLO
- Many others (see config_main.py TRAINERS dict)

---

## Quick Test Commands

### Test FedNTD (Not-True Distillation)
```bash
python main.py --trainer fedntd \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 100 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

### Test FedSAM (Sharpness-Aware Minimization)
```bash
python main.py --trainer fedsam \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 100 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

### Compare All Key Baselines
```bash
# Run each baseline and compare results
for trainer in fedavg fedprox fedrs fedntd fedsam fedsat; do
    python main.py --trainer $trainer \
        --dataset cifar10 \
        --dataset_type noiid_lbldir_b0_3_k100 \
        --num_rounds 200 \
        --clients_per_round 10 \
        --num_epochs 5 \
        --batch_size 32 \
        --learning_rate 0.01 \
        --seed 42
done
```

---

## File Structure

### New Files Created
```
flearn/
├── clients/
│   ├── fedntd.py          # FedNTD client (NEW) ✅
│   └── fedsam.py          # FedSAM client (NEW) ✅
└── trainers/
    ├── fedntd.py          # FedNTD server (NEW) ✅
    └── fedsam.py          # FedSAM server (NEW) ✅

baseline_papers/
└── BASELINE_IMPLEMENTATIONS.md  # Documentation (NEW) ✅
```

### Files Removed
```
flearn/
├── clients/
│   └── fedks.py           # Removed ❌
└── trainers/
    └── fedks.py           # Removed ❌
```

### Files Modified
```
flearn/config/config_main.py   # Updated TRAINERS dict
```

---

## Hyperparameter Tuning

### FedNTD Hyperparameters
- `beta_ntd`: Weight for not-true distillation loss (default: 1.0)
  - Higher values → more knowledge preservation from global model
  - Lower values → more focus on local data
  - Recommended range: [0.5, 2.0]

- `tau_ntd`: Temperature for distillation (default: 1.0)
  - Higher values → softer probability distribution
  - Lower values → sharper probability distribution
  - Recommended range: [0.5, 2.0]

### FedSAM Hyperparameters
- `rho`: Perturbation radius for SAM (default: 0.05)
  - Higher values → larger perturbations, flatter minima
  - Lower values → smaller perturbations, sharper minima
  - Recommended range: [0.01, 0.1]

---

## Expected Performance

### When to Use FedNTD
✅ Good for:
- Non-IID data with high heterogeneity
- Preventing catastrophic forgetting
- When global knowledge is important
- Class-imbalanced scenarios

❌ May not help much for:
- IID data
- Very small models
- Single-class client scenarios

### When to Use FedSAM
✅ Good for:
- Improving generalization
- Class-imbalanced datasets
- Heterogeneous client data
- When overfitting is a concern

❌ May not help much for:
- Already well-generalized models
- Computational constraints (2x forward passes)
- Very simple tasks

---

## Comparison with Your Method (FedSat+CALC)

| Aspect | FedNTD | FedSAM | FedSat+CALC |
|--------|--------|--------|-------------|
| **Approach** | Knowledge Distillation | Optimization | Loss + Aggregation |
| **Focus** | Global Knowledge | Flat Minima | Struggle-Aware |
| **Client Side** | ✅ | ✅ | ✅ |
| **Server Side** | Standard Agg | Standard Agg | ✅ Custom Agg |
| **Class Imbalance** | ✅ Indirect | ✅ Indirect | ✅ Direct |
| **Computation** | Low | High (2x) | Medium |
| **Memory** | Low | Low | Medium |

---

## Verification Checklist

- [x] FedKS completely removed from codebase
- [x] FedNTD client implemented
- [x] FedNTD server implemented
- [x] FedSAM client implemented
- [x] FedSAM server implemented
- [x] Configuration updated (TRAINERS dict)
- [x] Documentation created
- [ ] Test FedNTD execution (need environment setup)
- [ ] Test FedSAM execution (need environment setup)
- [ ] Run comparative experiments
- [ ] Document results

---

## Next Steps

1. **Set up Python environment** with required dependencies
2. **Run quick tests** to verify implementations work
3. **Design experimental matrix**:
   - Datasets: CIFAR-10, CIFAR-100, FMNIST
   - Non-IID levels: β ∈ {0.1, 0.3, 0.5}
   - Methods: FedAvg, FedRS, FedNTD, FedSAM, FedSat+CALC
4. **Collect results** and create comparison tables
5. **Write paper** highlighting your method's advantages
