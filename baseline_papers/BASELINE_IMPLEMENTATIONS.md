# New Baseline Methods Implementation Summary

## Overview
Removed the non-existent FedKS method and implemented two important baseline methods for federated learning with class imbalance and non-IID data.

## Removed Method

### ❌ FedKS (FedKSeed)
- **Status**: Removed
- **Reason**: Not a real published method - no corresponding research paper exists
- **Files Removed**:
  - `flearn/clients/fedks.py`
  - `flearn/trainers/fedks.py`
  - Entry in `TRAINERS` config

---

## Newly Implemented Methods

### ✅ 1. FedNTD - Not-True Distillation for Federated Learning

**Paper**: "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning"  
**Venue**: NeurIPS 2022  
**Link**: https://proceedings.neurips.cc/paper_files/paper/2022/file/fadec8f2e65f181d777507d1df69b92f-Paper-Conference.pdf

#### Key Idea
- Preserves global knowledge during local training using knowledge distillation
- For each sample, distills from logits of classes that are **NOT** the true label
- Prevents catastrophic forgetting of global model knowledge
- Particularly effective for:
  - Non-IID data distributions
  - Class-imbalanced scenarios
  - Preventing client drift

#### Implementation Details
- **Client**: `flearn/clients/fedntd.py` (`FedNTDClient`)
- **Server**: `flearn/trainers/fedntd.py` (`FedNTDServer`)
- **Loss Function**: Cross-entropy + Not-True KL Divergence
  ```
  L_total = L_CE(student, y_true) + β * KL(student_not-true || teacher_not-true)
  ```
- **Hyperparameters**:
  - `beta_ntd`: Weight for NTD loss (default: 1.0)
  - `tau_ntd`: Temperature for distillation (default: 1.0)

#### Usage
```bash
python main.py --trainer fedntd \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_1_k100 \
    --num_rounds 200 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

#### Why Compare Against This
- State-of-the-art knowledge distillation approach for FL
- Directly addresses the knowledge forgetting problem in non-IID settings
- Complements your CALC approach (local loss vs global knowledge preservation)

---

### ✅ 2. FedSAM - Sharpness-Aware Minimization for Federated Learning

**Paper**: "Improving Generalization in Federated Learning by Seeking Flat Minima"  
**Venue**: ECCV 2022  
**Link**: https://arxiv.org/abs/2203.11834

#### Key Idea
- Applies Sharpness-Aware Minimization (SAM) to federated learning
- Seeks flatter loss landscape minima which generalize better
- Two-step optimization per batch:
  1. Compute perturbation in direction of steepest ascent
  2. Update parameters based on gradient at perturbed point
- Particularly effective for:
  - Non-IID data distributions
  - Class-imbalanced datasets
  - Improving generalization across heterogeneous clients

#### Implementation Details
- **Client**: `flearn/clients/fedsam.py` (`FedSAMClient`)
- **Server**: `flearn/trainers/fedsam.py` (`FedSAMServer`)
- **Optimization**: SAM with double forward-backward pass
  ```
  ε_w = ρ * ∇L(w) / ||∇L(w)||
  w_t+1 = w_t - η * ∇L(w_t + ε_w)
  ```
- **Hyperparameters**:
  - `rho`: Perturbation radius (default: 0.05)

#### Usage
```bash
python main.py --trainer fedsam \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_1_k100 \
    --num_rounds 200 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

#### Why Compare Against This
- Addresses generalization under non-IID through optimization landscape
- Known to work well with class imbalance
- Different approach from your loss-based method (optimization vs loss design)

---

## Configuration Changes

### Updated Files
1. **`flearn/config/config_main.py`**:
   - Removed: `"fedks": {"server": "FedKSeedServer", "client": "FedKSeedClient"}`
   - Added: `"fedntd": {"server": "FedNTDServer", "client": "FedNTDClient"}`
   - Added: `"fedsam": {"server": "FedSAMServer", "client": "FedSAMClient"}`

### New Files Created
1. `flearn/clients/fedntd.py` - FedNTD client implementation
2. `flearn/trainers/fedntd.py` - FedNTD server implementation
3. `flearn/clients/fedsam.py` - FedSAM client implementation
4. `flearn/trainers/fedsam.py` - FedSAM server implementation

---

## Complete Baseline Method List

### Core FL Methods
- ✅ **FedAvg** - Standard federated averaging
- ✅ **FedProx** - Proximal term for heterogeneity
- ✅ **SCAFFOLD** - Variance reduction
- ✅ **MOON** - Model contrastive learning

### Class Imbalance Methods
- ✅ **FedRS** - Restricted softmax for label skew
- ✅ **FedSAM** - Sharpness-aware minimization (NEW)

### Knowledge Distillation Methods
- ✅ **FedNTD** - Not-true distillation (NEW)
- ✅ **FedProto** - Prototype-based learning

### Personalization Methods
- ✅ **Ditto** - Fair and robust personalization
- ✅ **PEFLL** - Personalized federated learning

### Your Methods
- ✅ **FedSat** - Struggle-aware targeted aggregation
- ✅ **FedSat + CALC** - Your proposed method

---

## Recommended Comparison Matrix

| Method | Type | Handles Non-IID | Handles Imbalance | Publication |
|--------|------|-----------------|-------------------|-------------|
| FedAvg | Standard | ❌ | ❌ | AISTATS 2017 |
| FedProx | Heterogeneity | ✅ | ❌ | MLSys 2020 |
| SCAFFOLD | Variance Reduction | ✅ | ❌ | ICML 2020 |
| FedRS | Class Imbalance | ✅ | ✅ | KDD 2021 |
| **FedNTD** | **Knowledge Distill** | **✅** | **✅** | **NeurIPS 2022** |
| **FedSAM** | **Optimization** | **✅** | **✅** | **ECCV 2022** |
| MOON | Contrastive | ✅ | ❌ | CVPR 2021 |
| FedProto | Prototype | ✅ | ✅ | AAAI 2022 |
| **FedSat+CALC** | **Your Method** | **✅** | **✅** | **Proposed** |

---

## Testing the Implementation

### Quick Test
```bash
# Test FedNTD
python main.py --trainer fedntd --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1 --batch_size 32

# Test FedSAM
python main.py --trainer fedsam --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1 --batch_size 32
```

### Verify Removal of FedKS
```bash
# This should give an error (fedks no longer valid)
python main.py --trainer fedks --dataset cifar10
# Error: argument --trainer: invalid choice: 'fedks'
```

---

## Next Steps

1. **Test the implementations** to ensure they work correctly
2. **Run comparative experiments** on your datasets:
   - CIFAR-10/100 with different β values
   - FMNIST, EMNIST with non-IID settings
3. **Document results** comparing:
   - FedAvg (baseline)
   - FedRS (class imbalance baseline)
   - FedNTD (knowledge distillation baseline)
   - FedSAM (optimization baseline)
   - FedSat + CALC (your method)
4. **Consider implementing** additional baselines if needed:
   - FedLC (Label Calibration) - if you want direct comparison to your CALC
   - CReFF (Class-Rebalancing Frequency Filter) - recent class imbalance work

---

## References

1. **FedNTD**: Lee et al., "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning", NeurIPS 2022
2. **FedSAM**: Caldarola et al., "Improving Generalization in Federated Learning by Seeking Flat Minima", ECCV 2022
