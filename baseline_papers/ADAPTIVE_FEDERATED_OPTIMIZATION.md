# Adaptive Federated Optimization Methods

## Overview
Implemented three adaptive server-side optimization methods from the paper:
**"Adaptive Federated Optimization"** by Reddi et al. (MLSys 2021)

Paper: https://arxiv.org/abs/2003.00295

## Key Insight

The paper proposes **server-side adaptive optimization** while clients use standard SGD. This approach:
- Maintains low client-side computational cost
- Adapts learning rates based on global gradient statistics
- Handles non-IID data more effectively than FedAvg
- Improves convergence in heterogeneous federated settings

---

## Implemented Methods

### ✅ 1. FedAdagrad - Adaptive Federated Optimization with Adagrad

**Paper**: Reddi et al., "Adaptive Federated Optimization", MLSys 2021

#### Algorithm
```
Server maintains:
- v_t: accumulated squared pseudo-gradients
- η: server learning rate

At round t:
1. Clients train with standard SGD → get Δx_i
2. Server computes: Δ_t = Σ_i (p_i * Δx_i)
3. Update: v_t = v_t + Δ_t²
4. Apply: x_t+1 = x_t + η * Δ_t / (√v_t + τ)
```

#### Key Features
- Accumulates squared gradients for adaptive step sizes
- Good for sparse gradients
- Simpler than Adam/Yogi (no momentum)
- Memory efficient

#### Implementation
- **Server**: `flearn/trainers/fedadagrad.py` (`FedAdagradServer`)
- **Client**: Uses `FedAvgClient` (standard SGD)
- **Hyperparameters**:
  - `server_learning_rate` (η): Default 0.01
  - `tau` (τ): Adaptive LR parameter, default 1e-3

#### Usage
```bash
python main.py --trainer fedadagrad \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

---

### ✅ 2. FedYogi - Adaptive Federated Optimization with Yogi

**Paper**: Reddi et al., "Adaptive Federated Optimization", MLSys 2021

#### Algorithm
```
Server maintains:
- m_t: first moment (momentum)
- v_t: second moment (adaptive LR)
- β1, β2: decay rates

At round t:
1. Clients train with standard SGD → get Δx_i
2. Server computes: Δ_t = Σ_i (p_i * Δx_i)
3. First moment: m_t = β1 * m_{t-1} + (1 - β1) * Δ_t
4. Second moment (KEY DIFFERENCE): 
   v_t = v_{t-1} - (1 - β2) * Δ_t² * sign(v_{t-1} - Δ_t²)
5. Bias correction: m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
6. Update: x_t+1 = x_t + η * m̂_t / (√v̂_t + τ)
```

#### Key Features
- **Additive** second moment update (vs Adam's multiplicative)
- More stable than Adam in federated settings
- Better handles non-IID data
- Recommended in the paper for most scenarios

#### Implementation
- **Server**: `flearn/trainers/fedyogi.py` (`FedYogiServer`)
- **Client**: Uses `FedAvgClient` (standard SGD)
- **Hyperparameters**:
  - `server_learning_rate` (η): Default 0.01
  - `beta1` (β1): First moment decay, default 0.9
  - `beta2` (β2): Second moment decay, default 0.99
  - `tau` (τ): Adaptive LR parameter, default 1e-3

#### Usage
```bash
python main.py --trainer fedyogi \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

---

### ✅ 3. FedAdam - Adaptive Federated Optimization with Adam

**Paper**: Reddi et al., "Adaptive Federated Optimization", MLSys 2021

#### Algorithm
```
Server maintains:
- m_t: first moment (momentum)
- v_t: second moment (adaptive LR)
- β1, β2: decay rates

At round t:
1. Clients train with standard SGD → get Δx_i
2. Server computes: Δ_t = Σ_i (p_i * Δx_i)
3. First moment: m_t = β1 * m_{t-1} + (1 - β1) * Δ_t
4. Second moment (STANDARD ADAM): 
   v_t = β2 * v_{t-1} + (1 - β2) * Δ_t²
5. Bias correction: m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
6. Update: x_t+1 = x_t + η * m̂_t / (√v̂_t + τ)
```

#### Key Features
- Standard Adam applied to federated pseudo-gradients
- **Multiplicative** second moment update
- Well-known and widely used
- May be less stable than Yogi for non-IID data

#### Implementation
- **Server**: `flearn/trainers/fedadam.py` (`FedAdamServer`)
- **Client**: Uses `FedAvgClient` (standard SGD)
- **Hyperparameters**:
  - `server_learning_rate` (η): Default 0.01
  - `beta1` (β1): First moment decay, default 0.9
  - `beta2` (β2): Second moment decay, default 0.99
  - `tau` (τ): Adaptive LR parameter, default 1e-3

#### Usage
```bash
python main.py --trainer fedadam \
    --dataset cifar10 \
    --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 \
    --clients_per_round 10 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01
```

---

## Comparison Table

| Method | Server Optimizer | Moments | Second Moment Update | Stability | Complexity |
|--------|-----------------|---------|---------------------|-----------|------------|
| **FedAvg** | None (weighted avg) | 0 | - | Medium | Low |
| **FedAdagrad** | Adagrad | 1 (adaptive only) | v_t = v_t + Δ_t² | High | Low |
| **FedAdam** | Adam | 2 (momentum + adaptive) | v_t = β2·v_{t-1} + (1-β2)·Δ_t² | Medium | Medium |
| **FedYogi** | Yogi | 2 (momentum + adaptive) | v_t = v_{t-1} - (1-β2)·Δ_t²·sign(...) | **High** | Medium |

---

## Key Differences from FedAvg Aggregation Methods

### Previous Implementation (in aggregator.py)
- Used `--agg fedadam` or `--agg fedyogi` with `--trainer fedavg`
- Part of aggregation module, not separate trainers
- Mixed with other aggregation strategies

### New Implementation (separate trainers)
- Use `--trainer fedadagrad`, `--trainer fedyogi`, or `--trainer fedadam`
- Standalone trainer methods following the paper's algorithm
- Clean separation of concerns
- Easier to compare against other FL methods

---

## When to Use Each Method

### Use FedAdagrad when:
- ✅ Sparse gradients are expected
- ✅ Simple adaptive method needed
- ✅ Memory is limited
- ✅ Gradients have widely different scales

### Use FedAdam when:
- ✅ Familiar with standard Adam
- ✅ Well-tuned Adam hyperparameters available
- ✅ Moderate non-IID settings
- ✅ Want standard baseline

### Use FedYogi when:
- ✅ **Highly non-IID data** (RECOMMENDED by paper)
- ✅ Class imbalanced scenarios
- ✅ Need stability over speed
- ✅ Heterogeneous client data
- ✅ **Best overall performance** (paper's conclusion)

---

## Hyperparameter Tuning Guidelines

### Server Learning Rate (η)
- **FedAdagrad**: Try [0.01, 0.05, 0.1]
- **FedAdam**: Try [0.001, 0.01, 0.1]
- **FedYogi**: Try [0.001, 0.01, 0.1]

### Beta1 (First Moment) - Adam/Yogi only
- Standard: 0.9
- Range: [0.8, 0.95]
- Higher = more momentum

### Beta2 (Second Moment) - Adam/Yogi only
- Standard: 0.99
- Range: [0.9, 0.999]
- Higher = slower adaptation

### Tau (τ) - All methods
- Standard: 1e-3
- Range: [1e-4, 1e-2]
- Prevents division by zero in adaptive LR

---

## Experimental Results from Paper

The paper shows that on non-IID federated datasets:

1. **FedYogi** > FedAdam > FedAdagrad > FedAvg (general trend)
2. **FedYogi** particularly strong on:
   - EMNIST (character recognition)
   - Stackoverflow (next-word prediction)
   - CIFAR-100 with label skew
3. Adaptive methods achieve **faster convergence** than FedAvg
4. **Server LR** is critical - needs tuning per dataset

---

## Configuration Changes

### Updated Files
1. **`flearn/config/config_main.py`**:
   - Added: `"fedadagrad": {"server": "FedAdagradServer", "client": "FedAvgClient"}`
   - Added: `"fedyogi": {"server": "FedYogiServer", "client": "FedAvgClient"}`
   - Added: `"fedadam": {"server": "FedAdamServer", "client": "FedAvgClient"}`

### New Files Created
1. `flearn/trainers/fedadagrad.py` - FedAdagrad server implementation
2. `flearn/trainers/fedyogi.py` - FedYogi server implementation
3. `flearn/trainers/fedadam.py` - FedAdam server implementation

---

## Complete Baseline Method List (Updated)

### Core FL Methods
- ✅ FedAvg - Standard federated averaging
- ✅ FedProx - Proximal term for heterogeneity
- ✅ SCAFFOLD - Variance reduction
- ✅ MOON - Model contrastive learning

### Adaptive Optimization (NEW) ⭐
- ✅ **FedAdagrad** - Server-side Adagrad
- ✅ **FedYogi** - Server-side Yogi (RECOMMENDED)
- ✅ **FedAdam** - Server-side Adam

### Class Imbalance Methods
- ✅ FedRS - Restricted softmax
- ✅ FedSAM - Sharpness-aware minimization

### Knowledge Distillation
- ✅ FedNTD - Not-true distillation
- ✅ FedProto - Prototype-based learning

### Your Methods
- ✅ FedSat - Struggle-aware aggregation
- ✅ FedSat + CALC - Your proposed method

---

## Recommended Experimental Comparison

### Baselines to Compare
1. **FedAvg** - Basic baseline
2. **FedYogi** - Best adaptive method (from paper)
3. **FedAdam** - Popular adaptive method
4. **FedRS** - Class imbalance baseline
5. **FedSat + CALC** - Your method

### Test Matrix
```bash
# For each dataset and β value
for dataset in cifar10 cifar100 fmnist; do
  for beta in 0.1 0.3 0.5; do
    for trainer in fedavg fedyogi fedadam fedrs fedsat; do
      python main.py --trainer $trainer \
        --dataset $dataset \
        --dataset_type noiid_lbldir_b${beta}_k100 \
        --num_rounds 200 \
        --clients_per_round 10 \
        --num_epochs 5 \
        --batch_size 32 \
        --learning_rate 0.01 \
        --seed 42
    done
  done
done
```

---

## Citation

If using these methods, cite:
```bibtex
@inproceedings{reddi2021adaptive,
  title={Adaptive Federated Optimization},
  author={Reddi, Sashank and Charles, Zachary and Zaheer, Manzil and Garrett, Zachary and Rush, Keith and Kone{\v{c}}n{\`y}, Jakub and Kumar, Sanjiv and McMahan, H Brendan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

---

## Testing the Implementation

### Quick Test
```bash
# Test FedAdagrad
python main.py --trainer fedadagrad --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1

# Test FedYogi
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1

# Test FedAdam
python main.py --trainer fedadam --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1
```

### Verify they're registered
```bash
python main.py --help | grep -E "fedadagrad|fedyogi|fedadam"
```

---

## Next Steps

1. ✅ Implementation complete
2. ⏳ Test on small dataset to verify correctness
3. ⏳ Run comparative experiments
4. ⏳ Tune server learning rates
5. ⏳ Document results vs FedSat+CALC
