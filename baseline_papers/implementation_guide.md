# Baseline Implementation Guide

This document provides detailed implementation instructions for each baseline method.

## Priority 1: Essential Baselines (Implement First)

### 1. FedAvg + LCCE
**Status**: Loss already implemented, need to test  
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CL --agg=fedavg
```

### 2. FedSat + CE (Ablation: Aggregation Only)
**Status**: Need to test  
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedsat
```

### 3. FedSat + CALC (Your Proposed Method)
**Status**: Should work  
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CALC --agg=fedsat
```

### 4. FedAvg + CALC (Ablation: Loss Only)
**Status**: Should work  
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CALC --agg=fedavg
```

---

## Priority 2: Standard Baselines

### 5. FedAvg + CE (Baseline)
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedavg
```

### 6. FedProx
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedprox --num_rounds=150 --loss=CE
```

### 7. SCAFFOLD
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=scaffold --num_rounds=150 --loss=CE
```

---

## Priority 3: Imbalance-Aware Baselines

### 8. FedAvg + Focal Loss
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=FL --agg=fedavg
```

### 9. FedAvg + Class-Balanced Loss
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CB --agg=fedavg
```

---

## Priority 4: Personalized FL Baselines

### 10. FedProto
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedproto --num_rounds=150
```

### 11. Ditto
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=ditto --num_rounds=150
```

### 12. MOON
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=moon --num_rounds=150
```

---

## Priority 5: Adaptive Aggregation

### 13. FedAvgM
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedavgm
```

### 14. FedAdam
**Command**:
```bash
python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedadam
```

---

## Experimental Matrix

### Full Experimental Setup (Example: CIFAR-10)

| Experiment ID | Trainer | Loss | Aggregation | Purpose |
|--------------|---------|------|-------------|---------|
| E1 | fedavg | CE | fedavg | Baseline |
| E2 | fedavg | LCCE | fedavg | Label calibration baseline |
| E3 | fedavg | CALC | fedavg | Ablation: loss only |
| E4 | fedavg | CE | fedsat | Ablation: aggregation only |
| E5 | fedavg | CALC | fedsat | **Proposed method** |
| E6 | fedavg | FL | fedavg | Focal loss baseline |
| E7 | fedavg | CB | fedavg | Class-balanced baseline |
| E8 | fedprox | CE | fedavg | FedProx baseline |
| E9 | scaffold | CE | fedavg | SCAFFOLD baseline |
| E10 | fedavg | CE | fedavgm | Server momentum |
| E11 | fedavg | CE | fedadam | Adaptive server |

### Dataset Variations

For each method above, test on:
1. **CIFAR-10**: β ∈ {0.05, 0.1, 0.3, 0.5}
2. **CIFAR-100**: β ∈ {0.1, 0.3, 0.5}
3. **FMNIST**: β ∈ {0.1, 0.3, 0.5}
4. **EMNIST**: β ∈ {0.1, 0.3}
5. **FEMNIST**: Natural heterogeneity

---

## Batch Experiment Script

Create `run_baselines.sh`:

```bash
#!/bin/bash

# Datasets and betas
DATASETS=("cifar" "cifar100" "fmnist" "emnist")
BETAS=(0.1 0.3 0.5)
ROUNDS=150
EPOCHS=5
CPR=10  # clients per round
NC=100  # num clients
BS=64
LR=0.01

# Baseline experiments
for dataset in "${DATASETS[@]}"; do
    for beta in "${BETAS[@]}"; do
        echo "Running experiments for $dataset with beta=$beta"
        
        # E1: FedAvg + CE (baseline)
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CE --agg=fedavg
        
        # E2: FedAvg + LCCE
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CL --agg=fedavg
        
        # E3: FedAvg + CALC (ablation: loss only)
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CALC --agg=fedavg
        
        # E4: FedSat + CE (ablation: aggregation only)
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CE --agg=fedsat
        
        # E5: FedSat + CALC (proposed method)
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CALC --agg=fedsat
        
        # E6: FedAvg + Focal Loss
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=FL --agg=fedavg
        
        # E7: FedAvg + Class-Balanced Loss
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedavg --num_rounds=$ROUNDS --loss=CB --agg=fedavg
        
        # E8: FedProx
        python main.py --num_epochs=$EPOCHS --clients_per_round=$CPR \
            --dataset=$dataset --dataset_type=noiid_lbldir --beta=$beta \
            --num_clients=$NC --batch_size=$BS --learning_rate=$LR \
            --trainer=fedprox --num_rounds=$ROUNDS --loss=CE
        
        echo "Completed $dataset with beta=$beta"
    done
done

echo "All baseline experiments completed!"
```

---

## Results Analysis Script

Create `analyze_results.py`:

```python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir="RESULTS/json_dump"):
    """Load all experiment results"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                results.append(data)
    return results

def create_comparison_table(results):
    """Create comparison table"""
    df = pd.DataFrame(results)
    
    # Calculate key metrics
    summary = df.groupby(['method', 'dataset', 'beta']).agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'worst_class_accuracy': ['mean', 'std'],
        'rounds_to_converge': 'mean',
        'communication_cost': 'sum'
    }).round(4)
    
    return summary

def plot_convergence(results):
    """Plot convergence curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, dataset in enumerate(['cifar', 'cifar100', 'fmnist', 'emnist']):
        ax = axes[i//2, i%2]
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        for method in ['FedAvg+CE', 'FedAvg+CALC', 'FedSat+CE', 'FedSat+CALC']:
            method_results = [r for r in dataset_results if r['method'] == method]
            if method_results:
                rounds = method_results[0]['rounds']
                accuracy = method_results[0]['test_accuracy_per_round']
                ax.plot(rounds, accuracy, label=method, linewidth=2)
        
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{dataset.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('RESULTS/figures/convergence_comparison.pdf')
    plt.close()

def plot_per_class_accuracy(results):
    """Plot per-class accuracy comparison"""
    # Implement per-class accuracy visualization
    pass

if __name__ == "__main__":
    results = load_results()
    summary = create_comparison_table(results)
    print(summary)
    summary.to_csv('RESULTS/comparison_summary.csv')
    
    plot_convergence(results)
    print("Results analysis completed!")
```

---

## Quick Start Guide

### 1. Verify Implementation
```bash
# Check if all losses are registered
python -c "from flearn.utils.losses import get_loss_fun; print(get_loss_fun('CALC'))"

# Check if aggregation methods work
python -c "from flearn.utils.aggregator import Aggregator; a = Aggregator('fedsat'); print(a.method)"
```

### 2. Run Single Experiment (Test)
```bash
# Quick test on small setup
python main.py --num_epochs=2 --clients_per_round=5 --dataset=cifar \
    --dataset_type=noiid_lbldir --beta=0.3 --num_clients=20 \
    --batch_size=64 --learning_rate=0.01 --trainer=fedavg \
    --num_rounds=10 --loss=CALC --agg=fedsat
```

### 3. Run Full Baseline Suite
```bash
chmod +x run_baselines.sh
./run_baselines.sh
```

### 4. Analyze Results
```bash
python analyze_results.py
```

---

## Expected Results Pattern

Your proposed method (FedSat + CALC) should show:
1. **Higher overall accuracy** than FedAvg + CE (baseline)
2. **Better worst-class accuracy** than aggregation-only (FedSat + CE)
3. **Faster convergence** than loss-only (FedAvg + CALC)
4. **More balanced per-class accuracy** than all baselines
5. **Robustness to beta** (works well across different non-IID levels)

If these hold, you have a strong publication case!

---

## Troubleshooting

### Common Issues

1. **Loss function not found**
   - Check `get_loss_fun()` in `losses.py` includes "CALC"
   - Verify import statements

2. **Aggregation method error**
   - Check `aggregator.py` has "fedsat" method
   - Verify client returns struggler scores

3. **Memory issues**
   - Reduce batch size
   - Use fewer clients per round
   - Enable gradient checkpointing

4. **Slow training**
   - Use GPU: Add `--device cuda`
   - Reduce model size
   - Use mixed precision training

---

## Timeline Recommendation

**Week 1-2**: Run all baseline experiments  
**Week 3**: Analyze results, create figures  
**Week 4**: Ablation studies and sensitivity analysis  
**Week 5-6**: Write paper draft  
**Week 7**: Internal review and revision  
**Week 8**: Submit to conference

---

**Last Updated**: October 27, 2025
