# Simple Example Configurations
# Copy and modify these commands for your experiments

# ========================================
# BASELINE METHODS
# ========================================

# FedAvg on CIFAR-10 with moderate non-IID
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedProx on CIFAR-100 with severe non-IID
python main.py --trainer fedprox --dataset cifar100 --dataset_type noiid_lbldir_b0_1_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --mu 0.01 --seed 42 --gpu 0 --cuda True

# SCAFFOLD on FMNIST
python main.py --trainer scaffold --dataset fmnist --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# ========================================
# ADAPTIVE OPTIMIZATION METHODS
# ========================================

# FedAdagrad on CIFAR-10
python main.py --trainer fedadagrad --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedYogi on CIFAR-10 (recommended for non-IID)
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedAdam on CIFAR-100
python main.py --trainer fedadam --dataset cifar100 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# ========================================
# CLASS IMBALANCE METHODS
# ========================================

# FedRS on CIFAR-10
python main.py --trainer fedrs --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedSAM on CIFAR-100
python main.py --trainer fedsam --dataset cifar100 --dataset_type noiid_lbldir_b0_1_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# ========================================
# KNOWLEDGE DISTILLATION METHODS
# ========================================

# FedNTD on CIFAR-10
python main.py --trainer fedntd --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedProto on CIFAR-100
python main.py --trainer fedproto --dataset cifar100 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# ========================================
# YOUR METHOD (FedSat)
# ========================================

# FedSat with Cross Entropy
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# FedSat with CALC (full proposed method)
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CALC --seed 42 --gpu 0 --cuda True

# FedSat with CACS
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CACS --seed 42 --gpu 0 --cuda True

# ========================================
# ABLATION STUDY
# ========================================

# 1. Baseline: FedAvg + CE
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# 2. CALC contribution: FedAvg + CALC
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CALC --seed 42 --gpu 0 --cuda True

# 3. Aggregation contribution: FedSat + CE
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# 4. Full method: FedSat + CALC
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CALC --seed 42 --gpu 0 --cuda True

# ========================================
# DIFFERENT NON-IID LEVELS
# ========================================

# Mild non-IID (beta=0.5)
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_5_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Moderate non-IID (beta=0.3)
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Severe non-IID (beta=0.1)
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_1_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Extreme non-IID (beta=0.05)
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_05_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# ========================================
# DIFFERENT LOSSES
# ========================================

# Cross Entropy (baseline)
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Focal Loss
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss FL --seed 42 --gpu 0 --cuda True

# Class Balanced Loss
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CB --seed 42 --gpu 0 --cuda True

# CALC (your loss)
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 200 --clients_per_round 10 --num_epochs 5 --batch_size 32 \
    --learning_rate 0.01 --loss CALC --seed 42 --gpu 0 --cuda True

# ========================================
# QUICK TESTS (10 rounds)
# ========================================

# Quick test FedAvg
python main.py --trainer fedavg --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Quick test FedYogi
python main.py --trainer fedyogi --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1 --batch_size 32 \
    --learning_rate 0.01 --loss CE --seed 42 --gpu 0 --cuda True

# Quick test FedSat+CALC
python main.py --trainer fedsat --dataset cifar10 --dataset_type noiid_lbldir_b0_3_k100 \
    --num_rounds 10 --clients_per_round 10 --num_epochs 1 --batch_size 32 \
    --learning_rate 0.01 --loss CALC --seed 42 --gpu 0 --cuda True
