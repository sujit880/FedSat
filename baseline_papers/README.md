# Baseline FL Approaches for Comparison

This directory contains information about baseline federated learning approaches that should be compared with the proposed **FedSat + CALC** approach.

## Proposed Approach: FedSat with CALC Loss

### Overview
Your approach combines:
1. **CALC (CACS-LC)**: Confusion-Aware Cost-Sensitive Loss with Label Calibration for client-side training
2. **FedSat Aggregation**: Struggle-Aware Targeted aggregation focusing on top-p struggling classes

### Key Innovations

#### 1. CALC Loss (Client-Side)
- **Confusion-Aware Cost Sensitivity**: Maintains EMA confusion matrix to dynamically adjust cost penalties Δ[y,j] for misclassification pairs
- **Label Calibration**: Applies global class frequency-based calibration (τ * π^(-0.25)) to handle class imbalance
- **Adaptive Mechanism**: Combines confusion-based adaptive margins with Bayesian prior adjustments
- **Struggler Score Computation**: Computes per-class struggle scores based on misclassification rates

#### 2. FedSat Aggregation (Server-Side)
- **Class-Specific Weighting**: Identifies top-p struggling classes globally
- **Client Competence Weighting**: For each struggling class, weights clients inversely by their struggler score (1.0 - s[cls])
- **Multiple Specialized Models**: Creates p class-specialized models, then averages them
- **Heterogeneity-Aware**: Better handles non-IID data by focusing updates where clients struggle most

### Novelty Assessment

#### ✅ **Strong Points for Publication:**

1. **Synergistic Design**: The combination is logical - CALC identifies struggling classes at the client level, and FedSat uses this information for targeted aggregation. This creates a feedback loop.

2. **Dual-Level Adaptation**: 
   - Client level: confusion-aware cost-sensitive learning
   - Server level: struggle-aware selective aggregation
   
3. **Addresses Multiple Challenges**:
   - Class imbalance (label calibration)
   - Confusion patterns (cost-sensitive margins)
   - Data heterogeneity (class-specific aggregation)
   - Client heterogeneity (competence weighting)

4. **Theoretical Soundness**: Both components have solid theoretical foundations in cost-sensitive learning and personalized FL.

#### ⚠️ **Concerns and Areas to Strengthen:**

1. **Incremental vs Revolutionary**: While the combination is novel, both components individually are evolutionary rather than revolutionary. The paper needs to demonstrate that the combination provides synergistic benefits beyond additive improvements.

2. **Related Work Overlap**: Need to carefully differentiate from:
   - FedLC (Federated Learning with Label Calibration)
   - FedProto and other prototype-based methods
   - Cost-sensitive FL approaches
   - Class-imbalanced FL methods (FedRS, FedSAM, etc.)

3. **Computational Overhead**: The confusion matrix tracking and multiple model aggregations add complexity. Need to demonstrate it's justified.

4. **Empirical Validation Requirements**:
   - Must show clear improvements over baselines across multiple datasets
   - Need ablation studies showing both components are necessary
   - Should demonstrate robustness to hyperparameters (top_p, tau, conf_beta, etc.)

---

## Essential Baseline Methods to Compare

### 1. **Standard FL Baselines**

#### FedAvg (McMahan et al., 2017)
- **Paper**: Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Link**: https://arxiv.org/abs/1602.05629
- **Why Compare**: Foundation baseline - must outperform this significantly
- **Implementation**: Already available in your codebase

#### FedProx (Li et al., 2020)
- **Paper**: Federated Optimization in Heterogeneous Networks
- **Link**: https://arxiv.org/abs/1812.06127
- **Why Compare**: Handles heterogeneity with proximal term
- **Implementation**: Already available (`fedprox.py`)

#### SCAFFOLD (Karimireddy et al., 2020)
- **Paper**: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
- **Link**: https://arxiv.org/abs/1910.06378
- **Why Compare**: Variance reduction for non-IID data
- **Implementation**: Already available (`scaffold.py`)

---

### 2. **Class Imbalance FL Methods**

#### FedRS (Luo et al., 2021)
- **Paper**: FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data
- **Link**: https://dl.acm.org/doi/10.1145/3447548.3467254
- **Why Compare**: Directly addresses label distribution skew
- **Key Idea**: Restricted softmax to focus on local class subset
- **Status**: ⚠️ Need to implement

#### FedSAM (Caldarola et al., 2022)
- **Paper**: Improving Generalization in Federated Learning by Seeking Flat Minima
- **Link**: https://arxiv.org/abs/2203.11834
- **Why Compare**: Addresses both heterogeneity and imbalance
- **Key Idea**: Sharpness-aware minimization for better generalization
- **Status**: ⚠️ Need to implement

#### CReFF (Shi et al., 2023)
- **Paper**: CReFF: Federated Learning with Class-Rebalancing Frequency Filter
- **Link**: https://arxiv.org/abs/2306.10125
- **Why Compare**: Recent work on class imbalance in FL
- **Key Idea**: Frequency-domain rebalancing
- **Status**: ⚠️ Need to implement

---

### 3. **Label Calibration / Cost-Sensitive Methods**

#### Label Calibration CE (LCCE) - Baseline Loss
- **Paper**: Various works on label distribution calibration
- **Why Compare**: Your CALC builds on this, need to show CALC > LCCE
- **Status**: ⚠️ Already implemented in `losses.py`, need to test as baseline

#### Focal Loss (Lin et al., 2017)
- **Paper**: Focal Loss for Dense Object Detection
- **Link**: https://arxiv.org/abs/1708.02002
- **Why Compare**: Popular cost-sensitive loss for imbalanced data
- **Status**: ✅ Already implemented (`FocalLoss` in losses.py)

#### Class-Balanced Loss (Cui et al., 2019)
- **Paper**: Class-Balanced Loss Based on Effective Number of Samples
- **Link**: https://arxiv.org/abs/1901.05555
- **Why Compare**: State-of-art for class imbalance
- **Status**: ✅ Already implemented (`ClassBalancedCELoss` in losses.py)

---

### 4. **Personalized/Adaptive FL Methods**

#### FedProto (Tan et al., 2022)
- **Paper**: FedProto: Federated Prototype Learning across Heterogeneous Clients
- **Link**: https://arxiv.org/abs/2105.00243
- **Why Compare**: Uses prototypes for personalization
- **Status**: ✅ Already implemented (`fedproto.py`)

#### Ditto (Li et al., 2021)
- **Paper**: Ditto: Fair and Robust Federated Learning Through Personalization
- **Link**: https://arxiv.org/abs/2012.04221
- **Why Compare**: Personalization through dual models
- **Status**: ✅ Already implemented (`ditto.py`)

#### MOON (Li et al., 2021)
- **Paper**: Model-Contrastive Federated Learning
- **Link**: https://arxiv.org/abs/2103.16257
- **Why Compare**: Contrastive learning for model consistency
- **Status**: ✅ Already implemented (`moon.py`)

---

### 5. **Aggregation Strategy Variants**

#### FedAvgM (Hsu et al., 2019)
- **Paper**: Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
- **Link**: https://arxiv.org/abs/1909.06335
- **Why Compare**: Server momentum can help with non-IID
- **Status**: ✅ Already implemented in `aggregator.py`

#### FedAdam (Reddi et al., 2021)
- **Paper**: Adaptive Federated Optimization
- **Link**: https://arxiv.org/abs/2003.00295
- **Why Compare**: Adaptive server-side optimization
- **Status**: ✅ Already implemented in `aggregator.py`

---

### 6. **Recent/Specialized Methods**

#### FedFA (Your existing implementation)
- **Status**: ✅ Already implemented (`fedfa.py`)
- **Note**: Compare against this as well

#### Elastic Aggregation
- **Status**: ✅ Already implemented in `aggregator.py`
- **Note**: Uses gradient sensitivity weighting

---

## Recommended Experimental Setup

### Datasets (Non-IID Settings)
1. **CIFAR-10**: Label distribution skew (β = 0.05, 0.1, 0.3, 0.5)
2. **CIFAR-100**: High class count, severe imbalance
3. **FMNIST**: Fashion MNIST with label skew
4. **EMNIST**: Extended MNIST with natural heterogeneity
5. **FEMNIST**: Real-world federated dataset

### Metrics to Report
1. **Accuracy**: Overall test accuracy
2. **Per-Class Accuracy**: Especially for minority/struggling classes
3. **Fairness Metrics**: 
   - Worst-class accuracy
   - Standard deviation across classes
4. **Communication Efficiency**: Rounds to convergence
5. **Computational Cost**: Training time per round

### Ablation Studies Required
1. **CALC vs CE**: Show confusion-aware cost-sensitivity helps
2. **CALC vs LCCE**: Show confusion awareness adds value over calibration alone
3. **FedSat vs FedAvg**: Show targeted aggregation helps
4. **FedAvg+CALC vs FedSat+CE**: Show both components needed
5. **FedSat+CALC vs FedSat+CALC (full)**: Complete ablation
6. **Hyperparameter Sensitivity**: top_p, tau, conf_beta, lmu, cmu

### Comparison Matrix

| Method | Loss | Aggregation | Handles Imbalance | Handles Non-IID | Complexity |
|--------|------|-------------|-------------------|-----------------|------------|
| FedAvg | CE | Weighted Avg | ❌ | ❌ | Low |
| FedProx | CE + Prox | Weighted Avg | ❌ | ✅ | Low |
| FedAvg + Focal | Focal | Weighted Avg | ✅ | ❌ | Low |
| FedAvg + CB | CB Loss | Weighted Avg | ✅ | ❌ | Low |
| FedAvg + LCCE | LCCE | Weighted Avg | ✅ | ❌ | Medium |
| FedSat + CE | CE | Struggle-Aware | ❌ | ✅ | Medium |
| **FedSat + CALC** | **CALC** | **Struggle-Aware** | **✅** | **✅** | **High** |

---

## Implementation Checklist

### High Priority (Must Have)
- [x] FedAvg baseline (already implemented)
- [x] FedProx (already implemented)
- [x] Focal Loss (already implemented)
- [x] Class-Balanced Loss (already implemented)
- [ ] FedAvg + LCCE (need to test)
- [ ] FedSat + CE (need to test)
- [ ] FedRS implementation
- [ ] Comprehensive ablation studies

### Medium Priority (Should Have)
- [ ] FedSAM implementation
- [ ] SCAFFOLD detailed comparison
- [ ] FedProto detailed comparison
- [ ] Communication cost analysis
- [ ] Convergence speed analysis

### Low Priority (Nice to Have)
- [ ] CReFF implementation
- [ ] Additional personalization baselines
- [ ] Fairness metric analysis
- [ ] Robustness to adversarial clients

---

## Key Papers to Download

### Must Read (Core Baselines)
1. FedAvg (2017) - McMahan et al.
2. FedProx (2020) - Li et al.
3. FedRS (2021) - Luo et al.
4. Adaptive Federated Optimization (2021) - Reddi et al.
5. SCAFFOLD (2020) - Karimireddy et al.

### Important for Positioning
6. Focal Loss (2017) - Lin et al.
7. Class-Balanced Loss (2019) - Cui et al.
8. FedProto (2022) - Tan et al.
9. Ditto (2021) - Li et al.
10. FedSAM (2022) - Caldarola et al.

### Recent Related Work
11. CReFF (2023) - Shi et al.
12. FedLC papers (search for "federated learning label calibration")
13. Cost-sensitive federated learning surveys
14. Class imbalance in FL surveys (2023-2024)

---

## Writing Strategy for Publication

### Title Suggestions
1. "FedSat-CALC: Confusion-Aware Federated Learning with Struggle-Targeted Aggregation"
2. "Synergistic Federated Learning: Combining Cost-Sensitive Training with Struggle-Aware Aggregation"
3. "Adaptive Class-Aware Federated Learning for Non-IID and Imbalanced Data"

### Key Contributions to Highlight
1. **Novel Synergy**: First work combining confusion-aware cost-sensitive loss with struggle-targeted aggregation
2. **Dual-Level Adaptation**: Client-level confusion tracking + server-level struggle-aware weighting
3. **Theoretical Justification**: Why this combination makes sense (provide analysis)
4. **Comprehensive Evaluation**: Extensive experiments on 5+ datasets with multiple non-IID settings
5. **Practical Insights**: When and why FedSat+CALC works better than alternatives

### Story Arc
1. **Problem**: FL with both class imbalance and data heterogeneity
2. **Gap**: Existing methods address one or the other, not both synergistically
3. **Solution**: CALC identifies struggling patterns, FedSat uses them for targeted aggregation
4. **Evidence**: Strong empirical results, especially on minority/struggling classes
5. **Impact**: Enables practical FL in challenging real-world scenarios

---

## Venue Suggestions

### Top-Tier Conferences (Aim for these)
- **NeurIPS** (Neural Information Processing Systems)
- **ICML** (International Conference on Machine Learning)
- **ICLR** (International Conference on Learning Representations)
- **CVPR** (Computer Vision and Pattern Recognition)
- **AAAI** (Association for Advancement of Artificial Intelligence)

### Strong FL-Focused Venues
- **MLSys** (Machine Learning and Systems)
- **AISTATS** (Artificial Intelligence and Statistics)
- **FedML** (Federated Learning workshops at major conferences)

### Journals (If conference doesn't work)
- **JMLR** (Journal of Machine Learning Research)
- **IEEE TPAMI** (Pattern Analysis and Machine Intelligence)
- **Neural Networks**

---

## Next Steps

1. **Literature Review**: Download and read all baseline papers listed above
2. **Implementation**: Implement missing baselines (FedRS priority)
3. **Experiments**: Run comprehensive comparison on all datasets
4. **Analysis**: Conduct ablation studies and sensitivity analysis
5. **Writing**: Draft paper with strong motivation and clear contributions
6. **Iteration**: Refine based on results and feedback

---

## Additional Resources

### Surveys to Read
1. "Federated Learning on Non-IID Data Silos: An Experimental Study" (2021)
2. "Survey on Federated Learning Threats: Concepts, Taxonomy on Attacks and Defences" (2023)
3. "A Survey on Personalized Federated Learning" (2023)

### Code Repositories
- FedML: https://github.com/FedML-AI/FedML
- LEAF Benchmark: https://github.com/TalwalkarLab/leaf
- FedLab: https://github.com/SMILELab-FL/FedLab

---

**Last Updated**: October 27, 2025
