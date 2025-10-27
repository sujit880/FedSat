# Quick Reference Checklist

## 📋 Publication Readiness Checklist

### Phase 1: Literature Review ✅ IN PROGRESS
- [x] Create baseline_papers directory
- [x] Document baseline methods
- [ ] Download core papers (run `./download_papers.sh`)
- [ ] Read FedAvg (2017)
- [ ] Read FedProx (2020)
- [ ] Read SCAFFOLD (2020)
- [ ] Read FedRS (2021) - **Priority 1**
- [ ] Read Non-IID Survey (2021)
- [ ] Identify recent 2023-2024 work

### Phase 2: Implementation
- [ ] Verify CALC loss works correctly
- [ ] Verify FedSat aggregation works
- [ ] Implement FedRS baseline
- [ ] Test all baseline combinations:
  - [ ] FedAvg + CE (baseline)
  - [ ] FedAvg + LCCE
  - [ ] FedAvg + CALC (ablation)
  - [ ] FedSat + CE (ablation)
  - [ ] FedSat + CALC (proposed)
  - [ ] FedAvg + Focal Loss
  - [ ] FedAvg + CB Loss
  - [ ] FedProx
  - [ ] SCAFFOLD

### Phase 3: Experiments
#### Main Results
- [ ] CIFAR-10: β ∈ {0.05, 0.1, 0.3, 0.5}
- [ ] CIFAR-100: β ∈ {0.1, 0.3, 0.5}
- [ ] FMNIST: β ∈ {0.1, 0.3, 0.5}
- [ ] EMNIST: β ∈ {0.1, 0.3}
- [ ] FEMNIST: Natural heterogeneity

#### Ablation Studies
- [ ] A1: CE + FedAvg (baseline)
- [ ] A2: LCCE + FedAvg (calibration only)
- [ ] A3: CALC + FedAvg (loss only)
- [ ] A4: CE + FedSat (aggregation only)
- [ ] A5: CALC + FedSat (full method)

#### Sensitivity Analysis
- [ ] top_p: {1, 3, 5, K/2, K}
- [ ] tau: {0.5, 1.0, 2.0}
- [ ] conf_beta: {0.1, 0.3, 0.5, 0.7}
- [ ] lmu/cmu: Different ratios

#### Performance Metrics
- [ ] Overall test accuracy
- [ ] Worst-class accuracy
- [ ] Per-class accuracy
- [ ] Convergence rounds
- [ ] Communication cost
- [ ] Computational overhead
- [ ] Fairness metrics (std, Gini)

### Phase 4: Analysis & Visualization
- [ ] Create convergence plots
- [ ] Create per-class accuracy bar charts
- [ ] Create comparison tables
- [ ] Analyze confusion matrices
- [ ] Plot hyperparameter sensitivity
- [ ] Calculate statistical significance
- [ ] Create supplementary figures

### Phase 5: Writing
- [ ] Abstract (highlight synergy)
- [ ] Introduction (motivation + contributions)
- [ ] Related Work (position clearly)
- [ ] Methodology
  - [ ] CALC loss description
  - [ ] FedSat aggregation description
  - [ ] Algorithm pseudocode
  - [ ] Information flow diagram
- [ ] Experiments
  - [ ] Setup details
  - [ ] Main results table
  - [ ] Ablation results
  - [ ] Analysis sections
- [ ] Discussion
  - [ ] When it works best
  - [ ] Limitations
  - [ ] Future work
- [ ] Conclusion

### Phase 6: Pre-Submission
- [ ] Internal review
- [ ] Check all claims have evidence
- [ ] Verify all figures/tables referenced
- [ ] Proofread for clarity
- [ ] Check citations format
- [ ] Prepare supplementary materials
- [ ] Code release (GitHub)
- [ ] Reproducibility checklist

---

## 🎯 Key Success Metrics

### Must Achieve (for Tier-1 acceptance)
- [ ] FedSat+CALC > FedAvg+CE by ≥5% overall accuracy
- [ ] FedSat+CALC > all baselines on worst-class by ≥10%
- [ ] Consistent across ≥4 datasets
- [ ] Synergistic: A5 > A3 + (A4 - A1)
- [ ] Clear experimental reproducibility

### Should Achieve
- [ ] Convergence ≥10% faster than FedAvg
- [ ] Robust to hyperparameters (±2%)
- [ ] Overhead <10%
- [ ] Works for β ∈ [0.05, 0.5]

---

## 📊 Experiment Quick Commands

### Test Single Setup
```bash
# Quick test (small scale)
python main.py --num_epochs=2 --clients_per_round=5 \
    --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 \
    --num_clients=20 --batch_size=64 --learning_rate=0.01 \
    --trainer=fedavg --num_rounds=10 --loss=CALC --agg=fedsat
```

### Full Experiment (Proposed Method)
```bash
# FedSat + CALC on CIFAR-10, beta=0.3
python main.py --num_epochs=5 --clients_per_round=10 \
    --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 \
    --num_clients=100 --batch_size=64 --learning_rate=0.01 \
    --trainer=fedavg --num_rounds=150 --loss=CALC --agg=fedsat
```

### Baseline (FedAvg + CE)
```bash
python main.py --num_epochs=5 --clients_per_round=10 \
    --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 \
    --num_clients=100 --batch_size=64 --learning_rate=0.01 \
    --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedavg
```

### Ablation: Loss Only
```bash
python main.py --num_epochs=5 --clients_per_round=10 \
    --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 \
    --num_clients=100 --batch_size=64 --learning_rate=0.01 \
    --trainer=fedavg --num_rounds=150 --loss=CALC --agg=fedavg
```

### Ablation: Aggregation Only
```bash
python main.py --num_epochs=5 --clients_per_round=10 \
    --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 \
    --num_clients=100 --batch_size=64 --learning_rate=0.01 \
    --trainer=fedavg --num_rounds=150 --loss=CE --agg=fedsat
```

---

## 📝 Paper Outline Template

### Title
"FedSat-CALC: Synergistic Federated Learning via Confusion-Aware Training and Struggle-Targeted Aggregation"

### Abstract (250 words)
- **Problem**: FL with non-IID + class imbalance
- **Gap**: Existing methods don't address both synergistically
- **Solution**: CALC (client) + FedSat (server)
- **Key Idea**: Confusion tracking → struggler scores → class-specific aggregation
- **Results**: X% improvement on worst-class accuracy

### Contributions (for Introduction)
1. Novel synergistic framework combining confusion-aware loss + struggle-targeted aggregation
2. CALC loss: label calibration + confusion-aware cost penalties + EMA adaptation
3. FedSat aggregation: class-specific weighting based on client competence
4. Comprehensive evaluation on 5+ datasets with ablation studies

### Key Figures to Create
1. **Figure 1**: System overview (client CALC → server FedSat feedback loop)
2. **Figure 2**: Convergence comparison (FedSat+CALC vs baselines)
3. **Figure 3**: Per-class accuracy comparison (bar charts)
4. **Figure 4**: Ablation study results
5. **Figure 5**: Hyperparameter sensitivity
6. **Figure 6**: Worst-class accuracy vs β (different non-IID levels)

### Key Tables to Create
1. **Table 1**: Main results (all methods × all datasets)
2. **Table 2**: Ablation study (A1-A5)
3. **Table 3**: Computational overhead
4. **Table 4**: Fairness metrics

---

## 🚀 Quick Start Guide

### Day 1: Setup
```bash
cd baseline_papers
./download_papers.sh
```
Read: FedAvg, Non-IID Survey

### Day 2-3: Read Core Papers
- FedAvg (foundation)
- FedProx (heterogeneity)
- SCAFFOLD (variance reduction)
- FedRS (class imbalance)

### Day 4-5: Verify Implementation
```bash
# Test CALC loss
python -c "from flearn.utils.losses import get_loss_fun; print(get_loss_fun('CALC'))"

# Test FedSat aggregation
python -c "from flearn.utils.aggregator import Aggregator; a = Aggregator('fedsat'); print(a.method)"

# Run quick test
python main.py --num_epochs=2 --clients_per_round=5 --dataset=cifar \
    --dataset_type=noiid_lbldir --beta=0.3 --num_clients=20 \
    --batch_size=64 --learning_rate=0.01 --trainer=fedavg \
    --num_rounds=10 --loss=CALC --agg=fedsat
```

### Week 2: Run Baselines
Use `implementation_guide.md` commands to run all baseline experiments

### Week 3-4: Full Experiments
Run complete experimental suite on all datasets

### Week 5-6: Analysis & Writing
Create figures, tables, write paper draft

### Week 7: Review & Revision
Internal review, iterate on feedback

### Week 8: Submit!
Target ICML, NeurIPS, or ICLR

---

## 💡 Quick Tips

### For Experiments
- Always use same random seed for reproducibility
- Save checkpoints regularly
- Log everything (tensorboard/wandb)
- Run baselines in parallel if possible
- Double-check hyperparameters match paper descriptions

### For Writing
- Lead with the problem, not the solution
- Use concrete examples in introduction
- Make contributions explicit and numbered
- Every claim needs evidence (cite or experiment)
- Figures should be self-explanatory

### For Rebuttal (if needed)
- Be respectful and constructive
- Acknowledge valid concerns
- Provide additional experiments if reasonable
- Clarify misunderstandings clearly
- Don't argue with opinions, address facts

---

## 📚 Key Resources

### Documentation
- `README.md`: Full novelty assessment and baseline overview
- `NOVELTY_ASSESSMENT.md`: Detailed novelty analysis
- `implementation_guide.md`: How to run each baseline
- `paper_links.md`: All paper download links
- This file: Quick reference checklist

### Code References
- `flearn/utils/losses.py`: CALC loss implementation
- `flearn/utils/aggregator.py`: FedSat aggregation
- `flearn/trainers/fedavg.py`: Training logic

### External Resources
- FedML: https://github.com/FedML-AI/FedML
- LEAF: https://github.com/TalwalkarLab/leaf
- Papers with Code: https://paperswithcode.com/task/federated-learning

---

## ✅ Current Status

**Date**: October 27, 2025

**Completed**:
- [x] Baseline directory structure created
- [x] Novelty assessment documented
- [x] Implementation guide created
- [x] Paper links compiled
- [x] Download script prepared

**Next Immediate Steps**:
1. Run `./download_papers.sh` to get papers
2. Read FedAvg and Non-IID Survey
3. Test CALC + FedSat on small experiment
4. Implement FedRS baseline
5. Run CIFAR-10 full comparison

**Timeline**: 8-10 weeks to submission

**Target Venue**: ICML 2025 or NeurIPS 2025

**Confidence**: High (7.5/10) - Strong publishable work with proper validation

---

## 🎓 Final Advice

### Do's ✅
- Focus on synergistic benefits in experiments
- Show consistent improvements across multiple settings
- Be thorough in ablation studies
- Write clearly and concisely
- Position work carefully in related work
- Provide reproducible code and instructions

### Don'ts ❌
- Don't oversell - be honest about limitations
- Don't ignore negative results - discuss them
- Don't cherry-pick best results - report averages
- Don't skip statistical significance tests
- Don't submit without internal review
- Don't ignore reviewer feedback

---

**Good luck with your publication! 🚀**

This is a solid, publishable piece of work. With thorough experiments and clear writing, you have a strong chance at a top-tier venue.

Remember: The novelty is in the **synergy**, not individual components. Make this crystal clear in your paper!
