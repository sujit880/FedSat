# Baseline Papers - Download Links and Resources

## Core Federated Learning Papers

### 1. FedAvg (2017) - McMahan et al.
**Title**: Communication-Efficient Learning of Deep Networks from Decentralized Data  
**Authors**: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas  
**Venue**: AISTATS 2017  
**Links**:
- arXiv: https://arxiv.org/abs/1602.05629
- PDF: https://arxiv.org/pdf/1602.05629.pdf
- Citation Count: ~11,000+
**Key Contribution**: Foundation of federated averaging algorithm
**Status**: ✅ Must download

---

### 2. FedProx (2020) - Li et al.
**Title**: Federated Optimization in Heterogeneous Networks  
**Authors**: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith  
**Venue**: MLSys 2020  
**Links**:
- arXiv: https://arxiv.org/abs/1812.06127
- PDF: https://arxiv.org/pdf/1812.06127.pdf
- Code: https://github.com/litian96/FedProx
**Key Contribution**: Proximal term for handling statistical and systems heterogeneity
**Status**: ✅ Must download

---

### 3. SCAFFOLD (2020) - Karimireddy et al.
**Title**: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning  
**Authors**: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh  
**Venue**: ICML 2020  
**Links**:
- arXiv: https://arxiv.org/abs/1910.06378
- PDF: https://arxiv.org/pdf/1910.06378.pdf
- Code: https://github.com/epfml/federated-learning-public-code
**Key Contribution**: Control variates for variance reduction in non-IID settings
**Status**: ✅ Must download

---

### 4. Adaptive Federated Optimization (2021) - Reddi et al.
**Title**: Adaptive Federated Optimization  
**Authors**: Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan  
**Venue**: ICLR 2021  
**Links**:
- arXiv: https://arxiv.org/abs/2003.00295
- PDF: https://arxiv.org/pdf/2003.00295.pdf
- OpenReview: https://openreview.net/forum?id=LkFG3lB13U5
**Key Contribution**: FedAdam, FedYogi, FedAdagrad - adaptive server-side optimization
**Status**: ✅ Must download

---

## Class Imbalance in Federated Learning

### 5. FedRS (2021) - Luo et al.
**Title**: FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data  
**Authors**: Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng  
**Venue**: KDD 2021  
**Links**:
- ACM DL: https://dl.acm.org/doi/10.1145/3447548.3467254
- arXiv: (check if available)
**Key Contribution**: Restricted softmax for label distribution skew
**Status**: ⚠️ Must implement - Priority 1

---

### 6. FedSAM (2022) - Caldarola et al.
**Title**: Improving Generalization in Federated Learning by Seeking Flat Minima  
**Authors**: Debora Caldarola, Barbara Caputo, Marco Ciccone  
**Venue**: ECCV 2022  
**Links**:
- arXiv: https://arxiv.org/abs/2203.11834
- PDF: https://arxiv.org/pdf/2203.11834.pdf
- Code: https://github.com/debcaldarola/fedsam
**Key Contribution**: Sharpness-aware minimization for FL with better generalization
**Status**: ⚠️ Should implement - Priority 2

---

### 7. CReFF (2023) - Shi et al.
**Title**: Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning  
**Authors**: Yujun Shi, Jian Liang, Wenqing Zhang, Vincent Y. F. Tan, Song Bai  
**Venue**: ICLR 2023  
**Links**:
- arXiv: https://arxiv.org/abs/2210.00226
- PDF: https://arxiv.org/pdf/2210.00226.pdf
- OpenReview: https://openreview.net/forum?id=EGyEL2FRxV-
- Code: https://github.com/bytedance/FedDecorr
**Key Contribution**: Feature decorrelation to prevent dimensional collapse
**Status**: ⚠️ Nice to have - Priority 3

---

### 8. FedFA (2023) - Frequency-Aware
**Title**: (Search for "Federated Learning Frequency Aware" or similar)  
**Note**: This appears to be in your codebase - document what paper it's from
**Status**: ⚠️ Need to identify source paper

---

## Cost-Sensitive and Loss Function Papers

### 9. Focal Loss (2017) - Lin et al.
**Title**: Focal Loss for Dense Object Detection  
**Authors**: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár  
**Venue**: ICCV 2017  
**Links**:
- arXiv: https://arxiv.org/abs/1708.02002
- PDF: https://arxiv.org/pdf/1708.02002.pdf
- Code (detectron2): https://github.com/facebookresearch/detectron2
**Key Contribution**: Focal loss for addressing class imbalance in object detection
**Status**: ✅ Already implemented, download for reference

---

### 10. Class-Balanced Loss (2019) - Cui et al.
**Title**: Class-Balanced Loss Based on Effective Number of Samples  
**Authors**: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie  
**Venue**: CVPR 2019  
**Links**:
- arXiv: https://arxiv.org/abs/1901.05555
- PDF: https://arxiv.org/pdf/1901.05555.pdf
- Code: https://github.com/richardaecn/class-balanced-loss
**Key Contribution**: Effective number of samples for re-weighting
**Status**: ✅ Already implemented, download for reference

---

### 11. Label Smoothing (2019) - Müller et al.
**Title**: When Does Label Smoothing Help?  
**Authors**: Rafael Müller, Simon Kornblith, Geoffrey Hinton  
**Venue**: NeurIPS 2019  
**Links**:
- arXiv: https://arxiv.org/abs/1906.02629
- PDF: https://arxiv.org/pdf/1906.02629.pdf
**Key Contribution**: Analysis of label smoothing regularization
**Status**: ✅ Already implemented, download for reference

---

## Personalized Federated Learning

### 12. FedProto (2022) - Tan et al.
**Title**: FedProto: Federated Prototype Learning across Heterogeneous Clients  
**Authors**: Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, Chengqi Zhang  
**Venue**: AAAI 2022  
**Links**:
- arXiv: https://arxiv.org/abs/2105.00243
- PDF: https://arxiv.org/pdf/2105.00243.pdf
- Code: https://github.com/yuetan031/fedproto
**Key Contribution**: Prototype-based personalization for heterogeneous clients
**Status**: ✅ Already implemented, download for reference

---

### 13. Ditto (2021) - Li et al.
**Title**: Ditto: Fair and Robust Federated Learning Through Personalization  
**Authors**: Tian Li, Shengyuan Hu, Ahmad Beirami, Virginia Smith  
**Venue**: ICML 2021  
**Links**:
- arXiv: https://arxiv.org/abs/2012.04221
- PDF: https://arxiv.org/pdf/2012.04221.pdf
- Code: https://github.com/litian96/ditto
**Key Contribution**: Fairness and robustness through personalized models
**Status**: ✅ Already implemented, download for reference

---

### 14. MOON (2021) - Li et al.
**Title**: Model-Contrastive Federated Learning  
**Authors**: Qinbin Li, Bingsheng He, Dawn Song  
**Venue**: CVPR 2021  
**Links**:
- arXiv: https://arxiv.org/abs/2103.16257
- PDF: https://arxiv.org/pdf/2103.16257.pdf
- Code: https://github.com/QinbinLi/MOON
**Key Contribution**: Contrastive learning for model alignment in FL
**Status**: ✅ Already implemented, download for reference

---

### 15. FedBN (2021) - Li et al.
**Title**: FedBN: Federated Learning on Non-IID Features via Local Batch Normalization  
**Authors**: Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou  
**Venue**: ICLR 2021  
**Links**:
- arXiv: https://arxiv.org/abs/2102.07623
- PDF: https://arxiv.org/pdf/2102.07623.pdf
- OpenReview: https://openreview.net/forum?id=6YEQUn0QICG
**Key Contribution**: Local batch normalization for feature distribution skew
**Status**: ⚠️ Consider implementing if dealing with feature shift

---

## Survey and Background Papers

### 16. Federated Learning Survey (2020) - Kairouz et al.
**Title**: Advances and Open Problems in Federated Learning  
**Authors**: Peter Kairouz et al. (54 authors)  
**Venue**: Foundations and Trends in Machine Learning  
**Links**:
- arXiv: https://arxiv.org/abs/1912.04977
- PDF: https://arxiv.org/pdf/1912.04977.pdf
**Key Contribution**: Comprehensive survey of FL landscape
**Status**: ✅ Must read - foundational

---

### 17. Non-IID Data in FL (2021) - Li et al.
**Title**: Federated Learning on Non-IID Data Silos: An Experimental Study  
**Authors**: Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He  
**Venue**: ICDE 2022  
**Links**:
- arXiv: https://arxiv.org/abs/2102.02079
- PDF: https://arxiv.org/pdf/2102.02079.pdf
**Key Contribution**: Systematic study of non-IID data challenges
**Status**: ✅ Must read - motivates your work

---

### 18. Personalized FL Survey (2023) - Tan et al.
**Title**: Towards Personalized Federated Learning  
**Authors**: Alysa Ziying Tan, Han Yu, Lizhen Cui, Qiang Yang  
**Venue**: IEEE TNNLS 2023  
**Links**:
- arXiv: https://arxiv.org/abs/2103.00710
- PDF: https://arxiv.org/pdf/2103.00710.pdf
**Key Contribution**: Survey of personalization techniques in FL
**Status**: ✅ Important background

---

## Related Work on Confusion/Cost-Sensitive Learning

### 19. Cost-Sensitive Learning (2006) - Elkan
**Title**: The Foundations of Cost-Sensitive Learning  
**Authors**: Charles Elkan  
**Venue**: IJCAI 2001  
**Links**:
- Citeseer: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.15.6792
- PDF: Search Google Scholar
**Key Contribution**: Theoretical foundations of cost-sensitive learning
**Status**: ✅ Theoretical background

---

### 20. Confusion Matrix in Deep Learning (2020)
**Title**: Learning from Label Proportions with Generative Adversarial Networks  
**Note**: Search for papers on "confusion matrix adaptation" or "confusion-aware learning"
**Status**: ⚠️ Need to find relevant recent papers

---

### 21. Label Distribution Learning (2016) - Geng
**Title**: Label Distribution Learning  
**Authors**: Xin Geng  
**Venue**: IEEE TKDE 2016  
**Links**:
- IEEE: https://ieeexplore.ieee.org/document/7439855
**Key Contribution**: Learning from label distributions rather than single labels
**Status**: ⚠️ Related work for label calibration

---

## Recent FL with Imbalance (2023-2024)

### 22. Search for Recent Papers
**Keywords to search**:
- "federated learning class imbalance 2023"
- "federated learning label distribution skew 2024"
- "federated learning cost-sensitive"
- "federated learning struggling classes"

**Venues to check**:
- NeurIPS 2023, 2024
- ICML 2023, 2024
- ICLR 2023, 2024
- CVPR 2023, 2024
- AAAI 2023, 2024

**Status**: ⚠️ Critical - must check latest work

---

## Benchmarks and Datasets

### 23. LEAF Benchmark (2019)
**Title**: LEAF: A Benchmark for Federated Settings  
**Authors**: Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar  
**Links**:
- arXiv: https://arxiv.org/abs/1812.01097
- PDF: https://arxiv.org/pdf/1812.01097.pdf
- Code: https://github.com/TalwalkarLab/leaf
**Key Contribution**: Standardized federated learning benchmarks
**Status**: ✅ Use for dataset generation

---

### 24. FedML Framework (2020)
**Title**: FedML: A Research Library and Benchmark for Federated Machine Learning  
**Authors**: Chaoyang He et al.  
**Links**:
- arXiv: https://arxiv.org/abs/2007.13518
- Website: https://fedml.ai/
- Code: https://github.com/FedML-AI/FedML
**Key Contribution**: Comprehensive FL framework and benchmarks
**Status**: ✅ Reference implementation

---

## Download Strategy

### Immediate Priority (Download Today)
1. FedAvg (2017) - Foundation
2. FedProx (2020) - Key baseline
3. SCAFFOLD (2020) - Key baseline
4. Adaptive FL (2021) - Server optimization
5. Non-IID Survey (2021) - Motivation
6. FL Survey (2020) - Background

### This Week
7. FedRS (2021) - Must implement
8. FedSAM (2022) - Should implement
9. Focal Loss (2017) - Loss baseline
10. Class-Balanced Loss (2019) - Loss baseline
11. FedProto (2022) - Personalization baseline
12. Ditto (2021) - Personalization baseline

### Next Week
13. CReFF (2023) - Recent work
14. MOON (2021) - Contrastive baseline
15. FedBN (2021) - Feature shift
16. Personalized FL Survey (2023) - Context
17. Latest papers from NeurIPS/ICML 2023-2024

---

## Organization Structure

```
baseline_papers/
├── README.md                      # This file
├── implementation_guide.md        # Implementation instructions
├── paper_links.md                # This file with links
├── core_fl/                      # Core FL papers
│   ├── fedavg_2017.pdf
│   ├── fedprox_2020.pdf
│   ├── scaffold_2020.pdf
│   └── adaptive_fl_2021.pdf
├── class_imbalance/              # Class imbalance papers
│   ├── fedrs_2021.pdf
│   ├── fedsam_2022.pdf
│   └── creff_2023.pdf
├── cost_sensitive/               # Cost-sensitive learning
│   ├── focal_loss_2017.pdf
│   ├── class_balanced_2019.pdf
│   └── cost_sensitive_survey.pdf
├── personalization/              # Personalized FL
│   ├── fedproto_2022.pdf
│   ├── ditto_2021.pdf
│   └── moon_2021.pdf
├── surveys/                      # Survey papers
│   ├── fl_survey_2020.pdf
│   ├── noniid_survey_2021.pdf
│   └── personalized_fl_survey_2023.pdf
├── recent_work/                  # 2023-2024 papers
│   └── (add as you find them)
└── notes/                        # Your reading notes
    ├── key_insights.md
    └── comparison_matrix.md
```

---

## Citation Management

Consider using:
- **Zotero**: Free, open-source reference manager
- **Mendeley**: Popular alternative
- **BibTeX file**: Create `references.bib` for LaTeX

Example BibTeX entries:

```bibtex
@inproceedings{mcmahan2017fedavg,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial intelligence and statistics},
  pages={1273--1282},
  year={2017},
  organization={PMLR}
}

@inproceedings{li2020fedprox,
  title={Federated optimization in heterogeneous networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={Proceedings of Machine Learning and Systems},
  volume={2},
  pages={429--450},
  year={2020}
}
```

---

## Automated Download Script

Create `download_papers.sh`:

```bash
#!/bin/bash

# Create directory structure
mkdir -p baseline_papers/{core_fl,class_imbalance,cost_sensitive,personalization,surveys,recent_work,notes}

# Download core FL papers
cd baseline_papers/core_fl
wget https://arxiv.org/pdf/1602.05629.pdf -O fedavg_2017.pdf
wget https://arxiv.org/pdf/1812.06127.pdf -O fedprox_2020.pdf
wget https://arxiv.org/pdf/1910.06378.pdf -O scaffold_2020.pdf
wget https://arxiv.org/pdf/2003.00295.pdf -O adaptive_fl_2021.pdf

# Download cost-sensitive papers
cd ../cost_sensitive
wget https://arxiv.org/pdf/1708.02002.pdf -O focal_loss_2017.pdf
wget https://arxiv.org/pdf/1901.05555.pdf -O class_balanced_2019.pdf

# Download personalization papers
cd ../personalization
wget https://arxiv.org/pdf/2105.00243.pdf -O fedproto_2022.pdf
wget https://arxiv.org/pdf/2012.04221.pdf -O ditto_2021.pdf
wget https://arxiv.org/pdf/2103.16257.pdf -O moon_2021.pdf

# Download surveys
cd ../surveys
wget https://arxiv.org/pdf/1912.04977.pdf -O fl_survey_2020.pdf
wget https://arxiv.org/pdf/2102.02079.pdf -O noniid_survey_2021.pdf

cd ../..
echo "Papers downloaded successfully!"
```

---

## Reading Priority Order

1. **Day 1**: FedAvg (foundation)
2. **Day 2**: Non-IID Survey (motivation)
3. **Day 3**: FedProx + SCAFFOLD (key baselines)
4. **Day 4**: FedRS (class imbalance - must implement)
5. **Day 5**: Focal Loss + Class-Balanced Loss (loss functions)
6. **Week 2**: FedSAM, FedProto, Ditto, MOON
7. **Week 3**: Recent 2023-2024 papers, CReFF
8. **Ongoing**: FL Survey (reference as needed)

---

**Last Updated**: October 27, 2025
