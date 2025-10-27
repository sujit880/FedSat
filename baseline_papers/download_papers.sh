#!/bin/bash

# Automated Paper Download Script
# Downloads key federated learning papers for baseline comparison

echo "Creating directory structure..."
cd "$(dirname "$0")"
mkdir -p core_fl class_imbalance cost_sensitive personalization surveys recent_work notes

echo "Downloading core FL papers..."
cd core_fl
echo "  - FedAvg (2017)"
wget -q https://arxiv.org/pdf/1602.05629.pdf -O fedavg_2017.pdf
echo "  - FedProx (2020)"
wget -q https://arxiv.org/pdf/1812.06127.pdf -O fedprox_2020.pdf
echo "  - SCAFFOLD (2020)"
wget -q https://arxiv.org/pdf/1910.06378.pdf -O scaffold_2020.pdf
echo "  - Adaptive Federated Optimization (2021)"
wget -q https://arxiv.org/pdf/2003.00295.pdf -O adaptive_fl_2021.pdf

echo "Downloading cost-sensitive learning papers..."
cd ../cost_sensitive
echo "  - Focal Loss (2017)"
wget -q https://arxiv.org/pdf/1708.02002.pdf -O focal_loss_2017.pdf
echo "  - Class-Balanced Loss (2019)"
wget -q https://arxiv.org/pdf/1901.05555.pdf -O class_balanced_2019.pdf
echo "  - Label Smoothing (2019)"
wget -q https://arxiv.org/pdf/1906.02629.pdf -O label_smoothing_2019.pdf

echo "Downloading personalization papers..."
cd ../personalization
echo "  - FedProto (2022)"
wget -q https://arxiv.org/pdf/2105.00243.pdf -O fedproto_2022.pdf
echo "  - Ditto (2021)"
wget -q https://arxiv.org/pdf/2012.04221.pdf -O ditto_2021.pdf
echo "  - MOON (2021)"
wget -q https://arxiv.org/pdf/2103.16257.pdf -O moon_2021.pdf
echo "  - FedBN (2021)"
wget -q https://arxiv.org/pdf/2102.07623.pdf -O fedbn_2021.pdf

echo "Downloading class imbalance papers..."
cd ../class_imbalance
echo "  - FedSAM (2022)"
wget -q https://arxiv.org/pdf/2203.11834.pdf -O fedsam_2022.pdf
echo "  - CReFF (2023)"
wget -q https://arxiv.org/pdf/2210.00226.pdf -O creff_2023.pdf

echo "Downloading survey papers..."
cd ../surveys
echo "  - Federated Learning Survey (2020)"
wget -q https://arxiv.org/pdf/1912.04977.pdf -O fl_survey_2020.pdf
echo "  - Non-IID Data Survey (2021)"
wget -q https://arxiv.org/pdf/2102.02079.pdf -O noniid_survey_2021.pdf
echo "  - Personalized FL Survey (2023)"
wget -q https://arxiv.org/pdf/2103.00710.pdf -O personalized_fl_survey_2023.pdf
echo "  - LEAF Benchmark (2019)"
wget -q https://arxiv.org/pdf/1812.01097.pdf -O leaf_benchmark_2019.pdf

cd ..

echo ""
echo "✅ Download complete!"
echo ""
echo "Papers downloaded to:"
echo "  - core_fl/         : FedAvg, FedProx, SCAFFOLD, Adaptive FL"
echo "  - cost_sensitive/  : Focal Loss, Class-Balanced Loss, Label Smoothing"
echo "  - personalization/ : FedProto, Ditto, MOON, FedBN"
echo "  - class_imbalance/ : FedSAM, CReFF"
echo "  - surveys/         : FL Survey, Non-IID Survey, Personalized FL Survey, LEAF"
echo ""
echo "Note: FedRS (2021) requires ACM Digital Library access"
echo "Download manually from: https://dl.acm.org/doi/10.1145/3447548.3467254"
echo ""
echo "Next steps:"
echo "1. Review paper_links.md for complete list"
echo "2. Check implementation_guide.md for baseline implementation"
echo "3. Read README.md for novelty assessment"
