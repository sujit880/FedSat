# Quick Start: Dataset Distribution Figures

## TL;DR

All figures have been **already generated** and are saved in: `./RESULTS/figures/`

### View Figures
```bash
cd ./RESULTS/figures/
ls *.png          # See all 11 PNG files
cat README.md     # Read detailed statistics
```

## One-Line Commands

### Generate (or regenerate) all figures
```bash
python generate_distribution_figures.py
```

### List available datasets
```bash
python list_available_datasets.py
```

### Quick stats
```bash
cd RESULTS/figures && ls -lh *.png
```

## What You Get

| Count | Type | Size |
|-------|------|------|
| 1 | Combined grid (3×4) | ~650 KB |
| 4 | FMNIST individual | ~120 KB each |
| 3 | CIFAR-10 individual | ~130 KB each |
| 3 | CIFAR-100 individual | ~130 KB each |
| **11** | **Total** | **~1.4 MB** |

## The 3 Datasets × 4 Beta Values

```
BETA VALUES:  0.5 (least)  0.3  0.1  0.05 (most) heterogeneous

FMNIST:       ✓           ✓    ✓    ✓
CIFAR-10:     ✗           ✓    ✓    ✓
CIFAR-100:    ✗           ✓    ✓    ✓

Legend: ✓ = Available, ✗ = Not in current set
```

## Understanding the Figures

### What Each Figure Shows
- **X-axis**: Client IDs (0-99)
- **Y-axis**: Number of samples per client
- **Bars**: Different colors represent client groupings
- **Box**: Statistics (mean, std dev, min, max)

### Interpretation
- **Uniform bars** = Balanced distribution (good for IID algorithms)
- **Mixed heights** = Realistic non-IID (normal FL scenario)
- **Extreme variation** = Stress test (β=0.05, like FMNIST)

## Key Numbers

| Dataset | Beta | Imbalance Ratio | Min Samples | Max Samples |
|---------|------|-----------------|-------------|-------------|
| FMNIST | 0.05 | **274.80x** | 51 | 14,015 |
| CIFAR-10 | 0.05 | **203.06x** | 177 | 35,941 |
| CIFAR-100 | 0.30 | **2.18x** | 5,271 | 11,507 |

## File Naming Convention

```
distribution_{dataset}_b{beta}.png

Examples:
- distribution_fmnist_b0_50.png    → FMNIST with β=0.5
- distribution_cifar_b0_05.png     → CIFAR-10 with β=0.05
- distribution_cifar100_b0_10.png  → CIFAR-100 with β=0.1
```

## Use Cases

### Case 1: Show Data Heterogeneity
```
Use: distribution_fmnist_b0_05.png
Why: Most extreme variation (274.80x imbalance)
```

### Case 2: Realistic Non-IID
```
Use: distribution_cifar_b0_10.png
Why: Moderate heterogeneity (112.98x imbalance)
```

### Case 3: Balanced Scenario
```
Use: distribution_cifar100_b0_30.png
Why: Nearly uniform (2.18x imbalance)
```

### Case 4: Overview
```
Use: data_distribution_grid.png
Why: See all scenarios at once
```

## Advanced Usage

### Customize Generation
Edit `generate_distribution_figures.py`:
```python
DATASETS = ["fmnist", "cifar", "cifar100", "femnist"]
BETA_VALUES = [0.5, 0.3, 0.1, 0.05, 0.01]
NUM_CLIENTS = 100
```

Then run: `python generate_distribution_figures.py`

### Check Available Data
```bash
python list_available_datasets.py
```

Output shows what's actually on disk:
```
📊 FMNIST:
   Beta       K          Clients         Path
   0.05       100        100             noiid_lbldir_b0_05_k100
   0.10       100        100             noiid_lbldir_b0_1_k100
   ...
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Files not found | Run `python list_available_datasets.py` |
| Want to regenerate | Run `python generate_distribution_figures.py` |
| Need different beta | Check availability then edit script |
| Colors wrong display | Update matplotlib: `pip install -U matplotlib` |

## Performance

- **Generation time**: ~30 seconds
- **File size**: ~1.4 MB total
- **Quality**: 300 DPI (print-ready)
- **Format**: PNG (universal compatibility)

## Tips

✓ Use combined grid for papers/presentations
✓ Use individual plots for detailed analysis  
✓ All figures are publication-ready (300 DPI)
✓ Zoom in to see exact sample counts on bars
✓ Statistics box provides numerical summary

---

**Status**: ✓ Complete and Ready
**Generated**: November 7, 2025
**Next**: See `RESULTS/figures/README.md` for detailed stats
