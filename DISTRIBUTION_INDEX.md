# Distribution Figures - Complete Index

## 📍 Quick Navigation

| Resource | Location | Purpose |
|----------|----------|---------|
| **Generated Figures** | `./RESULTS/figures/` | 11 PNG images (300 DPI) |
| **Quick Start** | `./QUICK_START_DISTRIBUTION.md` | 5-minute guide |
| **Full Guide** | `./DISTRIBUTION_FIGURES_SUMMARY.md` | Complete documentation |
| **Detailed Stats** | `./RESULTS/figures/README.md` | Statistical analysis |
| **Main Script** | `./generate_distribution_figures.py` | Figure generator |
| **Helper Script** | `./list_available_datasets.py` | Dataset discovery |

## 🎯 Choose Your Path

### Path A: I Just Want to Use the Figures
**Time: 2 minutes**
1. Open `./RESULTS/figures/data_distribution_grid.png` (overview)
2. Browse individual `distribution_*.png` files
3. Read statistics in `./RESULTS/figures/README.md`
✓ **Done** - Ready for papers, presentations, analysis

### Path B: I Want to Understand the Data
**Time: 10 minutes**
1. Read `QUICK_START_DISTRIBUTION.md`
2. Run: `python list_available_datasets.py`
3. Check statistics in `./RESULTS/figures/README.md`
✓ **Done** - Understand data heterogeneity levels

### Path C: I Want to Generate Custom Figures
**Time: 15 minutes**
1. Read `DISTRIBUTION_FIGURES_SUMMARY.md`
2. Edit `generate_distribution_figures.py` (lines 9-11)
3. Run: `python generate_distribution_figures.py`
✓ **Done** - Custom figures for your specific needs

### Path D: Complete Deep Dive
**Time: 30 minutes**
1. Read all documentation files
2. Study the Python scripts
3. Run `list_available_datasets.py`
4. Regenerate with custom parameters
5. Analyze results in detail
✓ **Done** - Expert understanding of all aspects

## 📊 Figure Inventory

### Combined Grid
- **File**: `data_distribution_grid.png`
- **Size**: ~650 KB
- **Content**: 3×4 subplots (all datasets × all betas)
- **Best for**: Overview, papers, presentations

### Individual FMNIST (4 figures)
```
distribution_fmnist_b0_50.png   (β=0.5, imbalance: 7.73x)
distribution_fmnist_b0_30.png   (β=0.3, imbalance: 21.65x)
distribution_fmnist_b0_10.png   (β=0.1, imbalance: 205.59x)
distribution_fmnist_b0_05.png   (β=0.05, imbalance: 274.80x)
```

### Individual CIFAR-10 (3 figures)
```
distribution_cifar_b0_30.png    (β=0.3, imbalance: 8.14x)
distribution_cifar_b0_10.png    (β=0.1, imbalance: 112.98x)
distribution_cifar_b0_05.png    (β=0.05, imbalance: 203.06x)
```

### Individual CIFAR-100 (3 figures)
```
distribution_cifar100_b0_30.png (β=0.3, imbalance: 2.18x)
distribution_cifar100_b0_10.png (β=0.1, imbalance: 2.82x)
distribution_cifar100_b0_05.png (β=0.05, imbalance: 5.90x)
```

## 📈 Data Heterogeneity Spectrum

```
HETEROGENEITY LEVEL:  LOW          MEDIUM         HIGH          EXTREME
                      |             |              |              |
β VALUE:              0.5           0.3            0.1            0.05
                      |             |              |              |
FMNIST:               ✓             ✓              ✓              ✓
                    7.73x         21.65x         205.59x        274.80x
                      |             |              |              |
CIFAR-10:             ✗             ✓              ✓              ✓
                                   8.14x         112.98x        203.06x
                      |             |              |              |
CIFAR-100:            ✗             ✓              ✓              ✓
                                   2.18x           2.82x          5.90x
```

## 🔍 Finding Figures by Use Case

### "I need to show data heterogeneity challenge"
→ Use: `distribution_fmnist_b0_05.png`
   - Most extreme: 274.80x imbalance

### "I need realistic non-IID scenario"
→ Use: `distribution_cifar_b0_10.png`
→ Or: `distribution_fmnist_b0_30.png`
   - Moderate to high heterogeneity

### "I need nearly balanced distribution"
→ Use: `distribution_cifar100_b0_30.png`
   - Only 2.18x imbalance

### "I need complete overview"
→ Use: `data_distribution_grid.png`
   - All combinations in one image

### "I need to compare β effects"
→ Use: All 4 FMNIST plots together
   - Shows progression: 7.73x → 21.65x → 205.59x → 274.80x

## 🛠️ Available Scripts

### `generate_distribution_figures.py`
Generate distribution figures for any available dataset/beta combination.

**Usage:**
```bash
python generate_distribution_figures.py
```

**Customization:**
Edit lines 9-11:
```python
DATASETS = ["fmnist", "cifar", "cifar100", ...]
BETA_VALUES = [0.5, 0.3, 0.1, 0.05, ...]
NUM_CLIENTS = 100
```

**Output:**
- Creates individual PNG files in `./RESULTS/figures/`
- Creates combined grid plot
- Prints statistics to console

### `list_available_datasets.py`
Discover all available dataset/beta configurations in your DATA directory.

**Usage:**
```bash
python list_available_datasets.py
```

**Output:**
```
📊 FMNIST:
   Beta       K          Clients         Path
   0.05       100        100             noiid_lbldir_b0_05_k100
   0.10       100        100             noiid_lbldir_b0_1_k100
   ...
```

## 📖 Documentation Files

### 1. QUICK_START_DISTRIBUTION.md
- **Length**: 5 minutes to read
- **Content**: Essential commands, quick reference
- **Best for**: Getting started immediately

### 2. DISTRIBUTION_FIGURES_SUMMARY.md
- **Length**: 15 minutes to read
- **Content**: Complete guide with troubleshooting
- **Best for**: Understanding everything

### 3. RESULTS/figures/README.md
- **Length**: 10 minutes to read
- **Content**: Detailed statistics and analysis
- **Best for**: Statistical deep dive

## 🔑 Key Statistics Summary

| Metric | Best (Most Balanced) | Worst (Most Heterogeneous) |
|--------|----------------------|-----------------------------|
| Dataset | CIFAR-100 | FMNIST |
| Beta | 0.30 | 0.05 |
| Imbalance Ratio | 2.18x | 274.80x |
| Min Samples | 5,271 | 51 |
| Max Samples | 11,507 | 14,015 |
| Std Dev | 854.6 | 2,747 |

## ✅ Quality Assurance

All figures meet publication standards:
- ✓ 300 DPI resolution
- ✓ Publication-ready quality
- ✓ Clear labels and legends
- ✓ Statistical annotations
- ✓ Consistent styling
- ✓ Color-blind friendly palette

## 🚀 Common Tasks

### Task 1: Use in a Paper
1. Open `data_distribution_grid.png` for methods section
2. Use individual plots for results/appendix
3. Reference statistics from README.md

### Task 2: Test Algorithm on Different Heterogeneity
1. Use β=0.30 (realistic)
2. Use β=0.10 (challenging)
3. Use β=0.05 (extreme stress test)

### Task 3: Create Custom Combinations
1. Edit `generate_distribution_figures.py`
2. Change DATASETS and BETA_VALUES
3. Run: `python generate_distribution_figures.py`

### Task 4: Document System Setup
1. Use `data_distribution_grid.png` for system overview
2. Reference imbalance ratios for heterogeneity claims
3. Include statistics from README.md

## 📞 Support

| Question | Answer | Location |
|----------|--------|----------|
| How do I view the figures? | Open PNG files | `./RESULTS/figures/` |
| What do the statistics mean? | See interpretation guide | `README.md` (figures dir) |
| How do I regenerate? | Run Python script | `generate_distribution_figures.py` |
| What datasets exist? | Run discovery script | `list_available_datasets.py` |
| How do I customize? | Edit script and regenerate | `DISTRIBUTION_FIGURES_SUMMARY.md` |

---

**Status**: ✓ Complete and Ready
**Last Updated**: November 7, 2025
**Next Steps**: Choose your path above and get started!
