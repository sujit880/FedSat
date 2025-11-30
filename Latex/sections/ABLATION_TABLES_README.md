# Ablation Study Tables - Usage Guide

I've created **4 versions** of the ablation study tables for your FedSat paper. Choose the one that best fits your paper's style and space constraints.

## Files Created

### 1. `ablation_study.tex` (Recommended - Full Version)
- **Best for:** Full papers, journal submissions
- **Features:**
  - Complete section with subsections
  - Two separate tables (component analysis + top-p sensitivity)
  - Detailed explanations and bullet-point observations
  - Summary section
- **Use when:** You have adequate space and want comprehensive presentation

### 2. `ablation_study_detailed.tex` (Most Detailed)
- **Best for:** Camera-ready versions, supplementary material
- **Features:**
  - Table with checkmarks (\cmark, \xmark) for component presence
  - Shows both "Δ vs CE" and "Δ vs CACS" columns
  - Paragraph-style explanations with findings
  - Most thorough analysis
- **Note:** Requires `\usepackage{multirow}` and `\usepackage{amssymb}` in preamble

### 3. `ablation_study_compact.tex` (Space-Efficient)
- **Best for:** Conference papers with strict page limits
- **Features:**
  - Two compact tables
  - Minimal prose, focuses on numbers
  - Brief explanations below tables
  - Can fit in 1 column or less
- **Use when:** Page limits are tight but you need both tables

### 4. `ablation_study_sidebyside.tex` (Visual Layout)
- **Best for:** Two-column papers, visually-oriented presentations
- **Features:**
  - Side-by-side layout using minipage
  - Component checkmarks for clarity
  - Space-efficient while remaining comprehensive
  - Good for comparing related results
- **Note:** Requires `table*` environment (spans two columns)

## Key Results Summary

### Component Analysis (All Datasets, β=0.3)
| Dataset    | CE Baseline | + CACS | + CACS + FedSat | Total Gain |
|------------|-------------|--------|-----------------|------------|
| CIFAR-10   | 60.47%      | 71.41% | **72.76%**      | +12.29%    |
| FMNIST     | 80.61%      | 84.39% | 83.92%          | +3.31%     |
| CIFAR-100  | 49.69%      | 50.91% | **51.29%**      | +1.60%     |

### Top-p Sensitivity (CIFAR-10, β=0.3)
| Top-p | Accuracy | Gain over CACS |
|-------|----------|----------------|
| --    | 71.41%   | --             |
| p=1   | 73.38%   | +1.97%         |
| p=2   | **73.82%** | +2.41%       |
| p=4   | 72.76%   | +1.35%         |
| p=5   | 73.37%   | +1.96%         |
| p=10  | 71.95%   | +0.54%         |

## How to Use in Your Paper

1. **Choose the version** that fits your paper style
2. **Copy the content** to your results/experiments section
3. **Add to preamble** (if needed):
   ```latex
   \usepackage{multirow}  % For detailed version
   \usepackage{amssymb}   % For checkmarks (\cmark, \xmark)
   \usepackage{booktabs}  % For professional tables (already standard)
   ```
4. **Adjust table positioning** (`[t]`, `[h]`, `[b]`) as needed
5. **Update references** to match your label style (e.g., `\ref{tab:ablation}`)

## Customization Tips

- **Change dataset order:** Reorder rows to emphasize your best results
- **Add more metrics:** Insert columns for per-class F1, worst-class accuracy, etc.
- **Modify deltas:** Change how improvements are displayed (percentage, absolute, relative)
- **Adjust font size:** Use `\footnotesize` or `\scriptsize` if space is very tight
- **Color coding:** Add `\usepackage{xcolor}` and highlight best results with `\textcolor{blue}{...}`

## LaTeX Packages Required

All versions require:
```latex
\usepackage{booktabs}  % \toprule, \midrule, \bottomrule
```

Versions 2 and 4 additionally require:
```latex
\usepackage{multirow}  % \multirow command
\usepackage{amssymb}   % \cmark and \xmark symbols
```

Version 4 requires:
```latex
\usepackage{subcaption}  % \subcaption command
```

## Notes

- All accuracies are from the final round of training (round 105)
- Experiments use: 100 clients, batch size 64, 5 local epochs
- Dirichlet parameter β=0.3 (severe non-IID)
- Top-p values tested: 1, 2, 4, 5, 10 (for 10-class CIFAR-10)

## Questions or Issues?

If you need:
- Different datasets or β values
- Additional metrics (convergence speed, communication cost)
- Different table layouts
- More/fewer decimal places

Just ask and I can generate customized versions!
