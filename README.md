# Kuramoto-Bedrosian Project

## Replication Repository

**Companion paper:**  
*Spectral Heterogeneity and Portfolio Resilience: A Kuramoto-Bedrosian Duality Approach to Asset Decoupling*  
Mauricio Benegas — Post Graduate Studies in Economics (CAEN), Federal University of Ceará (UFC)  
Submitted to *Quantitative Finance*, 2026.

---

## Overview

This repository contains the full replication pipeline for the empirical results reported in the paper. All scripts are written in Python and designed to run in Google Colab with data stored in Google Drive. The pipeline transforms raw log-price data into the Spectral Heterogeneity Index (SHI), validates the Bedrosian condition, runs the Longin-Solnik diversification test, and simulates SHI-filtered portfolios.

---

## Repository Structure

```
Kuramoto-Bedrosian-Project/
│
├── README.md                        # This file
│
├── code/                            # Replication scripts
│   ├── FRL-02_epsilon_analysis.py   # SHI computation and cross-section
│   ├── FRL-03_epsilon_rolling.py    # Rolling SHI analysis
│   ├── FRL-10_portfolio_simulation.py     # Main portfolio simulation
│   ├── FRL-11a_robustness_B.py      # Robustness Test B: exclude BANFP
│   ├── FRL-11b_robustness_C.py      # Robustness Test C: rolling window
│   ├── FRL-12_incremental_regression.py   # Incremental OLS regression
│   ├── FRL-13_quantile_regression.py      # Quantile regression (Test D)
│   └── FRL-14_longin_solnik_robust.py     # Permutation and bootstrap tests
│
├── data/
│   └── README_data.md               # Data sources and download protocol
│
├── outputs/
│   └── README_outputs.md            # Figure and table correspondence map
│
├── audit/
│   └── audit_protocol.pdf           # Formal audit and reproducibility protocol
│
└── environment/
    ├── requirements.txt             # Python dependencies
    └── colab_setup.py               # Google Colab setup script
```

---

## Data

The empirical analysis uses daily log-prices for **316 S&P 500 constituents** over the period **January 2007 to January 2026** (4,733 trading days). Data were obtained from public sources. Due to licensing restrictions, raw price data are not distributed in this repository. See `data/README_data.md` for the full data acquisition protocol.

The following processed files are used as inputs to the pipeline and are available upon request:

| File | Description |
|---|---|
| `log_returns_342_final.csv` | Daily log-returns, 316 assets × 4,733 days |
| `phase_342_bk.csv` | BK-filtered instantaneous phase, 316 assets |
| `bk_validation_meta.csv` | Bedrosian validation metadata per asset |
| `gics_sectors.json` | GICS sector classification |

---

## Pipeline

The replication pipeline follows this sequence:

```
Raw log-prices
    ↓ Baxter-King filter (K=84, bands [1/200, 1/10] cycles/day)
BK-filtered signal p̃ᵢ(t)
    ↓ Hilbert transform
Analytic signal zᵢ(t) = p̃ᵢ(t) + iH[p̃ᵢ](t)
    ↓ Instantaneous phase extraction
θᵢ(t) = arg(zᵢ(t))
    ↓ Bedrosian validation (εᵢ = |error_pct| / 100)
    ↓ SHI computation (εᵢ = |ω̂ᵢ - ω̄| / ω̄)
Spectral Heterogeneity Index
    ↓ FRL-02, FRL-03
Cross-sectional analysis + temporal stability
    ↓ FRL-14
Longin-Solnik test (permutation + bootstrap)
    ↓ FRL-10, FRL-11a, FRL-11b
Portfolio simulation + robustness tests
    ↓ FRL-12, FRL-13
Incremental regression + quantile regression
```

---

## Reproducing the Results

### Environment

All scripts run in **Google Colab** (Python 3.10+) with data stored in **Google Drive** at:
```
/content/drive/MyDrive/ste_matrices_corrected/
```

### Steps

1. Clone this repository or download the scripts from `code/`
2. Upload scripts to your Google Colab environment
3. Run the setup script:
   ```python
   exec(open('colab_setup.py').read())
   ```
4. Run scripts in order: FRL-02 → FRL-03 → FRL-10 → FRL-11a → FRL-11b → FRL-12 → FRL-13 → FRL-14

### Dependencies

See `environment/requirements.txt` for the full list. Main dependencies:
- `numpy >= 1.24`
- `pandas >= 2.0`
- `scipy >= 1.10`
- `statsmodels >= 0.14`
- `matplotlib >= 3.7`

---

## Figure and Table Correspondence

| Paper element | Script | Output file |
|---|---|---|
| Figure 1 (SHI histogram) | FRL-02 | `fig_FRL1_histogram.png` |
| Figure 2 (temporal stability) | FRL-02 | `fig_FRL2_stability.png` |
| Figure 3 (dynamic decoupling) | FRL-03 | `fig_FRL3_mechanism.png` |
| Figure 4 (rolling SHI) | FRL-03 | `fig_FRL4_rolling_lineplot.png` |
| Figure 5 (Longin-Solnik) | FRL-14 | `fig_FRL7_longin_solnik.png` |
| Figure 6 (cumulative returns) | FRL-10 | `fig_FRL_portfolio_cumret.png` |
| Figure 7 (drawdown) | FRL-10 | `fig_FRL_portfolio_drawdown.png` |
| Figure 8 (robustness B) | FRL-11a | `fig_robustness_B.png` |
| Figure 9 (robustness C) | FRL-11b | `fig_robustness_C.png` |
| Figure 10 (L-S robust) | FRL-14 | `fig_longin_solnik_robust.png` |
| Table 1 (identified assets) | FRL-02 | `epsilon_summary.csv` |
| Table 2 (subperiod SHI) | FRL-02 | `epsilon_stability.csv` |
| Table 3 (L-S test) | FRL-14 | `longin_solnik_test.csv` |
| Table 4 (portfolio performance) | FRL-10 | `portfolio_metrics.csv` |
| Table A.1 (robustness A) | FRL-10 | `robustness_A_extra_returns.csv` |
| Table A.2 (robustness B) | FRL-11a | `robustness_B_metrics.csv` |
| Table A.3 (robustness C) | FRL-11b | `robustness_C_metrics.csv` |
| Table A.4 (robustness D) | FRL-13 | `quantile_regression_results.csv` |

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| BK filter order | K = 84 | Half-length in trading days |
| BK lower bound | 1/200 | Minimum cycle frequency (cycles/day) |
| BK upper bound | 1/10 | Maximum cycle frequency (cycles/day) |
| Market clock ω̄ | 0.1064 rad/day | Cross-sectional mean instantaneous frequency |
| SHI threshold ε* | 0.20 | Identification threshold (Theorem 3.1) |
| κ_max | 0.021 rad/day | Maximum market coupling force |
| MIN_HIST | 504 days | Minimum history for portfolio simulation |
| REBAL_FREQ | 63 days | Rebalancing frequency (quarterly) |
| Risk-free rate | 2% p.a. | Used in Sharpe ratio computation |
| Permutation draws | 10,000 | For Longin-Solnik robustness test |
| Bootstrap draws | 10,000 | For Longin-Solnik robustness test |

---

## Citation

If you use this code or data in your research, please cite:

```
Benegas, M. (2026). Spectral Heterogeneity and Portfolio Resilience:
A Kuramoto-Bedrosian Duality Approach to Asset Decoupling.
Submitted to Quantitative Finance.
```

---

## Contact

Mauricio Benegas  
Post Graduate Studies in Economics (CAEN)  
Federal University of Ceará (UFC)  
Fortaleza, Brazil  
mbenegas@ufc.br
