# Data Sources and Acquisition Protocol

## Overview

The empirical analysis uses daily log-prices for 316 S&P 500
constituents over January 2007 to January 2026 (4,733 trading days).

## Primary Data Source

- **Provider:** Daily adjusted closing prices from public financial
  data providers (Yahoo Finance / CRSP)
- **Universe:** S&P 500 constituents as of January 2007, survivorship
  bias adjusted to include delisted and merged companies
- **Period:** 2007-01-02 to 2026-01-21
- **Frequency:** Daily (trading days only)
- **Transformation:** Log-returns: r_t = log(P_t) - log(P_{t-1})

## Files Used in Pipeline

| File | Rows | Columns | Description |
|---|---|---|---|
| log_returns_342_final.csv | 4,733 | 316 | Daily log-returns |
| phase_342_bk.csv | 4,733 | 316 | BK-filtered instantaneous phase |
| bk_validation_meta.csv | 316 | 5 | Bedrosian validation metadata |
| gics_sectors.json | 316 | — | GICS sector classification |

## Bedrosian Validation Criteria

Assets were included in the final sample of 316 if they satisfied:
1. **Criterion 1:** Mean instantaneous frequency within BK passband
   (error_pct <= 20%)
2. **Criterion 2:** Positive instantaneous frequency in >= 80% of
   observations

Starting universe: 342 assets
After Criterion 1: 326 assets
After Criterion 2: 316 assets (final sample)

## Data Availability

Raw price data are not distributed in this repository due to
licensing restrictions. Processed files (log_returns_342_final.csv,
phase_342_bk.csv, bk_validation_meta.csv) are available from the
corresponding author upon reasonable request.
