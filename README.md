# Stock Market Trend Analysis

## Overview

This project provides a reproducible Python workflow for collecting, cleaning, and exploring historical daily stock-market data for AAPL, MSFT, GOOGL, AMZN, and NVDA.

## Objectives

The project examines historical prices and volume, stock returns, volatility, moving-average trends, cross-stock return correlations, and an integrated risk-performance view. Results are descriptive and are not investment advice.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- yFinance
- Jupyter Notebook

## Analysis performed

- Data collection from Yahoo Finance through yFinance
- Data cleaning and exploratory analysis
- Daily and cumulative stock returns
- Performance metrics and comparison
- Daily, annualized, and rolling volatility
- 20-, 50-, and 200-day moving averages
- Descriptive trend and SMA20/SMA50 crossover analysis
- Daily-return correlation matrix and rolling correlation
- Final integrated risk-performance analysis

## Project structure

```text
stock-market-trend-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_performance_analysis.ipynb
│   ├── 03_risk_analysis.ipynb
│   ├── 04_trend_analysis.ipynb
│   ├── 05_correlation_analysis.ipynb
│   └── 06_final_analysis.ipynb
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── performance_analysis.py
│   ├── risk_analysis.py
│   ├── trend_analysis.py
│   ├── correlation_analysis.py
│   └── final_analysis.py
├── outputs/
│   └── figures/
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

## Key outputs

The final integration is implemented in `src/final_analysis.py` and summarized in `notebooks/06_final_analysis.ipynb`.

- `data/processed/final_stock_analysis.csv` — one consolidated row per stock
- `data/processed/final_insights.txt` — generated descriptive observations
- `outputs/figures/final_performance_comparison.png`
- `outputs/figures/final_risk_comparison.png`
- `outputs/figures/final_risk_vs_performance.png`

Earlier notebooks and phase-specific outputs remain available under `notebooks/`, `data/processed/`, and `outputs/figures/`.

## Benchmark Analysis — SPY

The additive benchmark analysis compares the five stocks with SPY over the same historical period. It calculates excess total return, relative performance, annualized-volatility differences, volatility ratios, and 60-trading-day rolling correlations with SPY.

The benchmark modules are `src/benchmark_data.py`, `src/benchmark_cleaning.py`, `src/benchmark_returns.py`, and `src/benchmark_analysis.py`. The notebook is `notebooks/07_benchmark_analysis.ipynb`. New benchmark data and figures are saved separately as `SPY_clean.csv`, `SPY_returns.csv`, `benchmark_summary.csv`, `benchmark_rolling_correlation.csv`, and benchmark figures under `outputs/figures/`. Existing project outputs are not changed.

## How to run

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

To reproduce the pipeline from the beginning, run:

```bash
python src/data_collection.py
python src/data_cleaning.py
python src/performance_analysis.py
python src/risk_analysis.py
python src/trend_analysis.py
python src/correlation_analysis.py
python src/final_analysis.py
```

The notebooks can then be executed in numerical order. Raw downloaded CSV files and generated processed/figure outputs are ignored by Git where appropriate; the scripts regenerate them locally.

## Limitations

- The analysis uses historical data only.
- It is descriptive rather than predictive.
- Historical performance does not guarantee future results.
- Correlations can change over time and do not imply causation.
- The project does not provide investment advice, trading signals, or portfolio recommendations.

## Interactive Dashboard

The read-only Streamlit dashboard presents the validated project outputs through Overview, Performance, Risk, Trends, Correlation, and Methodology sections, including SPY benchmark comparisons and rolling correlation. It reads processed datasets and does not modify project data or run analysis scripts.

Run it from the project root with:

```bash
streamlit run dashboard/app.py
```

For Streamlit Community Cloud, use `dashboard/app.py` as the main file path.
Live Demo: To be deployed.
