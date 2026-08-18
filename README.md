# Stock Market Trend Analysis

## Overview

Stock Market Trend Analysis is a reproducible Python project for descriptive analysis of historical daily market data. It covers stock performance, risk, moving-average trends, return correlations, comparison with the SPY benchmark, and an interactive Streamlit dashboard.

## Objectives

The project analyzes:

- AAPL — Apple
- MSFT — Microsoft
- GOOGL — Alphabet/Google
- AMZN — Amazon
- NVDA — NVIDIA

The stocks are compared with SPY, used as a broad U.S. equity-market benchmark. The analysis is historical and descriptive; it is not intended to provide investment advice.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- yFinance
- Jupyter Notebook
- Streamlit

## Analysis performed

- Historical market-data collection from Yahoo Finance through yFinance
- Data cleaning and exploratory data analysis
- Daily returns and cumulative returns
- Performance metrics and stock comparisons
- SPY benchmark comparison
- Excess return and relative performance
- Daily, annualized, and rolling volatility
- 20-day, 50-day, and 200-day simple moving averages
- Descriptive trend classification
- SMA20/SMA50 crossover analysis
- Daily-return correlation matrix and rolling correlation
- Integrated risk-performance analysis

## Benchmark analysis — SPY

SPY is used as the benchmark for comparing the five stocks with the broader U.S. equity market over the same historical period. The benchmark analysis includes:

- Total return comparison
- Excess return
- Relative performance
- Volatility comparison
- 60-trading-day rolling correlation with SPY

The benchmark workflow is implemented in `src/benchmark_data.py`, `src/benchmark_cleaning.py`, `src/benchmark_returns.py`, and `src/benchmark_analysis.py`. The corresponding notebook is `notebooks/07_benchmark_analysis.ipynb`.

## Interactive Dashboard

The read-only Streamlit dashboard presents the validated project outputs through:

- Overview
- Performance
- Risk
- Trends
- Correlation
- Methodology

It reads validated processed datasets and does not download data, run analysis scripts, or modify project files.

Run locally from the project root:

```bash
streamlit run dashboard/app.py
```

Live Demo: [stock-market-trend-analysis-hsk.streamlit.app](https://stock-market-trend-analysis-hsk.streamlit.app/)

## Project structure

```text
stock-market-trend-analysis/
├── dashboard/
│   ├── app.py
│   └── README.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_performance_analysis.ipynb
│   ├── 03_risk_analysis.ipynb
│   ├── 04_trend_analysis.ipynb
│   ├── 05_correlation_analysis.ipynb
│   ├── 06_final_analysis.ipynb
│   └── 07_benchmark_analysis.ipynb
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── performance_analysis.py
│   ├── risk_analysis.py
│   ├── trend_analysis.py
│   ├── correlation_analysis.py
│   ├── final_analysis.py
│   ├── benchmark_data.py
│   ├── benchmark_cleaning.py
│   ├── benchmark_returns.py
│   └── benchmark_analysis.py
├── outputs/
│   └── figures/
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

## Key outputs

Important generated and dashboard-consumed outputs include:

- `data/processed/final_stock_analysis.csv` — consolidated stock-level analysis
- `data/processed/final_insights.txt` — generated descriptive observations
- `data/processed/benchmark_summary.csv` — stock-versus-SPY comparison
- `data/processed/benchmark_rolling_correlation.csv` — rolling stock-versus-SPY correlations
- `data/processed/correlation_matrix.csv` — daily-return correlation matrix
- `data/processed/correlation_pairs.csv` — ranked unique stock pairs

Phase-specific figures and additional processed files are stored under `outputs/figures/` and `data/processed/`.

## Reproducibility

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

To rebuild the analytical pipeline and benchmark outputs, run the scripts from the project root:

```bash
python src/data_collection.py
python src/data_cleaning.py
python src/performance_analysis.py
python src/risk_analysis.py
python src/trend_analysis.py
python src/correlation_analysis.py
python src/final_analysis.py
python src/benchmark_data.py
python src/benchmark_cleaning.py
python src/benchmark_returns.py
python src/benchmark_analysis.py
```

The notebooks can then be executed in numerical order. To run the already-built dashboard, use only:

```bash
streamlit run dashboard/app.py
```

## Limitations

- The project uses historical data only; it does not provide real-time market data.
- The latest observation means the latest observation available in the datasets.
- The analysis is descriptive rather than predictive.
- Historical performance does not guarantee future results.
- Correlations can change over time and do not imply causation.
- The project does not provide investment advice or trading recommendations.
