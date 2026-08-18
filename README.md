
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
