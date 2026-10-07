# Dashboard

Read-only Streamlit dashboard for the Nifty 50 market analysis. Run from the project root:

```bash
streamlit run dashboard/app.py
```

It reads the files in `data/processed/` written by `python main.py`; it never downloads data or modifies files.

## Pages

- **Overview** – growth of ₹100 versus the Nifty, drawdown chart, and headline metrics (CAGR, excess CAGR, volatility, Sharpe, max drawdown, beta, alpha) for the selected stock
- **Universe** – risk-versus-return scatter by sector, sortable table of every stock's metrics, and sector medians
- **Risk** – VaR, CVaR, Sortino, Calmar, tracking error, rolling volatility and the daily-return distribution
- **Trends** – price with 20/50/200-day moving averages and crossover markers
- **Correlation** – sector-ordered correlation heatmap, most and least correlated pairs, rolling correlation with the Nifty, and market-wide average correlation over time
- **Data & Methodology** – metric definitions, the data-quality report and the generated findings

## Controls

The sidebar has a sector filter, a stock picker and a date range. Overview and Risk metrics are **recomputed for the selected date range** using the same tested functions as the pipeline (`src/metrics.py`). Universe, sector and correlation views cover the full analysis period.

Set `STOCK_ANALYSIS_DATA_DIR` to point the dashboard at a different processed-data folder (the tests use this).

Historical and descriptive only; not investment advice.
