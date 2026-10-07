# Nifty 50 Market Analysis

Historical performance, risk and correlation analysis of the 50 stocks in India's Nifty 50 index, benchmarked against the index itself, with an interactive Streamlit dashboard.

Live demo: [stock-market-trend-analysis-hsk.streamlit.app](https://stock-market-trend-analysis-hsk.streamlit.app/)

## What it does

- Downloads five years of daily prices (Oct 2021 – Sep 2026) for all 50 constituents and a total-return benchmark
- Cleans the data and writes a **data-quality report**: dropped rows, coverage, late listings, suspicious moves
- Computes returns from **adjusted closes**, so dividends and splits are included (total return, not price return)
- Calculates, per stock and for the benchmark:
  - **Return:** total return, CAGR, excess CAGR versus the benchmark over the same dates
  - **Risk:** annualised volatility, maximum drawdown with peak / trough / recovery dates, VaR and CVaR (95%, one day)
  - **Risk-adjusted:** Sharpe, Sortino, Calmar
  - **Versus the market:** beta, Jensen's alpha, correlation, tracking error, information ratio
- Summarises **sectors** (median metrics, share of stocks beating the index)
- Builds a sector-ordered **correlation matrix**, ranked pairs, rolling correlation with the index, and a market-wide average pairwise correlation over time
- Labels each stock's moving-average trend (SMA 20/50/200) and SMA20/SMA50 crossovers
- Writes a plain-text findings file and summary figures

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # download, clean, analyse and write all outputs
streamlit run dashboard/app.py     # open the dashboard
```

`python main.py --skip-download` re-runs the analysis on already-downloaded raw files.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The metric functions are tested against hand-calculated values. The end-to-end tests run the whole pipeline and every dashboard page on synthetic market data, so they need no network access. GitHub Actions runs them on every push.

## Project structure

```text
├── main.py                    # entry point: runs the whole pipeline
├── src/
│   ├── config.py              # universe, sectors, dates, assumptions, paths (edit here only)
│   ├── data_collection.py     # Yahoo Finance download with retries
│   ├── data_cleaning.py       # cleaning rules + data-quality report
│   ├── metrics.py             # pure metric functions (CAGR, Sharpe, drawdown, beta, VaR, ...)
│   ├── performance.py         # returns, stock / benchmark / sector summaries
│   ├── trends.py              # moving averages, trend labels, crossovers
│   ├── correlation.py         # correlation matrix, pairs, rolling correlations
│   ├── insights.py            # findings text and figures
│   └── pipeline.py            # orchestration
├── dashboard/app.py           # Streamlit dashboard (reads data/processed)
├── tests/                     # pytest: metrics, pipeline, dashboard
├── data/processed/            # outputs the dashboard reads (committed)
└── notebooks/archive_v1_us/   # original 5-stock US analysis (v1)
```

## Methodology

| Choice | Detail |
| --- | --- |
| Universe | Nifty 50 constituents effective 30 Sep 2026 (BSE replaced Wipro in the September rebalance) |
| Benchmark | Nippon India Nifty BeES ETF, adjusted close. Unlike the `^NSEI` price index it includes dividends, so it compares like-for-like with total-return stock figures |
| Returns | Simple daily returns of the adjusted close, computed per stock before aligning dates |
| Annualisation | 252 trading days |
| Risk-free rate | 6% a year, a constant approximation of the 91-day T-bill average over the period |
| Partial history | Stocks covering under 95% of the benchmark's trading days (recent listings, demergers) are kept, flagged, compared with the benchmark over their own dates, and left out of rankings. Stocks with under 252 days are excluded |
| Validation | Compounded returns reconcile with first/last adjusted close; latest SMAs and one correlation pair are recomputed independently; correlation matrix symmetry and bounds are checked |

## Limitations

- **Survivorship bias:** the universe is *today's* Nifty 50, so stocks that dropped out over the five years are missing and results lean optimistic.
- A constant risk-free rate shifts every Sharpe, Sortino and alpha equally; rankings are unaffected.
- Yahoo Finance data is unofficial and occasionally has gaps or adjustment errors. The data-quality report flags suspicious moves for review.
- The analysis is historical and descriptive. It does not forecast prices and is not investment advice.

## Roadmap

- [x] **Phase 1 – foundations:** single config, one-command pipeline, adjusted-close returns, full risk metrics, Nifty 50 universe, tests and CI
- [ ] **Phase 2 – analyst layer:** DuckDB + SQL analysis, question-driven findings (sector rotation, drawdown recovery, correlation in sell-offs, earnings event study), question-led dashboard
- [ ] **Phase 3 – data science layer:** hypothesis tests on return anomalies, volatility forecasting (GARCH vs gradient boosting, walk-forward validation), regime detection
- [ ] **Phase 4 – presentation:** findings-first README and a short written report

## Version history

v1 analysed five US tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA) against SPY. Its notebooks are kept in `notebooks/archive_v1_us/` and the original code is at commit `0183cb9`.
