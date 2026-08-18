# Interactive Streamlit Dashboard

This dashboard presents the completed Stock Market Trend Analysis results in a read-only interactive interface.

## Run

From the project root:

```bash
streamlit run dashboard/app.py
```

## Sections

- Overview
- Performance versus SPY
- Risk and volatility
- Moving averages and descriptive trends
- Return correlation and rolling correlation with SPY
- Methodology and final insights

The dashboard reads validated files from `data/processed/`. It does not download data, run analysis scripts, modify datasets, or overwrite figures. Run the project pipeline first if the processed files are not present.

## Limitations

The dashboard presents historical, descriptive analysis. It is not a live market dashboard and does not provide predictions, trading signals, or investment advice.
