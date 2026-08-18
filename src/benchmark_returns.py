"""Calculate SPY daily returns using the existing Phase 4 convention."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def calculate_spy_returns() -> pd.DataFrame:
    """Load SPY_clean.csv and add an intentionally missing first return."""
    path = PROCESSED_DATA_DIR / "SPY_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"SPY cleaned file not found: {path}")
    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty or not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError("SPY: invalid cleaned dates")
    data["Daily Return"] = data["Close"].pct_change()
    data["Cumulative Growth"] = (1 + data["Daily Return"].fillna(0)).cumprod()
    data["Cumulative Return"] = data["Cumulative Growth"] - 1
    if not pd.isna(data["Daily Return"].iloc[0]):
        raise ValueError("SPY: first daily return must be NaN")
    returns = data["Daily Return"].iloc[1:]
    if returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError("SPY: invalid daily returns")
    return data


def main() -> int:
    """Calculate and save SPY returns without modifying stock return files."""
    try:
        data = calculate_spy_returns()
        output = PROCESSED_DATA_DIR / "SPY_returns.csv"
        data[["Close", "Daily Return", "Cumulative Growth", "Cumulative Return"]].to_csv(output, index=True)
        print(f"SPY: {len(data)} rows with returns calculated")
        print("Return sanity check: PASS")
        print("Saved: data/processed/SPY_returns.csv")
        return 0
    except Exception as error:
        print(f"Benchmark returns failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
