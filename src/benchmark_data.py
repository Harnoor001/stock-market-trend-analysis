"""Download comparable daily SPY benchmark data for the existing stock period."""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def determine_common_period() -> tuple[str, str]:
    """Determine the stock analysis period from existing cleaned datasets."""
    dates = []
    for ticker in TICKERS:
        path = PROCESSED_DATA_DIR / f"{ticker}_clean.csv"
        if not path.exists():
            raise FileNotFoundError(f"Existing cleaned file not found: {path}")
        data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"{ticker}: invalid cleaned dataset")
        dates.extend([data.index.min(), data.index.max()])
    start = min(dates).date()
    # yFinance treats end as exclusive, so include the latest stock date.
    end = max(dates).date() + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def prepare_downloaded_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert a yFinance response into the project's raw CSV columns."""
    if isinstance(data.columns, pd.MultiIndex):
        if "SPY" in data.columns.get_level_values(-1):
            data = data.xs("SPY", axis=1, level=-1)
        elif "SPY" in data.columns.get_level_values(0):
            data = data.xs("SPY", axis=1, level=0)
    data = data.reset_index()
    date_column = "Date" if "Date" in data.columns else "Datetime"
    data = data.rename(columns={date_column: "Date"})
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return data[[column for column in EXPECTED_COLUMNS if column in data.columns]]


def validate_benchmark_data(data: pd.DataFrame) -> None:
    """Validate the downloaded SPY structure before saving it."""
    missing = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"SPY: missing columns {missing}")
    if data.empty or data["Date"].isna().any():
        raise ValueError("SPY: empty data or invalid dates")
    if not data["Date"].is_monotonic_increasing or data["Date"].duplicated().any():
        raise ValueError("SPY: dates are not sorted and unique")
    numeric = data[EXPECTED_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.map(lambda value: pd.api.types.is_number(value)).all().all():
        raise ValueError("SPY: OHLCV fields contain invalid values")


def download_spy() -> tuple[pd.DataFrame, str, str]:
    """Download SPY using the common stock-data period."""
    start, end = determine_common_period()
    downloaded = yf.download(
        "SPY", start=start, end=end, interval="1d", auto_adjust=False,
        progress=False, group_by="column", threads=False,
    )
    if downloaded.empty:
        raise ValueError("SPY: yFinance returned no data")
    data = prepare_downloaded_data(downloaded)
    validate_benchmark_data(data)
    return data, start, end


def main() -> int:
    """Download and save SPY raw data without touching stock files."""
    try:
        data, start, end = download_spy()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output = RAW_DATA_DIR / "SPY.csv"
        data.to_csv(output, index=False)
        print(f"SPY: {len(data)} rows collected ({start} to before {end})")
        print("Saved: data/raw/SPY.csv")
        return 0
    except Exception as error:
        print(f"Benchmark data collection failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
