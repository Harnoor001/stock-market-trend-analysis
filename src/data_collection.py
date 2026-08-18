"""Download historical stock data from Yahoo Finance through yFinance."""

from pathlib import Path
import time

import pandas as pd
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2021-08-16"
END_DATE = "2026-08-16"  # yFinance treats the end date as exclusive.
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DOWNLOAD_ATTEMPTS = 3


def _prepare_downloaded_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert one yFinance result into the project's raw CSV format."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)

    data = data.reset_index()
    date_column = "Date" if "Date" in data.columns else "Datetime"
    data = data.rename(columns={date_column: "Date"})
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return data[[column for column in EXPECTED_COLUMNS if column in data.columns]]


def validate_stock_data(data: pd.DataFrame) -> None:
    """Raise a clear error when downloaded data has an unexpected structure."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    if data.empty:
        raise ValueError("Downloaded dataset is empty")
    if data["Date"].isna().any():
        raise ValueError("One or more dates are invalid")
    if not data["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted chronologically")
    if data.duplicated().any():
        raise ValueError("Dataset contains completely duplicated rows")


def download_stock_data(ticker: str) -> pd.DataFrame:
    """Download and validate daily OHLCV data for one ticker."""
    downloaded = pd.DataFrame()
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            downloaded = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"yFinance request failed after {DOWNLOAD_ATTEMPTS} attempts: {error}") from error

        if not downloaded.empty:
            break
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(2)

    if downloaded.empty:
        raise ValueError(f"No data returned by yFinance after {DOWNLOAD_ATTEMPTS} attempts")

    data = _prepare_downloaded_data(downloaded, ticker)
    validate_stock_data(data)
    return data


def main() -> int:
    """Download all configured tickers and save them as raw CSV files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}

    for ticker in TICKERS:
        try:
            data = download_stock_data(ticker)
            output_path = RAW_DATA_DIR / f"{ticker}.csv"
            data.to_csv(output_path, index=False)

            saved_data = pd.read_csv(output_path, parse_dates=["Date"])
            validate_stock_data(saved_data)
            print(f"{ticker}: {len(saved_data)} rows collected")
        except Exception as error:  # Keep other tickers running if one request fails.
            failures[ticker] = str(error)
            print(f"{ticker}: download failed - {error}")

    if failures:
        print("\nData collection finished with errors:")
        for ticker, error in failures.items():
            print(f"- {ticker}: {error}")
        return 1

    print(f"\nSaved raw CSV files to: {RAW_DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
