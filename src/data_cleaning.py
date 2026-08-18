"""Clean and validate the raw stock CSV files for Phase 3 EDA."""

from pathlib import Path

import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def clean_stock_data(ticker: str) -> tuple[pd.DataFrame, int]:
    """Load, clean, and validate one raw stock dataset.

    Rows with invalid dates or missing required numeric fields are removed because
    they cannot support reliable EDA. Missing values are never imputed.
    """
    input_path = RAW_DATA_DIR / f"{ticker}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Raw file not found: {input_path}")

    data = pd.read_csv(input_path)
    rows_before = len(data)

    required_columns = ["Date", *REQUIRED_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{ticker}: missing columns {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Date"])
    data = data.drop_duplicates()
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset="Date", keep="first")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data.set_index("Date")
    data.index.name = "Date"

    validate_cleaned_data(data, ticker)
    return data, rows_before


def validate_cleaned_data(data: pd.DataFrame, ticker: str = "stock") -> None:
    """Validate the structure and required fields of a cleaned dataset."""
    if data.empty:
        raise ValueError(f"{ticker}: cleaned dataset is empty")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: Date must be a DatetimeIndex")
    if data.index.isna().any():
        raise ValueError(f"{ticker}: invalid dates remain")
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates are not sorted chronologically")
    if data.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate dates remain")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{ticker}: missing required numeric columns {missing_columns}")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in REQUIRED_COLUMNS):
        raise ValueError(f"{ticker}: required columns contain non-numeric values")
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{ticker}: required fields contain missing values")


def save_cleaned_data(ticker: str, data: pd.DataFrame) -> Path:
    """Save a cleaned dataset with Date retained as the CSV index column."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"{ticker}_clean.csv"
    data.to_csv(output_path, index=True)
    return output_path


def main() -> int:
    """Clean, validate, save, and summarize all configured stock datasets."""
    failures: dict[str, str] = {}

    for ticker in TICKERS:
        try:
            cleaned_data, rows_before = clean_stock_data(ticker)
            output_path = save_cleaned_data(ticker, cleaned_data)

            saved_data = pd.read_csv(output_path, index_col="Date", parse_dates=["Date"])
            validate_cleaned_data(saved_data, ticker)
            missing_values = int(saved_data.isna().sum().sum())
            duplicate_dates = int(saved_data.index.duplicated().sum())
            print(ticker)
            print(f"Rows: {len(saved_data)} (before cleaning: {rows_before})")
            print(f"Date range: {saved_data.index.min().date()} -> {saved_data.index.max().date()}")
            print(f"Missing values: {missing_values}")
            print(f"Duplicate dates: {duplicate_dates}")
            print("Status: PASS")
            print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}\n")
        except Exception as error:
            failures[ticker] = str(error)
            print(f"{ticker}: FAILED - {error}\n")

    if failures:
        print("Data cleaning finished with errors:")
        for ticker, error in failures.items():
            print(f"- {ticker}: {error}")
        return 1

    print("All stock datasets cleaned, validated, and saved successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
