"""Clean and validate the additive SPY benchmark dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SPY.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_spy_data() -> tuple[pd.DataFrame, int]:
    """Clean dates and required numeric fields without imputing values."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"SPY raw file not found: {RAW_DATA_PATH}")
    data = pd.read_csv(RAW_DATA_PATH)
    rows_before = len(data)
    missing = [column for column in ["Date", *REQUIRED_COLUMNS] if column not in data.columns]
    if missing:
        raise ValueError(f"SPY: missing columns {missing}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Date"]).drop_duplicates().sort_values("Date")
    data = data.drop_duplicates(subset="Date", keep="first").dropna(subset=REQUIRED_COLUMNS)
    data = data.set_index("Date")
    data.index.name = "Date"
    validate_cleaned_spy(data)
    return data, rows_before


def validate_cleaned_spy(data: pd.DataFrame) -> None:
    """Validate the cleaned SPY Date index and numeric price fields."""
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("SPY: cleaned dataset is empty or Date is not an index")
    if data.index.isna().any() or not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError("SPY: Date index is invalid")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in REQUIRED_COLUMNS):
        raise ValueError("SPY: required columns are not numeric")
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("SPY: missing values remain in required fields")
    if not data[REQUIRED_COLUMNS].replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
        raise ValueError("SPY: infinite values remain")


def main() -> int:
    """Clean SPY and save it separately from existing stock datasets."""
    try:
        data, rows_before = clean_spy_data()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output = PROCESSED_DATA_DIR / "SPY_clean.csv"
        data.to_csv(output, index=True)
        print(f"SPY: {len(data)} rows after cleaning (before cleaning: {rows_before})")
        print(f"Date range: {data.index.min().date()} -> {data.index.max().date()}")
        print("Missing values: 0")
        print("Status: PASS")
        print("Saved: data/processed/SPY_clean.csv")
        return 0
    except Exception as error:
        print(f"Benchmark cleaning failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
