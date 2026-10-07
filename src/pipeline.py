"""Run the whole analysis end to end.

    python main.py                  # download, clean, analyse, write outputs
    python main.py --skip-download  # reuse existing data/raw files
"""

import argparse
import time

from src import correlation, data_cleaning, data_collection, insights, performance, trends
from src.config import Paths


def run(paths: Paths = Paths(), download: bool = True, figures: bool = True) -> dict:
    started = time.perf_counter()
    paths.ensure()

    if download:
        print("[1/6] Downloading prices")
        failed = data_collection.run(paths)
        if failed:
            print(f"  warning: no data for {', '.join(failed)}; continuing with the rest")
    else:
        print("[1/6] Download skipped (using existing raw files)")

    print("[2/6] Cleaning and data-quality checks")
    prices, quality = data_cleaning.run(paths)

    print("[3/6] Returns and performance metrics")
    results = performance.run(prices, quality, paths)

    print("[4/6] Moving averages and trends")
    _, results["trend_summary"] = trends.run(prices, paths)

    print("[5/6] Correlations")
    results.update(correlation.run(results["returns"], results["stock_summary"], paths))

    print("[6/6] Insights and figures")
    insights.run(results, quality, paths, figures=figures)

    print(f"Done in {time.perf_counter() - started:.1f}s. Outputs in {paths.processed}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nifty 50 market analysis pipeline")
    parser.add_argument("--skip-download", action="store_true", help="reuse files already in data/raw")
    parser.add_argument("--no-figures", action="store_true", help="skip writing PNG figures")
    args = parser.parse_args(argv)
    run(download=not args.skip_download, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
