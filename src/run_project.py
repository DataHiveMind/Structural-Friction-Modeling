"""Run the end-to-end market-data refresh and dashboard workflow.

From the repository root, run::

    python -m src.run_project

This downloads daily OHLCV for the default ticker basket, upserts it to the
MongoDB configured in .env, and creates a multi-company dashboard and summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from pymongo.errors import PyMongoError

    from main import run_analysis

    from .yfinance_ingest import (
        DEFAULT_PERIOD,
        DEFAULT_TICKERS,
        ingest_yfinance_to_mongodb,
    )
except ModuleNotFoundError as error:
    if error.name in {"pymongo", "dotenv", "pandas", "numpy", "matplotlib"}:
        raise SystemExit(
            f"Missing project dependency {error.name!r} in this Python environment. "
            "From the repository root, run `python -m pip install -e .`, "
            "then retry `python -m src.run_project`."
        ) from error
    raise


def run_project(
    *,
    tickers: list[str] | tuple[str, ...] = DEFAULT_TICKERS,
    period: str = DEFAULT_PERIOD,
    output_dir: str | Path = "outputs",
    skip_download: bool = False,
) -> dict[str, Any]:
    """Refresh MongoDB market bars and generate the project overview dashboard."""
    ingestion = None
    if not skip_download:
        ingestion = ingest_yfinance_to_mongodb(tickers, period=period)

    analysis = run_analysis(
        output_dir=output_dir,
        source="mongodb",
    )
    return {
        "ingestion": asdict(ingestion) if ingestion is not None else None,
        "analysis": analysis,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh real daily company bars from Yahoo Finance into the .env "
            "MongoDB and build the multi-company project dashboard."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Ticker symbols to refresh (default: the project's eight-company basket).",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Yahoo Finance period such as 6mo, 1y, or 5y (default: 1y).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for market_dashboard.png and market_summary.json.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse bars already in MongoDB and only rebuild the dashboard.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = run_project(
            tickers=args.tickers,
            period=args.period,
            output_dir=args.output_dir,
            skip_download=args.skip_download,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        PyMongoError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
