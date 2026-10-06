"""Download Yahoo Finance OHLCV history and upsert it into project MongoDB.

Yahoo Finance data is provided by a third party, can be delayed or incomplete,
and is subject to Yahoo's terms and rate limits. It is suitable for research
prototyping; verify licensing and data quality for your intended use.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC
from typing import Any

import pandas as pd

from .market_data import MarketDataManager, TradeBar

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM")
DEFAULT_PERIOD = "1y"
YAHOO_EXCHANGE_FALLBACK = "YAHOO_FINANCE"


@dataclass(frozen=True)
class IngestionReport:
    """Outcome summary for a batch Yahoo Finance download and MongoDB upsert."""

    requested_symbols: list[str]
    downloaded_bars: int
    upserted_bars: int
    successful_symbols: list[str]
    failed_symbols: dict[str, str]


def _history_to_bars(
    symbol: str,
    history: pd.DataFrame,
    *,
    exchange: str,
) -> list[TradeBar]:
    """Convert yfinance history rows into the project's common bar schema."""
    if history.empty:
        return []

    frame = history.copy()
    frame.columns = pd.Index(
        [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    )
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{symbol}: Yahoo Finance response is missing columns: "
            f"{', '.join(sorted(missing))}."
        )

    timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.assign(timestamp=timestamps).dropna(subset=["timestamp"])
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame.loc[(frame["close"] > 0) & (frame["volume"] >= 0)]

    bars: list[TradeBar] = []
    for _, row in frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        bars.append(
            TradeBar(
                timestamp=timestamp,
                symbol=symbol,
                exchange=exchange,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return bars


def download_yfinance_bars(
    tickers: list[str] | tuple[str, ...],
    *,
    period: str = DEFAULT_PERIOD,
    ticker_factory: Callable[[str], Any] | None = None,
) -> tuple[list[TradeBar], dict[str, str]]:
    """Download daily adjusted OHLCV bars; return bars and per-symbol failures."""
    symbols = list(
        dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip())
    )
    if not symbols:
        raise ValueError("Provide at least one ticker symbol.")
    if not period.strip():
        raise ValueError("period must not be empty.")

    if ticker_factory is None:
        try:
            yfinance = importlib.import_module("yfinance")  # type: ignore[reportMissingTypeStubs]
        except ImportError as error:
            raise RuntimeError(
                "yfinance is not installed. Install the project dependencies with "
                "pip install -e ."
            ) from error
        ticker_factory = getattr(yfinance, "Ticker")

    all_bars: list[TradeBar] = []
    failures: dict[str, str] = {}
    for symbol in symbols:
        try:
            ticker = ticker_factory(symbol)
            history = ticker.history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            if history.empty:
                failures[symbol] = "Yahoo Finance returned no daily history."
                continue
            metadata = getattr(ticker, "history_metadata", None) or {}
            exchange = (
                metadata.get("exchangeName")
                or metadata.get("fullExchangeName")
                or metadata.get("exchange")
                or YAHOO_EXCHANGE_FALLBACK
            )
            symbol_bars = _history_to_bars(symbol, history, exchange=str(exchange))
            if not symbol_bars:
                failures[symbol] = "No valid OHLCV rows remained after validation."
                continue
            all_bars.extend(symbol_bars)
        except Exception as error:
            failures[symbol] = f"{type(error).__name__}: {error}"

    return all_bars, failures


def ingest_yfinance_to_mongodb(
    tickers: list[str] | tuple[str, ...],
    *,
    period: str = DEFAULT_PERIOD,
    ticker_factory: Callable[[str], Any] | None = None,
    manager: MarketDataManager | None = None,
) -> IngestionReport:
    """Download daily histories and upsert them into the configured MongoDB.

    When a manager is injected, ownership stays with the caller. Otherwise, a
    manager is created from the project's .env configuration and closed here.
    """
    symbols = list(
        dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip())
    )
    bars, failures = download_yfinance_bars(
        symbols,
        period=period,
        ticker_factory=ticker_factory,
    )
    if not bars:
        details = "; ".join(
            f"{symbol}: {message}" for symbol, message in failures.items()
        )
        raise RuntimeError(f"No market bars were downloaded. {details}")

    owns_manager = manager is None
    market_manager = manager or MarketDataManager()
    try:
        upserted = market_manager.insert_bars(bars)
    finally:
        if owns_manager:
            market_manager.close()

    successful_symbols = sorted({bar.symbol for bar in bars})
    return IngestionReport(
        requested_symbols=symbols,
        downloaded_bars=len(bars),
        upserted_bars=upserted,
        successful_symbols=successful_symbols,
        failed_symbols=failures,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download adjusted daily OHLCV from Yahoo Finance and upsert it into "
            "the MongoDB database configured in .env."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Ticker symbols (default: a diversified set of eight large U.S. companies).",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Yahoo Finance history period, e.g. 6mo, 1y, 5y (default: 1y).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = ingest_yfinance_to_mongodb(args.tickers, period=args.period)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(report), indent=2))
    return 1 if report.failed_symbols else 0


if __name__ == "__main__":
    raise SystemExit(main())
