"""Analyze real OHLCV/quote CSV data and export a market-quality dashboard.

Example::

    python main.py --input data/raw/market_bars.csv --symbol ABC --output-dir outputs

Required CSV columns are ``timestamp`` and ``close``. Optional columns include
``symbol``, ``exchange``, ``open``, ``high``, ``low``, ``volume``, ``bid_price``,
``ask_price``, ``bid_size``, ``ask_size``, ``effective_spread``, and
``market_depth``. An optional halt-events CSV can be supplied with ``--halts``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.db_client import MongoDBClient, MongoDBConfig

REQUIRED_COLUMNS = {"timestamp", "close"}
NUMERIC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "bid_depth",
    "ask_depth",
    "market_depth",
    "effective_spread",
    "spread",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV header variations to lowercase snake case."""
    frame = frame.copy()
    normalized_columns = {
        column: str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    }
    frame = frame.rename(columns=normalized_columns)
    aliases = {"date": "timestamp", "datetime": "timestamp"}
    return frame.rename(
        columns={name: target for name, target in aliases.items() if name in frame}
    )


def load_market_data(
    input_path: str | Path,
    *,
    symbol: str | None = None,
    exchange: str | None = None,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Load and validate one market series from a CSV file.

    Timestamps are interpreted as UTC when timezone-naive and converted to UTC
    when timezone-aware. Returns and rolling volatility are computed per bar;
    volatility is not annualized because the bar frequency is user-specific.
    """
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2 bars.")

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market-data CSV not found: {path}")

    return prepare_market_data(
        pd.read_csv(path),
        symbol=symbol,
        exchange=exchange,
        rolling_window=rolling_window,
    )


def prepare_market_data(
    data: pd.DataFrame,
    *,
    symbol: str | None = None,
    exchange: str | None = None,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Validate and derive metrics for loaded bar records."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2 bars.")

    frame = _normalize_columns(data)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        expected = ", ".join(sorted(REQUIRED_COLUMNS))
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Required columns are: {expected}."
        )

    if "symbol" in frame.columns:
        symbols = frame["symbol"].dropna().astype(str).unique()
        if symbol is not None:
            frame = frame.loc[frame["symbol"].astype(str) == symbol].copy()
        elif len(symbols) > 1:
            raise ValueError("CSV contains multiple symbols; select one with --symbol.")
    elif symbol is not None:
        raise ValueError("--symbol was supplied, but the CSV has no symbol column.")

    if "exchange" in frame.columns:
        exchanges = frame["exchange"].dropna().astype(str).unique()
        if exchange is not None:
            frame = frame.loc[frame["exchange"].astype(str) == exchange].copy()
        elif len(exchanges) > 1:
            raise ValueError(
                "CSV contains multiple exchanges; select one with --exchange."
            )
    elif exchange is not None:
        raise ValueError("--exchange was supplied, but the CSV has no exchange column.")

    if frame.empty:
        raise ValueError("No rows remain after filtering by symbol and exchange.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["timestamp", "close"])
    frame = frame.loc[np.isfinite(frame["close"]) & (frame["close"] > 0)].copy()
    if "volume" in frame:
        frame = frame.loc[frame["volume"].isna() | (frame["volume"] >= 0)].copy()
    if frame.empty:
        raise ValueError("No valid rows remain after timestamp and price validation.")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        duplicates = int(frame["timestamp"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate timestamps for the selected series. "
            "Aggregate the input to one row per bar timestamp first."
        )

    if "effective_spread" not in frame:
        if "spread" in frame:
            frame["effective_spread"] = frame["spread"]
        elif {"bid_price", "ask_price"}.issubset(frame.columns):
            frame["effective_spread"] = frame["ask_price"] - frame["bid_price"]
    if "effective_spread" in frame:
        invalid_spread = frame["effective_spread"] < 0
        frame.loc[invalid_spread, "effective_spread"] = np.nan

    if "market_depth" not in frame:
        if {"bid_depth", "ask_depth"}.issubset(frame.columns):
            frame["market_depth"] = frame["bid_depth"] + frame["ask_depth"]
        elif {"bid_size", "ask_size"}.issubset(frame.columns):
            frame["market_depth"] = frame["bid_size"] + frame["ask_size"]

    frame["return"] = frame["close"].pct_change(fill_method=None)
    frame["rolling_volatility"] = (
        frame["return"]
        .rolling(
            window=rolling_window,
            min_periods=2,
        )
        .std()
    )
    return frame


def load_market_data_from_mongodb(
    *,
    symbol: str | None = None,
    exchange: str | None = None,
    database_name: str | None = None,
    bars_collection: str = "market_bars",
    quotes_collection: str = "market_quotes",
    halts_collection: str = "halt_events",
    rolling_window: int = 20,
    limit: int = 100_000,
    client: MongoClient[Any] | None = None,
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Read actual bar, quote, and halt records from the configured MongoDB.

    Connection URI and default database are resolved by ``MongoDBConfig`` from
    process variables and the project ``.env``. This function pings the server
    before querying and closes its client on completion.
    """
    if limit < 1:
        raise ValueError("MongoDB query limit must be positive.")

    config = MongoDBConfig.from_env()
    if database_name is not None:
        config = MongoDBConfig(config.uri, database_name)

    with MongoDBClient(config=config, client=client) as db_client:
        db_client.ping()
        database = db_client.database
        series_filter: dict[str, str] = {}
        if exchange is not None:
            series_filter["exchange"] = exchange

        bars_collection_handle = database[bars_collection]
        if symbol is None:
            symbols = sorted(
                {
                    str(value).strip()
                    for value in bars_collection_handle.distinct(
                        "symbol",
                        filter=series_filter,
                    )
                    if value is not None and str(value).strip()
                }
            )
            if not symbols:
                raise ValueError(
                    f"No symbols found in MongoDB collection {bars_collection!r}. "
                    "Check the configured database and ingest market bars first."
                )
            if len(symbols) > 1:
                available_symbols = ", ".join(symbols[:20])
                extra_count = len(symbols) - 20
                suffix = f", and {extra_count} more" if extra_count > 0 else ""
                raise ValueError(
                    "More than one symbol is available in MongoDB. Choose one "
                    f"with --symbol. Available: {available_symbols}{suffix}."
                )
            symbol = symbols[0]

        series_filter["symbol"] = symbol
        bar_documents = list(
            bars_collection_handle.find(series_filter, {"_id": 0})
            .sort("timestamp", 1)
            .limit(limit)
        )
        if not bar_documents:
            raise ValueError(
                f"No market bars found for {symbol} in collection "
                f"{bars_collection!r}. Check the symbol, exchange, database, "
                "and collection settings."
            )

        bars = _normalize_columns(pd.DataFrame(bar_documents))
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp")
        if bars.empty:
            raise ValueError("MongoDB returned bars without valid timestamps.")

        first_timestamp = cast(pd.Timestamp, bars["timestamp"].iloc[0])
        last_timestamp = cast(pd.Timestamp, bars["timestamp"].iloc[-1])
        quote_filter: dict[str, Any] = {
            **series_filter,
            "timestamp": {
                "$gte": first_timestamp.to_pydatetime(),
                "$lte": last_timestamp.to_pydatetime(),
            },
        }
        quote_documents = list(
            database[quotes_collection]
            .find(quote_filter, {"_id": 0})
            .sort("timestamp", 1)
            .limit(limit)
        )
        if quote_documents:
            quotes = _normalize_columns(pd.DataFrame(quote_documents))
            quotes["timestamp"] = pd.to_datetime(
                quotes["timestamp"], utc=True, errors="coerce"
            )
            quote_aliases = {
                "spread": "effective_spread",
                "bid_size": "bid_depth",
                "ask_size": "ask_depth",
            }
            quotes = quotes.rename(
                columns={
                    name: target
                    for name, target in quote_aliases.items()
                    if name in quotes and target not in quotes
                }
            )
            quotes = quotes.dropna(subset=["timestamp"])
            if not quotes.empty:
                # The Mongo schema stores quotes as observations. Aggregate them
                # to each bar timestamp's minute before joining the bar series.
                quote_metrics = [
                    column
                    for column in ("effective_spread", "bid_depth", "ask_depth")
                    if column in quotes
                ]
                if quote_metrics:
                    quotes["timestamp"] = quotes["timestamp"].dt.floor("min")
                    quote_agg = quotes.groupby("timestamp", as_index=False)[
                        quote_metrics
                    ].mean()
                    bars["_quote_minute"] = bars["timestamp"].dt.floor("min")
                    quote_agg = quote_agg.rename(columns={"timestamp": "_quote_minute"})
                    bars = bars.merge(
                        quote_agg,
                        on="_quote_minute",
                        how="left",
                        suffixes=("", "_quote"),
                    ).drop(columns="_quote_minute")
                    if "effective_spread_quote" in bars:
                        if "effective_spread" in bars:
                            bars["effective_spread"] = bars["effective_spread"].fillna(
                                bars["effective_spread_quote"]
                            )
                            bars = bars.drop(columns="effective_spread_quote")
                        else:
                            bars = bars.rename(
                                columns={"effective_spread_quote": "effective_spread"}
                            )
                    if {"bid_depth", "ask_depth"}.issubset(bars.columns):
                        bars["market_depth"] = bars["bid_depth"] + bars["ask_depth"]

        halt_filter: dict[str, str] = {"symbol": symbol}
        if exchange is not None:
            halt_filter["venue"] = exchange
        halt_documents = list(
            database[halts_collection]
            .find(halt_filter, {"_id": 0})
            .sort("halt_start", 1)
            .limit(limit)
        )

    halt_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for event in halt_documents:
        start = pd.to_datetime(event.get("halt_start"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        end = pd.to_datetime(event.get("halt_end"), utc=True, errors="coerce")
        if pd.isna(end):
            end = last_timestamp
        if end < start:
            raise ValueError(f"Halt end {end} occurs before halt start {start}.")
        halt_windows.append((cast(pd.Timestamp, start), cast(pd.Timestamp, end)))

    result = prepare_market_data(
        bars,
        symbol=symbol,
        exchange=exchange,
        rolling_window=rolling_window,
    )
    return result, halt_windows


def load_all_market_data_from_mongodb(
    *,
    exchange: str | None = None,
    database_name: str | None = None,
    bars_collection: str = "market_bars",
    rolling_window: int = 20,
    limit_per_symbol: int = 100_000,
) -> dict[str, pd.DataFrame]:
    """Load every available ticker's bars from the configured MongoDB."""
    if limit_per_symbol < 1:
        raise ValueError("MongoDB query limit must be positive.")

    config = MongoDBConfig.from_env()
    if database_name is not None:
        config = MongoDBConfig(config.uri, database_name)

    with MongoDBClient(config=config) as db_client:
        db_client.ping()
        collection = db_client.database[bars_collection]
        symbol_filter = {"exchange": exchange} if exchange is not None else {}
        symbols = sorted(
            {
                str(value).strip()
                for value in collection.distinct("symbol", filter=symbol_filter)
                if value is not None and str(value).strip()
            }
        )
        if not symbols:
            raise ValueError(
                f"No symbols found in MongoDB collection {bars_collection!r}. "
                "Check the configured database and ingest market bars first."
            )

        company_data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            query: dict[str, str] = {"symbol": symbol}
            if exchange is not None:
                query["exchange"] = exchange
            documents = list(
                collection.find(query, {"_id": 0})
                .sort("timestamp", 1)
                .limit(limit_per_symbol)
            )
            if not documents:
                continue
            bars = _normalize_columns(pd.DataFrame(documents))
            company_data[symbol] = prepare_market_data(
                bars,
                symbol=symbol,
                exchange=exchange,
                rolling_window=rolling_window,
            )

    if not company_data:
        raise ValueError("MongoDB had symbols but no valid market bars to chart.")
    return company_data


def load_halt_events(
    halt_path: str | Path | None,
    *,
    symbol: str | None,
    exchange: str | None,
    last_timestamp: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Load halt intervals from an optional CSV with halt_start/halt_end columns."""
    if halt_path is None:
        return []

    path = Path(halt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Halt-events CSV not found: {path}")
    events = _normalize_columns(pd.read_csv(path))
    if "halt_start" not in events.columns:
        raise ValueError("Halt-events CSV must include a halt_start column.")

    for column, value in (("symbol", symbol), ("exchange", exchange)):
        if value is not None and column in events.columns:
            events = events.loc[events[column].astype(str) == value]

    events["halt_start"] = pd.to_datetime(
        events["halt_start"], utc=True, errors="coerce"
    )
    if "halt_end" not in events.columns:
        events["halt_end"] = pd.NaT
    events["halt_end"] = pd.to_datetime(events["halt_end"], utc=True, errors="coerce")
    events = events.dropna(subset=["halt_start"])

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_value, end_value in events[["halt_start", "halt_end"]].itertuples(
        index=False,
        name=None,
    ):
        start = cast(pd.Timestamp, start_value)
        end = last_timestamp if pd.isna(end_value) else cast(pd.Timestamp, end_value)
        if end < start:
            raise ValueError(f"Halt end {end} occurs before halt start {start}.")
        windows.append((start, end))
    return windows


def _row_halt_windows(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert a row-level halt flag into contiguous chart intervals."""
    flag_column = next(
        (column for column in ("is_halt_interval", "halted") if column in frame),
        None,
    )
    if flag_column is None:
        return []

    values = frame[flag_column]
    if values.dtype == bool:
        flags = values.fillna(False).to_numpy()
    else:
        flags = (
            values.astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "halted"})
            .to_numpy()
        )

    positions = np.flatnonzero(flags)
    if positions.size == 0:
        return []
    groups = np.split(positions, np.where(np.diff(positions) > 1)[0] + 1)
    timestamp_ns = frame["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    timestamp_deltas_ns = np.diff(timestamp_ns)
    positive_deltas_ns = timestamp_deltas_ns[timestamp_deltas_ns > 0]
    median_step_ns = (
        int(np.median(positive_deltas_ns))
        if positive_deltas_ns.size
        else int(pd.Timedelta(minutes=1).value)
    )
    timestamp_step = pd.Timedelta(median_step_ns, unit="ns")

    return [
        (
            cast(pd.Timestamp, frame.loc[int(group[0]), "timestamp"]),
            cast(pd.Timestamp, frame.loc[int(group[-1]), "timestamp"]) + timestamp_step,
        )
        for group in groups
    ]


def create_dashboard(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    symbol: str | None,
    exchange: str | None,
    halt_windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> Path:
    """Save a four-panel price, volume, volatility, and liquidity dashboard."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1, 1.2, 1.3]},
    )
    title_parts = [part for part in (symbol, exchange) if part]
    title = " · ".join(title_parts) if title_parts else "Market data"

    price_axis, volume_axis, volatility_axis, liquidity_axis = axes
    price_axis.plot(frame["timestamp"], frame["close"], color="#155e75", linewidth=1.5)
    if {"low", "high"}.issubset(frame.columns):
        price_axis.fill_between(
            frame["timestamp"],
            frame["low"],
            frame["high"],
            color="#67e8f9",
            alpha=0.18,
            label="Bar range",
        )
    price_axis.set_ylabel("Price")
    price_axis.set_title(f"{title}: real market-data overview")
    price_axis.grid(alpha=0.22)

    if "volume" in frame and frame["volume"].notna().any():
        volume_axis.fill_between(
            frame["timestamp"],
            frame["volume"].fillna(0),
            color="#6366f1",
            alpha=0.45,
            step="mid",
        )
        volume_axis.set_ylabel("Volume")
        volume_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        volume_axis.text(
            0.5,
            0.5,
            "No volume column supplied",
            ha="center",
            va="center",
            transform=volume_axis.transAxes,
        )
        volume_axis.set_ylabel("Volume")
    volume_axis.grid(alpha=0.22)

    volatility_axis.plot(
        frame["timestamp"],
        frame["rolling_volatility"],
        color="#7c3aed",
        linewidth=1.2,
    )
    volatility_axis.set_ylabel("Rolling σ\n(per bar)")
    volatility_axis.grid(alpha=0.22)

    has_liquidity_metric = False
    if "effective_spread" in frame and frame["effective_spread"].notna().any():
        liquidity_axis.plot(
            frame["timestamp"],
            frame["effective_spread"],
            color="#dc2626",
            linewidth=1.1,
            label="Quoted spread",
        )
        liquidity_axis.set_ylabel("Spread")
        has_liquidity_metric = True
    if "market_depth" in frame and frame["market_depth"].notna().any():
        depth_axis = liquidity_axis.twinx()
        depth_axis.plot(
            frame["timestamp"],
            frame["market_depth"],
            color="#059669",
            linewidth=1.1,
            alpha=0.9,
            label="Top-of-book depth",
        )
        depth_axis.set_ylabel("Depth (bid + ask size)")
        lines, labels = liquidity_axis.get_legend_handles_labels()
        depth_lines, depth_labels = depth_axis.get_legend_handles_labels()
        liquidity_axis.legend(
            lines + depth_lines, labels + depth_labels, loc="upper right"
        )
        has_liquidity_metric = True
    if not has_liquidity_metric:
        liquidity_axis.text(
            0.5,
            0.5,
            "No quote spread/depth supplied; not inferred from trade volume",
            ha="center",
            va="center",
            transform=liquidity_axis.transAxes,
        )
        liquidity_axis.set_ylabel("Liquidity")
    liquidity_axis.grid(alpha=0.22)

    windows = halt_windows if halt_windows is not None else _row_halt_windows(frame)
    for index, (start, end) in enumerate(windows):
        price_axis.axvspan(
            start,
            end,
            color="#f59e0b",
            alpha=0.2,
            label="Halt window" if index == 0 else None,
        )
        for axis in axes[1:]:
            axis.axvspan(start, end, color="#f59e0b", alpha=0.12)
    if windows:
        price_axis.legend(loc="upper left")

    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
    )
    axes[-1].set_xlabel("Timestamp (UTC)")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def create_company_overview(
    company_data: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Plot normalized performance, rolling volatility, and volume per ticker."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for index, (symbol, frame) in enumerate(sorted(company_data.items())):
        color = colors(index % 10)
        baseline = float(frame["close"].iloc[0])
        relative_price = frame["close"] / baseline * 100
        axes[0].plot(
            frame["timestamp"],
            relative_price,
            label=symbol,
            color=color,
            linewidth=1.5,
        )
        axes[1].plot(
            frame["timestamp"],
            frame["rolling_volatility"] * 100,
            label=symbol,
            color=color,
            linewidth=1.1,
        )
        if "volume" in frame and frame["volume"].notna().any():
            axes[2].plot(
                frame["timestamp"],
                frame["volume"] / 1_000_000,
                label=symbol,
                color=color,
                linewidth=1.0,
                alpha=0.85,
            )

    axes[0].axhline(100, color="#475569", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Indexed close (start = 100)")
    axes[0].set_title("Company market overview — MongoDB daily OHLCV")
    axes[0].legend(ncol=4, loc="upper left", fontsize=9)
    axes[1].set_ylabel("Rolling volatility (% / bar)")
    axes[2].set_ylabel("Volume (millions)")
    axes[2].set_xlabel("Timestamp (UTC)")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def summarize_company_overview(
    company_data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Return comparable per-ticker summary statistics."""
    summaries: dict[str, dict[str, Any]] = {}
    for symbol, frame in sorted(company_data.items()):
        close = frame["close"]
        summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "start_utc": frame["timestamp"].iloc[0].isoformat(),
            "end_utc": frame["timestamp"].iloc[-1].isoformat(),
            "first_close": float(close.iloc[0]),
            "last_close": float(close.iloc[-1]),
            "close_change_percent": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            "mean_rolling_volatility_per_bar": _finite_mean(
                frame["rolling_volatility"]
            ),
        }
        if "volume" in frame:
            summary["total_volume"] = float(frame["volume"].sum())
        if "exchange" in frame and frame["exchange"].nunique() == 1:
            summary["exchange"] = str(frame["exchange"].iloc[0])
        summaries[symbol] = summary
    return summaries


def summarize_market_data(
    frame: pd.DataFrame,
    *,
    symbol: str | None,
    exchange: str | None,
    halt_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    """Create machine-readable summary statistics for the selected series."""
    close = frame["close"]
    summary: dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "rows": int(len(frame)),
        "start_utc": frame["timestamp"].iloc[0].isoformat(),
        "end_utc": frame["timestamp"].iloc[-1].isoformat(),
        "first_close": float(close.iloc[0]),
        "last_close": float(close.iloc[-1]),
        "close_change_percent": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "mean_rolling_volatility_per_bar": _finite_mean(frame["rolling_volatility"]),
        "halt_event_count": len(halt_windows),
        "halt_duration_minutes": float(
            sum((end - start).total_seconds() for start, end in halt_windows) / 60
        ),
    }
    for column, name, operation in (
        ("volume", "total_volume", "sum"),
        ("effective_spread", "mean_spread", "mean"),
        ("market_depth", "median_top_of_book_depth", "median"),
    ):
        if column in frame:
            values = frame[column].dropna()
            if not values.empty:
                value = (
                    values.sum() if operation == "sum" else getattr(values, operation)()
                )
                summary[name] = float(value)
    return summary


def _finite_mean(values: pd.Series) -> float | None:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else None


def run_analysis(
    input_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
    *,
    source: str = "csv",
    symbol: str | None = None,
    exchange: str | None = None,
    halt_path: str | Path | None = None,
    rolling_window: int = 20,
    database_name: str | None = None,
    bars_collection: str = "market_bars",
    quotes_collection: str = "market_quotes",
    halts_collection: str = "halt_events",
    query_limit: int = 100_000,
) -> dict[str, Any]:
    """Analyze CSV or .env-configured MongoDB data and save its dashboard."""
    if source == "mongodb":
        if symbol is None:
            company_data = load_all_market_data_from_mongodb(
                exchange=exchange,
                database_name=database_name,
                bars_collection=bars_collection,
                rolling_window=rolling_window,
                limit_per_symbol=query_limit,
            )
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            chart_path = create_company_overview(
                company_data,
                output / "market_dashboard.png",
            )
            summary: dict[str, Any] = {
                "source": source,
                "database": database_name or MongoDBConfig.from_env().database_name,
                "bars_collection": bars_collection,
                "symbols": sorted(company_data),
                "companies": summarize_company_overview(company_data),
                "dashboard_png": str(chart_path),
            }
            (output / "market_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return summary

        frame, halts = load_market_data_from_mongodb(
            symbol=symbol,
            exchange=exchange,
            database_name=database_name,
            bars_collection=bars_collection,
            quotes_collection=quotes_collection,
            halts_collection=halts_collection,
            rolling_window=rolling_window,
            limit=query_limit,
        )
    elif source == "csv":
        if input_path is None:
            raise ValueError("--input is required when --source csv is used.")
        frame = load_market_data(
            input_path,
            symbol=symbol,
            exchange=exchange,
            rolling_window=rolling_window,
        )
        halts = load_halt_events(
            halt_path,
            symbol=symbol,
            exchange=exchange,
            last_timestamp=frame["timestamp"].iloc[-1],
        )
    else:
        raise ValueError("source must be either 'mongodb' or 'csv'.")

    selected_symbol = symbol
    selected_exchange = exchange
    if selected_symbol is None and "symbol" in frame and frame["symbol"].nunique() == 1:
        selected_symbol = str(frame["symbol"].iloc[0])
    if (
        selected_exchange is None
        and "exchange" in frame
        and frame["exchange"].nunique() == 1
    ):
        selected_exchange = str(frame["exchange"].iloc[0])

    if not halts:
        halts = _row_halt_windows(frame)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart_path = create_dashboard(
        frame,
        output / "market_dashboard.png",
        symbol=selected_symbol,
        exchange=selected_exchange,
        halt_windows=halts,
    )
    summary = summarize_market_data(
        frame,
        symbol=selected_symbol,
        exchange=selected_exchange,
        halt_windows=halts,
    )
    summary["source"] = source
    if input_path is not None:
        summary["source_csv"] = str(Path(input_path))
    if source == "mongodb":
        summary["database"] = database_name or MongoDBConfig.from_env().database_name
        summary["bars_collection"] = bars_collection
    summary["dashboard_png"] = str(chart_path)
    summary_path = output / "market_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze real CSV or MongoDB market data and save a dashboard.",
        epilog=(
            "MongoDB defaults to the project .env MONGODB_URI/database and reads "
            "market_bars, market_quotes, and halt_events. CSV bars require "
            "timestamp,close; optional halt CSV columns: halt_start,halt_end."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("mongodb", "csv"),
        default="mongodb",
        help="Data source (default: mongodb using the project .env).",
    )
    parser.add_argument(
        "--input", help="Path to market-bars CSV when using --source csv."
    )
    parser.add_argument(
        "--symbol",
        help=(
            "Select a symbol; without it, MongoDB mode compares all available "
            "symbols in one dashboard."
        ),
    )
    parser.add_argument("--exchange", help="Filter a multi-exchange CSV.")
    parser.add_argument(
        "--halts", help="Optional CSV with halt_start and halt_end columns."
    )
    parser.add_argument(
        "--database",
        help="Override the .env MongoDB database name; URI still comes from .env.",
    )
    parser.add_argument("--bars-collection", default="market_bars")
    parser.add_argument("--quotes-collection", default="market_quotes")
    parser.add_argument("--halts-collection", default="halt_events")
    parser.add_argument("--query-limit", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Directory for market_dashboard.png and market_summary.json "
            "(default: outputs)."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Number of bars in realized-volatility window (default: 20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.input,
            args.output_dir,
            source=args.source,
            symbol=args.symbol,
            exchange=args.exchange,
            halt_path=args.halts,
            rolling_window=args.rolling_window,
            database_name=args.database,
            bars_collection=args.bars_collection,
            quotes_collection=args.quotes_collection,
            halts_collection=args.halts_collection,
            query_limit=args.query_limit,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        pd.errors.ParserError,
        PyMongoError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
