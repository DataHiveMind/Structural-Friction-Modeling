"""
src/ingestion/market_data.py

Market data ingestion, schema enforcement, and time-alignment engine.
Manages high-frequency bars, quotes, exchange halt events, confounder time-series,
and portfolio states with unified UTC timestamp normalization in MongoDB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from .db_client import DEFAULT_DATABASE_NAME, MongoDBConfig


def _as_utc(value: datetime) -> datetime:
    """Treat naive timestamps as UTC and convert aware timestamps to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeBar:
    """Intraday OHLCV bar or aggregated trade window."""

    timestamp: datetime  # UTC
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["timestamp"] = _as_utc(doc["timestamp"])
        return doc


@dataclass(frozen=True)
class Quote:
    """Timestamped top-of-book quotes for bid-ask spread expansion analysis."""

    timestamp: datetime  # UTC
    symbol: str
    exchange: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        return (self.ask_price + self.bid_price) / 2.0

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["spread"] = self.spread
        doc["mid_price"] = self.mid_price
        doc["timestamp"] = _as_utc(doc["timestamp"])
        return doc


@dataclass(frozen=True)
class HaltEvent:
    """Exchange-level trading halt identifying the treatment interval T."""

    symbol: str
    venue: str
    halt_start: datetime  # UTC
    halt_end: datetime | None  # UTC; None if ongoing
    status: str  # e.g., 'HALTED', 'RESUMED'
    reason: str | None = None  # e.g., 'LUDP', 'NEWS_PENDING', 'VOLATILITY'

    @property
    def duration_seconds(self) -> float | None:
        if self.halt_end is not None:
            return (_as_utc(self.halt_end) - _as_utc(self.halt_start)).total_seconds()
        return None

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["duration_seconds"] = self.duration_seconds
        for field in ("halt_start", "halt_end"):
            if doc[field] is not None:
                doc[field] = _as_utc(doc[field])
        return doc


@dataclass(frozen=True)
class ConfounderRecord:
    """Time-aligned external covariates (X) to control for backdoor confounding."""

    timestamp: datetime  # UTC
    benchmark_symbol: str
    benchmark_return: float
    etf_net_flow: float | None = None
    vix_level: float | None = None
    sector_return: float | None = None
    correlated_asset_returns: dict[str, float] | None = None

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["timestamp"] = _as_utc(doc["timestamp"])
        return doc


@dataclass(frozen=True)
class PortfolioPosition:
    """Holding state for portfolio-level L-CVaR evaluation."""

    symbol: str
    weight: float
    quantity: float
    currency: str = "USD"


# ---------------------------------------------------------------------------
# MongoDB Storage & Time Alignment Manager
# ---------------------------------------------------------------------------


class MarketDataManager:
    """
    Manages data ingestion, index lifecycle, and causal dataset construction.
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        database_name: str | None = None,
        *,
        client: MongoClient[Any] | None = None,
        ensure_indexes: bool = True,
    ) -> None:
        if mongo_uri is None:
            config = MongoDBConfig.from_env()
            if database_name is not None:
                config = MongoDBConfig(config.uri, database_name)
        else:
            config = MongoDBConfig(
                mongo_uri,
                database_name or DEFAULT_DATABASE_NAME,
            )

        self.client: MongoClient[Any] = (
            client
            if client is not None
            else MongoClient(config.uri, serverSelectionTimeoutMS=5_000)
        )
        self.db: Database[Any] = self.client[config.database_name]

        self.bars: Collection = self.db["market_bars"]
        self.quotes: Collection = self.db["market_quotes"]
        self.halts: Collection = self.db["halt_events"]
        self.confounders: Collection = self.db["confounder_series"]
        self.portfolios: Collection = self.db["portfolios"]

        if ensure_indexes:
            self._ensure_indexes()

    def close(self) -> None:
        """Close the underlying MongoDB connection pool."""
        self.client.close()

    def __enter__(self) -> MarketDataManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_indexes(self) -> None:
        """Create compound indexes for fast time-range querying."""
        self.bars.create_index(
            [
                ("symbol", ASCENDING),
                ("exchange", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
        )
        self.quotes.create_index(
            [
                ("symbol", ASCENDING),
                ("exchange", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
        )
        self.halts.create_index(
            [
                ("symbol", ASCENDING),
                ("venue", ASCENDING),
                ("halt_start", ASCENDING),
            ],
            unique=True,
        )
        self.confounders.create_index(
            [("timestamp", ASCENDING), ("benchmark_symbol", ASCENDING)],
            unique=True,
        )
        self.portfolios.create_index([("portfolio_id", ASCENDING)], unique=True)

    def insert_bars(self, bars: list[TradeBar]) -> int:
        """Upsert intraday bars in batches."""
        if not bars:
            return 0
        operations = [
            UpdateOne(
                {
                    "symbol": b.symbol,
                    "exchange": b.exchange,
                    "timestamp": _as_utc(b.timestamp),
                },
                {"$set": b.to_doc()},
                upsert=True,
            )
            for b in bars
        ]
        result = self.bars.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def insert_quotes(self, quotes: list[Quote]) -> int:
        """Upsert timestamped quotes so repeated ingestion is idempotent."""
        if not quotes:
            return 0
        operations = [
            UpdateOne(
                {
                    "symbol": quote.symbol,
                    "exchange": quote.exchange,
                    "timestamp": _as_utc(quote.timestamp),
                },
                {"$set": quote.to_doc()},
                upsert=True,
            )
            for quote in quotes
        ]
        result = self.quotes.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def record_halt(self, halt: HaltEvent) -> None:
        """Upsert a trading halt event."""
        self.halts.update_one(
            {
                "symbol": halt.symbol,
                "venue": halt.venue,
                "halt_start": _as_utc(halt.halt_start),
            },
            {"$set": halt.to_doc()},
            upsert=True,
        )

    def insert_confounders(self, records: list[ConfounderRecord]) -> int:
        """Upsert exogenous macro and market covariates (X)."""
        if not records:
            return 0
        operations = [
            UpdateOne(
                {
                    "timestamp": _as_utc(r.timestamp),
                    "benchmark_symbol": r.benchmark_symbol,
                },
                {"$set": r.to_doc()},
                upsert=True,
            )
            for r in records
        ]
        result = self.confounders.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def save_portfolio(
        self,
        portfolio_id: str,
        positions: list[PortfolioPosition],
    ) -> None:
        """Persist portfolio weights for L-CVaR evaluation."""
        self.portfolios.update_one(
            {"portfolio_id": portfolio_id},
            {
                "$set": {
                    "portfolio_id": portfolio_id,
                    "updated_at": datetime.now(UTC),
                    "positions": [asdict(p) for p in positions],
                }
            },
            upsert=True,
        )

    def build_causal_dataset(
        self,
        symbol: str,
        halt_event: HaltEvent,
        pre_halt_minutes: int = 30,
        post_halt_minutes: int = 30,
        control_day_offset: int = 1,
    ) -> pd.DataFrame:
        """
        Extracts treatment (halt) and synthetic control (non-halt) windows,
        joining quotes, returns, and confounder series into an aligned DataFrame.

        Treatment (T=1): [halt_start - pre, halt_end + post]
        Control   (T=0): Identical time of day shifted by `control_day_offset` days.
        """
        if halt_event.halt_end is None:
            raise ValueError(
                "Halt event must have a defined resumption time (halt_end)."
            )
        if pre_halt_minutes < 0 or post_halt_minutes < 0:
            raise ValueError("Pre- and post-halt windows must be non-negative.")
        if control_day_offset < 1:
            raise ValueError("control_day_offset must be at least one day.")

        halt_start = _as_utc(halt_event.halt_start)
        halt_end = _as_utc(halt_event.halt_end)
        if halt_end <= halt_start:
            raise ValueError("halt_end must be later than halt_start.")

        treatment_start = halt_start - timedelta(minutes=pre_halt_minutes)
        treatment_end = halt_end + timedelta(minutes=post_halt_minutes)

        control_start = treatment_start - timedelta(days=control_day_offset)
        control_end = treatment_end - timedelta(days=control_day_offset)

        df_treat = self._extract_window_metrics(
            symbol=symbol,
            exchange=halt_event.venue,
            start=treatment_start,
            end=treatment_end,
            treatment_flag=1,
            halt_start=halt_start,
            halt_end=halt_end,
        )

        df_ctrl = self._extract_window_metrics(
            symbol=symbol,
            exchange=halt_event.venue,
            start=control_start,
            end=control_end,
            treatment_flag=0,
            halt_start=None,
            halt_end=None,
        )

        if df_treat.empty or df_ctrl.empty:
            raise ValueError(
                "Both treatment and control windows must contain market bars."
            )

        dataset = pd.concat([df_treat, df_ctrl], ignore_index=True)
        return dataset.sort_values("timestamp").reset_index(drop=True)

    def _extract_window_metrics(
        self,
        symbol: str,
        exchange: str,
        start: datetime,
        end: datetime,
        treatment_flag: int,
        halt_start: datetime | None,
        halt_end: datetime | None,
    ) -> pd.DataFrame:
        """Pulls time-aligned bars, quotes, and covariates for a window."""
        start = _as_utc(start)
        end = _as_utc(end)
        bar_cursor = self.bars.find(
            {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": {"$gte": start, "$lte": end},
            },
            {"_id": 0},
        ).sort("timestamp", ASCENDING)
        bars_df = pd.DataFrame(list(bar_cursor))

        quote_cursor = self.quotes.find(
            {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": {"$gte": start, "$lte": end},
            },
            {"_id": 0},
        ).sort("timestamp", ASCENDING)
        quotes_df = pd.DataFrame(list(quote_cursor))

        confounder_cursor = self.confounders.find(
            {"timestamp": {"$gte": start, "$lte": end}},
            {"_id": 0},
        ).sort("timestamp", ASCENDING)
        conf_df = pd.DataFrame(list(confounder_cursor))

        if bars_df.empty:
            return pd.DataFrame()

        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
        bars_df.set_index("timestamp", inplace=True)

        # Resample and aggregate quote metrics (mean spread, mean depth) to bar interval
        if not quotes_df.empty:
            quotes_df["timestamp"] = pd.to_datetime(quotes_df["timestamp"], utc=True)
            quotes_agg = (
                quotes_df.set_index("timestamp")
                .resample("1min")
                .agg(
                    {
                        "spread": "mean",
                        "bid_size": "mean",
                        "ask_size": "mean",
                    }
                )
                .rename(
                    columns={
                        "spread": "effective_spread",
                        "bid_size": "bid_depth",
                        "ask_size": "ask_depth",
                    }
                )
            )
            merged = bars_df.join(quotes_agg, how="left")
        else:
            merged = bars_df.copy()
            merged["effective_spread"] = np.nan
            merged["bid_depth"] = np.nan
            merged["ask_depth"] = np.nan

        # Join macro/confounder covariates (X)
        if not conf_df.empty:
            conf_df["timestamp"] = pd.to_datetime(conf_df["timestamp"], utc=True)
            conf_agg = conf_df.set_index("timestamp").resample("1min").last()
            merged = merged.join(conf_agg, how="left")
        else:
            merged["benchmark_return"] = np.nan
            merged["vix_level"] = np.nan
            merged["etf_net_flow"] = np.nan

        for field in (
            "benchmark_return",
            "vix_level",
            "etf_net_flow",
            "sector_return",
            "correlated_asset_returns",
        ):
            if field not in merged:
                merged[field] = np.nan

        merged.reset_index(inplace=True)

        # Derived targets and status flags
        relative_prices = merged["close"] / merged["close"].shift(1)
        merged["log_return"] = pd.Series(
            np.log(relative_prices.to_numpy()),
            index=merged.index,
        )
        merged["realized_volatility"] = (
            merged["log_return"]
            .rolling(
                window=5,
                min_periods=2,
            )
            .std()
        )

        merged["is_halt_interval"] = 0
        if halt_start is not None and halt_end is not None:
            halt_start = _as_utc(halt_start)
            halt_end = _as_utc(halt_end)
            mask = (merged["timestamp"] >= halt_start) & (
                merged["timestamp"] <= halt_end
            )
            merged.loc[mask, "is_halt_interval"] = 1

        merged["treatment_group"] = treatment_flag
        return merged
