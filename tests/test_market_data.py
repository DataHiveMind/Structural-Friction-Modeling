from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.market_data import MarketDataManager, Quote, TradeBar


def test_manager_uses_database_settings_from_environment(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://configured.example")
    monkeypatch.setenv("MONGODB_DATABASE", "market_research")
    pymongo_client = MagicMock()
    manager_factory = MagicMock(return_value=pymongo_client)
    original_factory = __import__("src.market_data", fromlist=["MongoClient"])
    monkeypatch.setattr(original_factory, "MongoClient", manager_factory)

    manager = MarketDataManager(ensure_indexes=False)

    manager_factory.assert_called_once_with(
        "mongodb://configured.example",
        serverSelectionTimeoutMS=5_000,
    )
    assert manager.db is pymongo_client["market_research"]
    manager.close()
    pymongo_client.close.assert_called_once_with()


def test_quote_document_normalizes_timestamp_to_utc():
    local_time = datetime(
        2025,
        1,
        1,
        12,
        tzinfo=timezone(timedelta(hours=3)),
    )
    quote = Quote(
        timestamp=local_time,
        symbol="ABC",
        exchange="XNAS",
        bid_price=99.0,
        ask_price=101.0,
        bid_size=10.0,
        ask_size=12.0,
    )

    doc = quote.to_doc()

    assert doc["timestamp"].tzinfo == UTC
    assert doc["timestamp"].hour == 9
    assert doc["spread"] == 2.0


def test_window_metrics_do_not_invent_missing_quotes_or_confounders():
    manager = MarketDataManager(
        "mongodb://example.test",
        "research",
        client=MagicMock(),
        ensure_indexes=False,
    )
    timestamp = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
    manager.bars = MagicMock()
    manager.quotes = MagicMock()
    manager.confounders = MagicMock()
    manager.bars.find.return_value.sort.return_value = [
        {
            "timestamp": timestamp,
            "symbol": "ABC",
            "exchange": "XNAS",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 500.0,
        }
    ]
    manager.quotes.find.return_value.sort.return_value = []
    manager.confounders.find.return_value.sort.return_value = []

    result = manager._extract_window_metrics(
        symbol="ABC",
        exchange="XNAS",
        start=timestamp,
        end=timestamp + timedelta(minutes=1),
        treatment_flag=1,
        halt_start=None,
        halt_end=None,
    )

    assert np.isnan(result.loc[0, "effective_spread"])
    assert np.isnan(result.loc[0, "bid_depth"])
    assert np.isnan(result.loc[0, "ask_depth"])
    assert np.isnan(result.loc[0, "benchmark_return"])
    assert np.isnan(result.loc[0, "vix_level"])
    assert pd.isna(result.loc[0, "log_return"])


def test_bar_document_treats_naive_timestamp_as_utc():
    bar = TradeBar(
        timestamp=datetime(2025, 1, 1, 9, 30),
        symbol="ABC",
        exchange="XNAS",
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=100.0,
    )

    assert bar.to_doc()["timestamp"].utcoffset() == timedelta(0)
