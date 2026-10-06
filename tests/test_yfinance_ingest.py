from datetime import UTC
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.yfinance_ingest import (
    download_yfinance_bars,
    ingest_yfinance_to_mongodb,
)


def test_yfinance_history_converts_to_project_bars():
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000, 1_100],
        },
        index=pd.date_range("2025-01-02", periods=2, freq="D"),
    )
    ticker = MagicMock()
    ticker.history.return_value = history
    ticker.history_metadata = {"exchangeName": "NMS"}

    bars, failures = download_yfinance_bars(
        [" aapl "],
        period="6mo",
        ticker_factory=lambda symbol: ticker,
    )

    assert failures == {}
    assert len(bars) == 2
    assert bars[0].symbol == "AAPL"
    assert bars[0].exchange == "NMS"
    assert bars[0].timestamp.tzinfo == UTC
    assert bars[1].close == 102.0
    ticker.history.assert_called_once_with(
        period="6mo",
        interval="1d",
        auto_adjust=True,
        actions=False,
    )


def test_ingestion_upserts_bars_into_market_data_manager():
    history = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [500],
        },
        index=pd.DatetimeIndex(["2025-02-03"], name="Date"),
    )
    ticker = MagicMock()
    ticker.history.return_value = history
    ticker.history_metadata = {}
    manager = MagicMock()
    manager.insert_bars.return_value = 1

    report = ingest_yfinance_to_mongodb(
        ["abc", "ABC"],
        ticker_factory=lambda symbol: ticker,
        manager=manager,
    )

    manager.insert_bars.assert_called_once()
    stored_bars = manager.insert_bars.call_args.args[0]
    assert [bar.symbol for bar in stored_bars] == ["ABC"]
    assert report.requested_symbols == ["ABC"]
    assert report.downloaded_bars == 1
    assert report.upserted_bars == 1
    assert report.successful_symbols == ["ABC"]
    assert report.failed_symbols == {}


def test_download_reports_empty_history_and_requires_tickers():
    ticker = MagicMock()
    ticker.history.return_value = pd.DataFrame()

    bars, failures = download_yfinance_bars(
        ["MISSING"],
        ticker_factory=lambda symbol: ticker,
    )

    assert bars == []
    assert "MISSING" in failures
    with pytest.raises(ValueError, match="at least one ticker"):
        download_yfinance_bars([], ticker_factory=lambda symbol: ticker)
