import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

market_dashboard = importlib.import_module("main")
load_market_data = market_dashboard.load_market_data
load_market_data_from_mongodb = market_dashboard.load_market_data_from_mongodb
main = market_dashboard.main
run_analysis = market_dashboard.run_analysis


def test_run_analysis_builds_dashboard_and_summary(tmp_path):
    bars_path = tmp_path / "bars.csv"
    halts_path = tmp_path / "halts.csv"
    output_dir = tmp_path / "results"
    timestamps = pd.date_range("2025-01-02 14:30", periods=4, freq="min", tz="UTC")
    pd.DataFrame(
        {
            "Datetime": timestamps,
            "Symbol": ["ABC"] * 4,
            "Exchange": ["XNAS"] * 4,
            "Open": [100.0, 100.2, 100.1, 100.5],
            "High": [100.3, 100.4, 100.6, 100.7],
            "Low": [99.9, 100.0, 100.0, 100.3],
            "Close": [100.1, 100.2, 100.5, 100.6],
            "Volume": [1_000, 1_200, 900, 1_100],
            "Bid Price": [100.0, 100.1, 100.3, 100.4],
            "Ask Price": [100.2, 100.3, 100.5, 100.8],
            "Bid Size": [300, 320, 280, 290],
            "Ask Size": [250, 270, 260, 310],
        }
    ).to_csv(bars_path, index=False)
    pd.DataFrame(
        {
            "halt_start": [timestamps[1]],
            "halt_end": [timestamps[2]],
            "symbol": ["ABC"],
            "exchange": ["XNAS"],
        }
    ).to_csv(halts_path, index=False)

    summary = run_analysis(
        bars_path,
        output_dir,
        halt_path=halts_path,
        rolling_window=2,
    )

    assert summary["rows"] == 4
    assert summary["symbol"] == "ABC"
    assert summary["exchange"] == "XNAS"
    assert summary["halt_event_count"] == 1
    assert summary["mean_spread"] == pytest.approx(0.25)
    assert summary["total_volume"] == 4_200
    assert (output_dir / "market_dashboard.png").is_file()
    saved_summary = json.loads((output_dir / "market_summary.json").read_text())
    assert saved_summary["last_close"] == pytest.approx(100.6)

    cli_output_dir = tmp_path / "cli-results"
    assert (
        main(
            [
                "--input",
                str(bars_path),
                "--source",
                "csv",
                "--symbol",
                "ABC",
                "--halts",
                str(halts_path),
                "--output-dir",
                str(cli_output_dir),
                "--rolling-window",
                "2",
            ]
        )
        == 0
    )
    assert (cli_output_dir / "market_dashboard.png").is_file()


def test_load_market_data_requires_symbol_for_multiple_instruments(tmp_path):
    bars_path = tmp_path / "multi.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-02T14:30:00Z", "2025-01-02T14:30:00Z"],
            "symbol": ["ABC", "XYZ"],
            "close": [100.0, 25.0],
        }
    ).to_csv(bars_path, index=False)

    with pytest.raises(ValueError, match="multiple symbols"):
        load_market_data(bars_path)

    selected = load_market_data(bars_path, symbol="ABC")
    assert selected["symbol"].tolist() == ["ABC"]


def test_run_analysis_groups_row_level_halt_flags(tmp_path):
    bars_path = tmp_path / "halted_bars.csv"
    timestamps = pd.date_range("2025-01-02 14:30", periods=4, freq="min", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [50.0, 50.0, 50.0, 50.2],
            "is_halt_interval": [0, 1, 1, 0],
        }
    ).to_csv(bars_path, index=False)

    summary = run_analysis(bars_path, tmp_path / "halt-flag-results")

    assert summary["halt_event_count"] == 1
    assert summary["halt_duration_minutes"] == pytest.approx(2.0)


def test_mongodb_without_symbol_builds_multi_company_overview(tmp_path, monkeypatch):
    timestamps = pd.date_range("2025-01-02", periods=3, freq="D", tz="UTC")
    company_data = {
        "AAA": pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [10.0, 11.0, 12.0],
                "volume": [100, 120, 130],
                "rolling_volatility": [None, 0.1, 0.08],
                "exchange": ["NMS"] * 3,
            }
        ),
        "BBB": pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [20.0, 19.0, 21.0],
                "volume": [200, 220, 230],
                "rolling_volatility": [None, 0.05, 0.07],
                "exchange": ["NMS"] * 3,
            }
        ),
    }
    monkeypatch.setattr(
        market_dashboard,
        "load_all_market_data_from_mongodb",
        lambda **kwargs: company_data,
    )

    summary = run_analysis(
        source="mongodb",
        output_dir=tmp_path,
        database_name="research",
    )

    assert summary["symbols"] == ["AAA", "BBB"]
    assert summary["companies"]["AAA"]["close_change_percent"] == pytest.approx(20.0)
    assert (tmp_path / "market_dashboard.png").is_file()
    assert (tmp_path / "market_summary.json").is_file()


def test_mongodb_loader_uses_env_config_and_reads_real_collections(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://configured.example")
    monkeypatch.setenv("MONGODB_DATABASE", "live_market_data")

    client = MagicMock()
    database = MagicMock()
    client.__getitem__.return_value = database
    bars_collection = MagicMock()
    quotes_collection = MagicMock()
    halts_collection = MagicMock()
    database.__getitem__.side_effect = {
        "market_bars": bars_collection,
        "market_quotes": quotes_collection,
        "halt_events": halts_collection,
    }.__getitem__

    timestamp = pd.Timestamp("2025-01-02T14:30:00Z")
    bars_collection.distinct.return_value = ["ABC"]
    bars_collection.find.return_value.sort.return_value.limit.return_value = [
        {
            "timestamp": timestamp.to_pydatetime(),
            "symbol": "ABC",
            "exchange": "XNAS",
            "close": 100.0,
            "volume": 1000,
        }
    ]
    quotes_collection.find.return_value.sort.return_value.limit.return_value = [
        {
            "timestamp": timestamp.to_pydatetime(),
            "symbol": "ABC",
            "exchange": "XNAS",
            "spread": 0.2,
            "bid_size": 200,
            "ask_size": 300,
        }
    ]
    halts_collection.find.return_value.sort.return_value.limit.return_value = []

    frame, halt_windows = load_market_data_from_mongodb(
        client=client,
        rolling_window=2,
    )

    client.admin.command.assert_called_once_with("ping")
    database.__getitem__.assert_any_call("market_bars")
    database.__getitem__.assert_any_call("market_quotes")
    database.__getitem__.assert_any_call("halt_events")
    assert frame.loc[0, "effective_spread"] == pytest.approx(0.2)
    assert frame.loc[0, "market_depth"] == 500
    assert halt_windows == []
    client.close.assert_called_once_with()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-o", "addopts="]))
