from pathlib import Path

import src.run_project as runner
from src.yfinance_ingest import IngestionReport


def test_run_project_refreshes_data_then_builds_dashboard(tmp_path, monkeypatch):
    calls: list[tuple[str, object]] = []
    report = IngestionReport(
        requested_symbols=["ABC"],
        downloaded_bars=3,
        upserted_bars=3,
        successful_symbols=["ABC"],
        failed_symbols={},
    )

    def fake_ingest(tickers, *, period):
        calls.append(("ingest", (tickers, period)))
        return report

    def fake_analysis(**kwargs):
        calls.append(("analysis", kwargs))
        return {
            "dashboard_png": str(Path(kwargs["output_dir"]) / "market_dashboard.png")
        }

    monkeypatch.setattr(runner, "ingest_yfinance_to_mongodb", fake_ingest)
    monkeypatch.setattr(runner, "run_analysis", fake_analysis)

    result = runner.run_project(
        tickers=["ABC"],
        period="6mo",
        output_dir=tmp_path,
    )

    assert calls[0] == ("ingest", (["ABC"], "6mo"))
    assert calls[1][0] == "analysis"
    assert calls[1][1]["source"] == "mongodb"
    assert result["ingestion"]["upserted_bars"] == 3
    assert result["analysis"]["dashboard_png"].endswith("market_dashboard.png")


def test_run_project_can_skip_refresh(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "ingest_yfinance_to_mongodb",
        lambda *args, **kwargs: calls.append("ingest"),
    )
    monkeypatch.setattr(
        runner,
        "run_analysis",
        lambda **kwargs: calls.append("analysis") or {"symbols": ["ABC"]},
    )

    result = runner.run_project(output_dir=tmp_path, skip_download=True)

    assert calls == ["analysis"]
    assert result["ingestion"] is None
    assert result["analysis"]["symbols"] == ["ABC"]
