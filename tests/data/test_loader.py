from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fxlab.data import loader as loader_module
from fxlab.data.loader import load_range
from fxlab.data.store import cached_ranges, save
from fxlab.data.types import OfferSide


def _daily_df(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    n = len(index)
    return pd.DataFrame(
        {
            "open": [float(i) for i in range(n)],
            "high": [float(i) + 0.5 for i in range(n)],
            "low": [float(i) - 0.5 for i in range(n)],
            "close": [float(i) + 0.2 for i in range(n)],
            "volume": [10.0 for _ in range(n)],
        },
        index=index,
    )


def test_empty_cache_downloads_every_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        return _daily_df(start.strftime("%Y-%m-%d"), periods=1)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 15, tzinfo=UTC),
        datetime(2024, 3, 10, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
    )

    # un tramo por mes natural: [15 ene, 1 feb), [1 feb, 1 mar), [1 mar, 10 mar)
    assert len(calls) == 3
    assert len(result) == 3
    # lo descargado queda cacheado, no solo devuelto
    assert len(cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)) > 0


def test_full_cache_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _daily_df("2024-01-01", periods=31)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    def fail_if_called(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("no debería descargar nada: el rango ya está cacheado")

    monkeypatch.setattr(loader_module, "download", fail_if_called)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
    )

    assert len(result) == 31


def test_partial_cache_downloads_only_missing_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jan_df = _daily_df("2024-01-01", periods=31)
    save(jan_df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        return _daily_df(start.strftime("%Y-%m-%d"), periods=5)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
    )

    # solo se descarga febrero, enero ya estaba cacheado completo
    assert calls == [(datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC))]
    assert len(result) == 31 + 5


def test_result_satisfies_data_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fxlab.data.validate import validate_ohlcv_contract

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        return _daily_df(start.strftime("%Y-%m-%d"), periods=3)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 10, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
    )

    validate_ohlcv_contract(result)
