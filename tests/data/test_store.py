from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fxlab.data.store import cached_ranges, load, save
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


def test_save_creates_one_file_per_month(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)  # cruza de enero a febrero 2024
    written = save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    assert len(written) == 2
    directory = tmp_path / "EUR_USD" / "1DAY" / "bid"
    assert (directory / "2024-01.parquet").exists()
    assert (directory / "2024-02.parquet").exists()


def test_save_load_roundtrip_is_lossless(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    loaded = load(
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(days=1),
        base_path=tmp_path,
    )

    # El roundtrip por parquet no preserva el atributo `freq` del índice,
    # solo los valores y su orden: eso es lo que importa para "sin pérdida".
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_load_filters_to_requested_range(tmp_path: Path) -> None:
    df = _daily_df("2024-01-01", periods=20)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    start = pd.Timestamp("2024-01-10", tz="UTC")
    end = pd.Timestamp("2024-01-15", tz="UTC")
    loaded = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)

    assert loaded.index.min() == start
    assert loaded.index.max() == pd.Timestamp("2024-01-14", tz="UTC")
    assert (loaded.index >= start).all()
    assert (loaded.index < end).all()


def test_load_missing_symbol_returns_empty_frame(tmp_path: Path) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-02-01", tz="UTC")
    loaded = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)

    assert loaded.empty
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_bid_and_ask_are_stored_separately(tmp_path: Path) -> None:
    bid_df = _daily_df("2024-01-01", periods=5)
    ask_df = bid_df.copy()
    ask_df["open"] += 0.001

    save(bid_df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)
    save(ask_df, "EUR/USD", "1DAY", OfferSide.ASK, base_path=tmp_path)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-06", tz="UTC")
    loaded_bid = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)
    loaded_ask = load("EUR/USD", "1DAY", OfferSide.ASK, start=start, end=end, base_path=tmp_path)

    assert not loaded_bid["open"].equals(loaded_ask["open"])


def test_cached_ranges_reports_saved_data(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    ranges = cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    assert len(ranges) == 1
    start, end = ranges[0]
    assert start == df.index.min()
    assert end == df.index.max()


def test_cached_ranges_empty_when_nothing_saved(tmp_path: Path) -> None:
    assert cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path) == []
