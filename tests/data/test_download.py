from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import dukascopy_python
import pandas as pd
import pytest

from fxlab.data import download as download_module
from fxlab.data.download import DownloadError, OfferSide, download, month_chunks


def _fake_fetch_df(start: datetime) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(start).tz_convert("UTC")], name="timestamp")
    return pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.5],
            "low": [0.5],
            "close": [1.2],
            "volume": [100.0],
        },
        index=index,
    )


def test_month_chunks_splits_by_calendar_month() -> None:
    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end = datetime(2024, 3, 10, tzinfo=timezone.utc)

    chunks = list(month_chunks(start, end))

    assert chunks == [
        (datetime(2024, 1, 15, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
        (datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 3, 1, tzinfo=timezone.utc)),
        (datetime(2024, 3, 1, tzinfo=timezone.utc), datetime(2024, 3, 10, tzinfo=timezone.utc)),
    ]


def test_month_chunks_empty_range_yields_nothing() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert list(month_chunks(start, start)) == []


def test_download_fetches_one_chunk_per_month(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fetch(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return _fake_fetch_df(kwargs["start"])

    monkeypatch.setattr(dukascopy_python, "fetch", fake_fetch)

    result = download(
        "EUR/USD",
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        datetime(2024, 3, 10, tzinfo=timezone.utc),
    )

    assert len(calls) == 3
    assert len(result) == 3
    assert result.index.is_monotonic_increasing


def test_download_uses_correct_offer_side_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fetch(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return _fake_fetch_df(kwargs["start"])

    monkeypatch.setattr(dukascopy_python, "fetch", fake_fetch)

    download(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        offer_side=OfferSide.ASK,
    )

    assert calls[0]["offer_side"] == dukascopy_python.OFFER_SIDE_ASK


def test_download_retries_with_backoff_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def flaky_fetch(**kwargs: Any) -> pd.DataFrame:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("network blip")
        return _fake_fetch_df(kwargs["start"])

    monkeypatch.setattr(dukascopy_python, "fetch", flaky_fetch)
    monkeypatch.setattr(download_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = download(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        max_retries=5,
        backoff_seconds=1.0,
    )

    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]  # backoff exponencial: 1, 2, 4, ...
    assert len(result) == 1


def test_download_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fails(**kwargs: Any) -> pd.DataFrame:
        raise ConnectionError("network down")

    monkeypatch.setattr(dukascopy_python, "fetch", always_fails)
    monkeypatch.setattr(download_module.time, "sleep", lambda seconds: None)

    with pytest.raises(DownloadError):
        download(
            "EUR/USD",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 15, tzinfo=timezone.utc),
            max_retries=2,
        )


def test_download_empty_range_returns_empty_frame() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = download("EUR/USD", start, start)

    assert result.empty
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
