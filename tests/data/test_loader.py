from __future__ import annotations

import calendar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fxlab.data import loader as loader_module
from fxlab.data.download import DownloadError
from fxlab.data.loader import load_range
from fxlab.data.store import is_month_complete, save
from fxlab.data.types import OfferSide
from fxlab.data.validate import validate_ohlcv_contract


def _month_daily_df(year: int, month: int, periods: int | None = None) -> pd.DataFrame:
    """Barras diarias para un mes natural (o los primeros `periods` días)."""
    days_in_month = calendar.monthrange(year, month)[1]
    n = periods if periods is not None else days_in_month
    index = pd.date_range(f"{year:04d}-{month:02d}-01", periods=n, freq="1D", tz="UTC")
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


_NOW = datetime(2024, 6, 1, tzinfo=UTC)  # muy posterior a cualquier fecha usada en los tests


# --- Los tres tests pedidos por la tarea ------------------------------------


def test_failed_download_is_not_marked_complete_and_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un mes cuya descarga falla no se reporta como cubierto y se reintenta."""
    call_count = {"n": 0}

    def flaky_then_ok(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise DownloadError("fallo de red simulado a mitad de la descarga de enero")
        return _month_daily_df(2024, 1)

    monkeypatch.setattr(loader_module, "download", flaky_then_ok)

    with pytest.raises(DownloadError):
        load_range(
            "EUR/USD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
            "1DAY",
            OfferSide.BID,
            data_dir=tmp_path,
            now=_NOW,
        )

    # tras el fallo, nada queda marcado como completo
    assert not is_month_complete("EUR/USD", "1DAY", OfferSide.BID, 2024, 1, base_path=tmp_path)

    # la siguiente llamada vuelve a intentar descargar enero (no se dio por bueno)
    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    assert call_count["n"] == 2
    assert len(result) == 31
    assert is_month_complete("EUR/USD", "1DAY", OfferSide.BID, 2024, 1, base_path=tmp_path)


def test_month_with_fewer_bars_due_to_holidays_is_marked_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un mes con menos barras por festivos/fines de semana sí se marca completo."""
    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        # enero natural tiene 31 días; aquí solo llegan 20 barras (festivos y
        # fines de semana), pero la descarga en sí termina sin error
        return _month_daily_df(2024, 1, periods=20)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result_1 = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )
    assert len(calls) == 1
    assert len(result_1) == 20
    assert is_month_complete("EUR/USD", "1DAY", OfferSide.BID, 2024, 1, base_path=tmp_path)

    def fail_if_called(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("no debería redescargar: el mes ya está marcado completo")

    monkeypatch.setattr(loader_module, "download", fail_if_called)

    result_2 = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )
    assert len(result_2) == 20


def test_complete_month_not_redownloaded_for_different_sub_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un mes completo no se redescarga aunque se pida una sub-ventana distinta."""

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        return _month_daily_df(2024, 1)

    monkeypatch.setattr(loader_module, "download", fake_download)

    load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("no debería redescargar: el mes ya está marcado completo")

    monkeypatch.setattr(loader_module, "download", fail_if_called)

    # fechas que no coinciden ni con el mes natural ni con la petición anterior
    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 10, tzinfo=UTC),
        datetime(2024, 1, 20, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )
    assert len(result) == 10  # días 10 a 19


# --- Caso especial: el mes en curso nunca se marca completo -----------------


def test_current_month_is_never_marked_complete_and_is_always_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)  # a mitad de enero
    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        assert end == now  # el mes en curso solo se pide hasta "ahora"
        return _month_daily_df(2024, 1, periods=15)

    monkeypatch.setattr(loader_module, "download", fake_download)

    for _ in range(2):
        load_range(
            "EUR/USD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
            "1DAY",
            OfferSide.BID,
            data_dir=tmp_path,
            now=now,
        )

    assert len(calls) == 2  # se ha vuelto a descargar en la segunda llamada
    assert not is_month_complete("EUR/USD", "1DAY", OfferSide.BID, 2024, 1, base_path=tmp_path)


# --- Comportamiento general de load_range -----------------------------------


def test_empty_cache_downloads_every_full_calendar_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        df = _month_daily_df(start.year, start.month)
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 15, tzinfo=UTC),
        datetime(2024, 3, 10, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    # se descarga cada mes natural COMPLETO que solapa el rango, no solo el
    # tramo pedido: enero, febrero y marzo de 2024
    assert calls == [
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
        (datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)),
        (datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)),
    ]
    # pero el resultado devuelto sí se recorta al rango pedido
    days_jan = 31 - 15 + 1  # 15..31
    days_feb = calendar.monthrange(2024, 2)[1]
    days_mar = 9  # 1..9
    assert len(result) == days_jan + days_feb + days_mar
    for month in (1, 2, 3):
        assert is_month_complete("EUR/USD", "1DAY", OfferSide.BID, 2024, month, base_path=tmp_path)


def test_full_cache_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(
        _month_daily_df(2024, 1),
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        complete=True,
        base_path=tmp_path,
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("no debería descargar nada: el mes ya está cacheado y completo")

    monkeypatch.setattr(loader_module, "download", fail_if_called)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    assert len(result) == 31


def test_partial_cache_downloads_only_missing_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save(
        _month_daily_df(2024, 1),
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        complete=True,
        base_path=tmp_path,
    )

    calls: list[tuple[datetime, datetime]] = []

    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        calls.append((start, end))
        return _month_daily_df(start.year, start.month)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    # solo se descarga febrero, enero ya estaba cacheado y completo
    assert calls == [(datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC))]
    assert len(result) == 31 + calendar.monthrange(2024, 2)[1]


def test_result_satisfies_data_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(
        symbol: str, start: datetime, end: datetime, interval: str, offer_side: OfferSide
    ) -> pd.DataFrame:
        return _month_daily_df(start.year, start.month)

    monkeypatch.setattr(loader_module, "download", fake_download)

    result = load_range(
        "EUR/USD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 10, tzinfo=UTC),
        "1DAY",
        OfferSide.BID,
        data_dir=tmp_path,
        now=_NOW,
    )

    validate_ohlcv_contract(result)
