"""Descarga de velas OHLCV históricas desde Dukascopy.

Responsabilidad única de este módulo: hablar con Dukascopy y devolver
DataFrames que cumplan el contrato de datos (ver `fxlab.data.validate`). No
sabe nada de caché en disco (`fxlab.data.store`) ni de resampleo
(`fxlab.data.resample`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast

import dukascopy_python
import pandas as pd

from fxlab.data.types import OfferSide
from fxlab.data.validate import validate_ohlcv_contract

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 1.0

_DUKASCOPY_OFFER_SIDE = {
    OfferSide.BID: dukascopy_python.OFFER_SIDE_BID,
    OfferSide.ASK: dukascopy_python.OFFER_SIDE_ASK,
}


class DownloadError(RuntimeError):
    """Se lanza cuando la descarga de un tramo falla tras agotar los reintentos."""


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def month_chunks(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Divide [start, end) en tramos por mes natural.

    Descargar por trozos evita que un rango largo sea una única petición
    gigante. El último tramo se recorta a `end`.
    """
    if end <= start:
        return

    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1)
        chunk_start = max(cursor, start)
        chunk_end = min(next_month, end)
        yield chunk_start, chunk_end
        cursor = next_month


def _fetch_chunk_with_retry(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    start: datetime,
    end: datetime,
    max_retries: int,
    backoff_seconds: float,
) -> pd.DataFrame:
    """Descarga un tramo con reintentos y backoff exponencial ante fallo de red."""
    attempt = 0
    while True:
        try:
            return cast(
                pd.DataFrame,
                dukascopy_python.fetch(
                    instrument=symbol,
                    interval=interval,
                    offer_side=_DUKASCOPY_OFFER_SIDE[offer_side],
                    start=start,
                    end=end,
                ),
            )
        except Exception as exc:
            if attempt >= max_retries:
                raise DownloadError(
                    f"fallo al descargar {symbol} [{start.isoformat()}, {end.isoformat()}) "
                    f"tras {max_retries} reintentos: {exc}"
                ) from exc
            wait = backoff_seconds * (2**attempt)
            logger.warning(
                "fallo descargando %s [%s, %s) (intento %d/%d), reintentando en %.1fs: %s",
                symbol,
                start.isoformat(),
                end.isoformat(),
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
            attempt += 1


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    index = df.index
    assert isinstance(index, pd.DatetimeIndex)
    df.index = index.tz_convert("UTC")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype("float64")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def download(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = dukascopy_python.INTERVAL_MIN_1,
    offer_side: OfferSide = OfferSide.BID,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Descarga velas OHLCV de Dukascopy para un símbolo y rango de fechas.

    Descarga por trozos mensuales (ver `month_chunks`) para no hacer una
    única petición gigante en rangos largos, con reintentos y backoff
    exponencial ante fallo de red en cada trozo.

    Args:
        symbol: instrumento Dukascopy, p.ej. "EUR/USD".
        start: inicio del rango (inclusive). Si es naive se asume UTC.
        end: fin del rango (exclusive). Si es naive se asume UTC.
        interval: intervalo nativo de Dukascopy (constantes `INTERVAL_*` de
            `dukascopy_python`), p.ej. `dukascopy_python.INTERVAL_MIN_1`.
        offer_side: `OfferSide.BID` o `OfferSide.ASK`. Bid y ask deben
            descargarse por separado para poder calcular el spread real.
        max_retries: reintentos máximos por tramo antes de propagar el error.
        backoff_seconds: base del backoff exponencial entre reintentos.

    Returns:
        DataFrame que cumple el contrato de datos OHLCV (ver
        `fxlab.data.validate.validate_ohlcv_contract`).
    """
    start = _ensure_utc(start)
    end = _ensure_utc(end)

    frames = []
    for chunk_start, chunk_end in month_chunks(start, end):
        chunk_df = _fetch_chunk_with_retry(
            symbol, interval, offer_side, chunk_start, chunk_end, max_retries, backoff_seconds
        )
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")
        empty.index = pd.DatetimeIndex([], tz="UTC")
        validate_ohlcv_contract(empty)
        return empty

    result = _normalize(pd.concat(frames))
    validate_ohlcv_contract(result)
    return result
