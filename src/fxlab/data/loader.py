"""Orquestación de caché y descarga: punto de entrada único para obtener datos.

Responsabilidad de este módulo: decidir, tramo mensual a tramo mensual, qué
datos ya están cacheados y cuáles hay que descargar — combinando
`fxlab.data.download` y `fxlab.data.store` sin que esos dos módulos se
conozcan entre sí. Cualquier código que necesite datos OHLCV (el barrido de
parámetros en fases futuras, scripts, notebooks) debe pasar por
`load_range` en vez de reimplementar esta lógica; así solo hay una versión
del comportamiento de caché.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fxlab.data.download import download, month_chunks
from fxlab.data.store import DEFAULT_DATA_DIR, cached_ranges, load, save
from fxlab.data.types import OfferSide
from fxlab.data.validate import validate_ohlcv_contract

logger = logging.getLogger(__name__)

# Tolerancia al comprobar si un tramo mensual ya está cubierto por un rango
# cacheado. Las barras se etiquetan por su timestamp de apertura, así que el
# último dato de un mes completo siempre queda algo antes de la medianoche
# del mes siguiente (hasta un intervalo de barra; más si hubo festivos o
# fin de semana al cierre del mes). Mismo criterio que el gap de fusión de
# `fxlab.data.store.cached_ranges`.
_CHUNK_COVERAGE_TOLERANCE = pd.Timedelta(days=3)


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _is_chunk_cached(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    data_dir: Path,
) -> bool:
    for cached_start, cached_end in cached_ranges(symbol, interval, offer_side, data_dir):
        if cached_start <= chunk_start and chunk_end - cached_end <= _CHUNK_COVERAGE_TOLERANCE:
            return True
    return False


def load_range(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    offer_side: OfferSide,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Obtiene velas OHLCV de un símbolo y rango, descargando solo lo que falte.

    Por cada tramo mensual del rango pedido (ver
    `fxlab.data.download.month_chunks`):

    1. consulta qué hay ya cacheado (`fxlab.data.store.cached_ranges`);
    2. si el tramo ya está cubierto, lo lee de disco
       (`fxlab.data.store.load`) sin tocar la red;
    3. si no, lo descarga (`fxlab.data.download.download`) y lo guarda
       (`fxlab.data.store.save`);

    y al final valida el contrato de datos del resultado combinado y
    devuelve el DataFrame completo del rango pedido.

    Args:
        symbol: instrumento, p.ej. "EUR/USD".
        start: inicio del rango (inclusive). Si es naive se asume UTC.
        end: fin del rango (exclusive). Si es naive se asume UTC.
        interval: intervalo nativo de Dukascopy (constantes `INTERVAL_*` de
            `dukascopy_python`).
        offer_side: `OfferSide.BID` o `OfferSide.ASK`.
        data_dir: raíz de la caché en disco.

    Returns:
        DataFrame que cumple el contrato de datos OHLCV.
    """
    start = _ensure_utc(start)
    end = _ensure_utc(end)

    frames = []
    for chunk_start, chunk_end in month_chunks(start, end):
        ts_start = pd.Timestamp(chunk_start)
        ts_end = pd.Timestamp(chunk_end)

        if _is_chunk_cached(symbol, interval, offer_side, ts_start, ts_end, data_dir):
            logger.info("tramo cacheado, leyendo de disco: %s a %s", ts_start, ts_end)
            chunk_df = load(symbol, interval, offer_side, ts_start, ts_end, data_dir)
        else:
            logger.info("tramo no cacheado, descargando: %s a %s", ts_start, ts_end)
            chunk_df = download(symbol, chunk_start, chunk_end, interval, offer_side)
            if not chunk_df.empty:
                save(chunk_df, symbol, interval, offer_side, data_dir)

        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        result = load(
            symbol, interval, offer_side, pd.Timestamp(start), pd.Timestamp(end), data_dir
        )
    else:
        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep="last")]

    validate_ohlcv_contract(result)
    return result
