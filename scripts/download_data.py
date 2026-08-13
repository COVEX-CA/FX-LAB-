#!/usr/bin/env python
"""CLI para descargar un símbolo y rango de fechas a la caché local en parquet.

Orquesta `fxlab.data.download` y `fxlab.data.store`: por cada tramo mensual
del rango pedido, si ya está cacheado en disco lo lee de ahí (sin red); si
no, lo descarga de Dukascopy y lo guarda. Esto es lo que hace idempotente
volver a pedir un rango ya descargado.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import dukascopy_python
import pandas as pd

from fxlab.data.download import download, month_chunks
from fxlab.data.store import DEFAULT_DATA_DIR, cached_ranges, load, save
from fxlab.data.types import OfferSide

logger = logging.getLogger("fxlab.download_data")

_INTERVAL_CHOICES = {
    "1MIN": dukascopy_python.INTERVAL_MIN_1,
    "5MIN": dukascopy_python.INTERVAL_MIN_5,
    "15MIN": dukascopy_python.INTERVAL_MIN_15,
    "30MIN": dukascopy_python.INTERVAL_MIN_30,
    "1HOUR": dukascopy_python.INTERVAL_HOUR_1,
    "4HOUR": dukascopy_python.INTERVAL_HOUR_4,
    "1DAY": dukascopy_python.INTERVAL_DAY_1,
}


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def _is_chunk_cached(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    data_dir: Path,
) -> bool:
    for cached_start, cached_end in cached_ranges(symbol, interval, offer_side, data_dir):
        if cached_start <= chunk_start and chunk_end <= cached_end:
            return True
    return False


def download_to_cache(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    offer_side: OfferSide,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Descarga (o lee de caché) un rango completo, tramo mensual a tramo mensual."""
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
        return load(symbol, interval, offer_side, pd.Timestamp(start), pd.Timestamp(end), data_dir)

    result = pd.concat(frames).sort_index()
    return result[~result.index.duplicated(keep="last")]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help='Instrumento Dukascopy, p.ej. "EUR/USD".')
    parser.add_argument("--start", required=True, type=_parse_date, help="Fecha inicio YYYY-MM-DD.")
    parser.add_argument(
        "--end", required=True, type=_parse_date, help="Fecha fin YYYY-MM-DD (exclusive)."
    )
    parser.add_argument(
        "--interval",
        default="1MIN",
        choices=sorted(_INTERVAL_CHOICES),
        help="Intervalo nativo de Dukascopy a descargar (por defecto 1MIN).",
    )
    parser.add_argument(
        "--offer-side",
        default="bid",
        choices=["bid", "ask"],
        help="Lado de la cotización a descargar (por defecto bid).",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Raíz de la caché local en disco (por defecto 'data/').",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    interval = _INTERVAL_CHOICES[args.interval]
    offer_side = OfferSide(args.offer_side)

    result = download_to_cache(
        args.symbol, args.start, args.end, interval, offer_side, args.data_dir
    )
    logger.info(
        "%d velas disponibles en caché para %s [%s, %s)",
        len(result),
        args.symbol,
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()
