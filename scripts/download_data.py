#!/usr/bin/env python
"""CLI para descargar un símbolo y rango de fechas a la caché local en parquet.

Solo parsea argumentos: toda la lógica de caché/descarga vive en
`fxlab.data.loader.load_range`.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import dukascopy_python

from fxlab.data.loader import load_range
from fxlab.data.store import DEFAULT_DATA_DIR
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

    result = load_range(args.symbol, args.start, args.end, interval, offer_side, args.data_dir)
    logger.info(
        "%d velas disponibles en caché para %s [%s, %s)",
        len(result),
        args.symbol,
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()
