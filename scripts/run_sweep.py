#!/usr/bin/env python
"""CLI para ejecutar un barrido de parámetros de `fxlab.strategies.pullback`
y registrar cada combinación evaluada.

Opera sobre la partición de desarrollo por defecto; acceder al holdout
exige `--partition holdout` explícito (ver `fxlab.split`). No imprime
rankings ni "mejores resultados" — eso es la fase de validación
estadística, que decide con cuidado, no un script de barrido.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dukascopy_python
import pandas as pd

from fxlab.data.loader import load_range
from fxlab.data.store import DEFAULT_DATA_DIR
from fxlab.data.types import OfferSide
from fxlab.split import DEVELOPMENT_START, HOLDOUT_START, Partition
from fxlab.sweep.config import load_config
from fxlab.sweep.engine import run_sweep
from fxlab.sweep.registry import TrialRegistry

logger = logging.getLogger("fxlab.run_sweep")

_INTERVAL_CHOICES = {
    "1MIN": dukascopy_python.INTERVAL_MIN_1,
    "5MIN": dukascopy_python.INTERVAL_MIN_5,
    "15MIN": dukascopy_python.INTERVAL_MIN_15,
    "30MIN": dukascopy_python.INTERVAL_MIN_30,
    "1HOUR": dukascopy_python.INTERVAL_HOUR_1,
    "4HOUR": dukascopy_python.INTERVAL_HOUR_4,
    "1DAY": dukascopy_python.INTERVAL_DAY_1,
}

# Frecuencia pandas equivalente, para anualizar métricas en VectorBT.
_VBT_FREQ = {
    "1MIN": "1min",
    "5MIN": "5min",
    "15MIN": "15min",
    "30MIN": "30min",
    "1HOUR": "1h",
    "4HOUR": "4h",
    "1DAY": "1D",
}

_DEFAULT_REGISTRY_PATH = Path("data") / "sweep_trials.db"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Fichero YAML de configuración del barrido.")
    parser.add_argument(
        "--partition",
        default="development",
        choices=["development", "holdout"],
        help="Partición temporal a usar (por defecto 'development'). "
        "'holdout' debe pedirse explícitamente, nunca es el valor por defecto.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Raíz de la caché local de datos (por defecto 'data/').",
    )
    parser.add_argument(
        "--registry-db",
        default=_DEFAULT_REGISTRY_PATH,
        type=Path,
        help=f"Fichero SQLite del registro de pruebas (por defecto {_DEFAULT_REGISTRY_PATH}).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    config = load_config(args.config)
    partition = Partition(args.partition)
    interval = _INTERVAL_CHOICES[config.interval]

    if partition is Partition.HOLDOUT:
        start, end = HOLDOUT_START, pd.Timestamp.now(tz="UTC")
    else:
        start, end = DEVELOPMENT_START, HOLDOUT_START

    logger.info(
        "cargando %s %s, partición %s, [%s, %s)",
        config.symbol,
        config.interval,
        partition.value,
        start,
        end,
    )
    bid = load_range(config.symbol, start, end, interval, OfferSide.BID, args.data_dir)
    ask = load_range(config.symbol, start, end, interval, OfferSide.ASK, args.data_dir)

    def _progress(done: int, total: int) -> None:
        logger.info("combinación %d/%d", done, total)

    with TrialRegistry(args.registry_db) as registry:
        total = run_sweep(
            bid,
            ask,
            config.grid,
            config.cost_model,
            registry,
            experiment_id=config.experiment_id,
            symbol=config.symbol,
            interval=config.interval,
            freq=_VBT_FREQ[config.interval],
            partition=partition,
            progress=_progress,
        )

    logger.info(
        "barrido '%s' terminado: %d combinaciones registradas en %s",
        config.experiment_id,
        total,
        args.registry_db,
    )


if __name__ == "__main__":
    main()
