#!/usr/bin/env python
"""CLI para ejecutar un barrido de parámetros de `fxlab.strategies.pullback`
y registrar cada combinación evaluada.

Opera sobre la partición de desarrollo por defecto; acceder al holdout
exige `--partition holdout` explícito (ver `fxlab.split`). No imprime
rankings ni "mejores resultados" — eso es la fase de validación
estadística, que decide con cuidado, no un script de barrido.

Por defecto barre la partición entera. `--start`/`--end` permiten
restringirlo a un subperíodo (p. ej. para una prueba rápida de la tubería),
pero **nunca** pueden cruzar el corte de holdout (2020-01-01): si el rango
pedido alcanza la partición reservada, el script falla con un error
explícito en vez de recortar en silencio, porque un recorte silencioso
tocaría datos que deben permanecer sin ver. Elegir subperíodo es una forma
de selección; el rango efectivo queda registrado en cada fila del registro
(`start_date`/`end_date`) para que sea auditable.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
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


def _parse_date(value: str) -> pd.Timestamp:
    """Convierte 'YYYY-MM-DD' en un `Timestamp` UTC (medianoche)."""
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC))


def resolve_range(
    partition: Partition,
    requested_start: pd.Timestamp | None,
    requested_end: pd.Timestamp | None,
    now: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Rango efectivo [start, end) a cargar, validado contra el corte de holdout.

    Si no se pide `--start`/`--end`, se usa la partición entera. Si se piden,
    acotan el rango, pero nunca pueden cruzar el corte de holdout
    (`fxlab.split.HOLDOUT_START`): un barrido de desarrollo que alcance esas
    fechas se rechaza, y una ejecución sobre holdout no puede retroceder a
    desarrollo. El fallo es explícito a propósito — recortar en silencio
    tocaría datos reservados y ocultaría que se han mirado.

    Args:
        partition: partición elegida (`DEVELOPMENT` por defecto en el CLI).
        requested_start: `--start` pedido, o `None` para el inicio de la partición.
        requested_end: `--end` pedido (exclusivo), o `None` para el fin de la partición.
        now: instante considerado "presente" (fin por defecto del holdout);
            inyectable para tests.

    Returns:
        `(start, end)` como `Timestamp` UTC, con `end` exclusivo.

    Raises:
        ValueError: si el rango queda vacío, o si cruza el corte de holdout
            en cualquiera de los dos sentidos.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if partition is Partition.HOLDOUT:
        default_start, default_end = HOLDOUT_START, now
    else:
        default_start, default_end = DEVELOPMENT_START, HOLDOUT_START

    start = requested_start if requested_start is not None else default_start
    end = requested_end if requested_end is not None else default_end

    if start >= end:
        raise ValueError(
            f"el rango pedido está vacío: --start {start.date()} no es anterior a "
            f"--end {end.date()} (recuerda que --end es exclusivo)"
        )

    if partition is Partition.DEVELOPMENT and end > HOLDOUT_START:
        raise ValueError(
            f"--end {end.date()} alcanza la partición reservada (holdout, >= "
            f"{HOLDOUT_START.date()}): un barrido de desarrollo no puede cruzar ese "
            f"corte. Usa como máximo --end {HOLDOUT_START.date()}, o pide "
            "--partition holdout de forma explícita."
        )
    if partition is Partition.HOLDOUT and start < HOLDOUT_START:
        raise ValueError(
            f"--start {start.date()} es anterior al corte de holdout "
            f"({HOLDOUT_START.date()}): una ejecución sobre holdout no puede "
            "alcanzar la partición de desarrollo."
        )
    return start, end


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
        "--start",
        type=_parse_date,
        default=None,
        help="Inicio del rango YYYY-MM-DD (inclusive). Por defecto, el inicio "
        "de la partición. No puede cruzar el corte de holdout.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="Fin del rango YYYY-MM-DD (exclusivo). Por defecto, el fin de la "
        "partición. Un --end que alcance el holdout hace fallar el barrido.",
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

    # El rango se valida ANTES de cargar: si cruza el holdout, se falla aquí
    # y `load_range` no llega a descargar ni tocar datos reservados.
    start, end = resolve_range(partition, args.start, args.end)

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
