"""Partición temporal fija entre desarrollo (in-sample) y reservado (holdout).

- **Desarrollo**: 2004-01-01 a 2019-12-31, ambos inclusive.
- **Reservado (holdout)**: 2020-01-01 en adelante.

Este corte no se modifica en fases posteriores. Toda función de barrido debe
aceptar un parámetro de partición y operar sobre desarrollo por defecto;
acceder al tramo reservado exige pasar `partition=Partition.HOLDOUT`
explícitamente — nunca es el comportamiento por defecto ni algo que ocurra
por omisión.

Por qué es tan rígido: el tramo reservado solo sirve como validación
independiente del sesgo de selección mientras nadie lo haya usado para
decidir nada. En el momento en que se mira —aunque sea una sola vez, aunque
sea "solo para curiosear"— dejan de ser datos no vistos, y esa pérdida de
independencia no se puede deshacer ajustando el código después. Por eso el
corte es una constante fija en este módulo, no un parámetro configurable
por experimento.
"""

from __future__ import annotations

import logging
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)

DEVELOPMENT_START = pd.Timestamp("2004-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2019-12-31", tz="UTC")
HOLDOUT_START = pd.Timestamp("2020-01-01", tz="UTC")


class Partition(Enum):
    """Qué tramo de la partición temporal fija usar."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


def filter_partition(
    df: pd.DataFrame, partition: Partition = Partition.DEVELOPMENT
) -> pd.DataFrame:
    """Recorta `df` (índice `DatetimeIndex` UTC) a la partición pedida.

    Por defecto, `Partition.DEVELOPMENT`: cualquier código que no pida
    explícitamente el holdout nunca lo ve. Pedir `Partition.HOLDOUT` emite
    un `logging.warning` — el holdout se consume, no se consulta de paso.

    Args:
        df: DataFrame con índice `DatetimeIndex` timezone-aware.
        partition: `Partition.DEVELOPMENT` (por defecto) o `Partition.HOLDOUT`.

    Returns:
        Subconjunto de `df` dentro de la partición pedida.
    """
    if partition is Partition.HOLDOUT:
        logger.warning(
            "accediendo a la partición reservada (holdout, >= %s): una vez "
            "observada, deja de servir como validación independiente del "
            "sesgo de selección",
            HOLDOUT_START.date(),
        )
        return df[df.index >= HOLDOUT_START]

    # DEVELOPMENT_END (2019-12-31) es la fecha inclusive documentada; se
    # implementa como "< HOLDOUT_START" para incluir toda la última barra
    # del 31 de diciembre de 2019 sea cual sea el timeframe (una barra D1
    # esa fecha, o la última barra H1 del día, ambas tienen timestamp de
    # apertura ese mismo día, siempre < 2020-01-01).
    return df[df.index < HOLDOUT_START]
