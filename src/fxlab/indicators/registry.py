"""Registro único de todos los indicadores públicos del paquete.

Existe para que los tests (y, en fases futuras, el barrido de parámetros)
puedan recorrer *todos* los indicadores — no solo las doce medias — sin
conocerlos uno a uno. En concreto, es lo que permite:

1. un test de no-lookahead genérico que cubre `moving_averages`,
   `distance`, `bands` y `trend_strength` a la vez, y
2. un test de completitud que falla si alguna función pública de indicador
   se queda sin registrar (ver `tests/indicators/test_lookahead.py`).

Cada entrada de `INDICATORS` envuelve al indicador real en un adaptador de
firma uniforme: recibe un DataFrame con columnas "high", "low", "close" y
devuelve una `Series` o un `DataFrame`. Los parámetros propios de cada
indicador que son escalares (p.ej. `method` en `distance`) se fijan con
`functools.partial`; las series que dependen de los propios datos (p.ej. la
media central de las bandas, o la `ma` que pide `distance`) no se pueden
fijar como constante y se calculan dentro del adaptador.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pandas as pd

from fxlab.indicators.bands import bollinger, keltner
from fxlab.indicators.distance import distance
from fxlab.indicators.moving_averages import MOVING_AVERAGES, sma
from fxlab.indicators.trend_strength import adx

Indicator = Callable[[pd.DataFrame], "pd.Series | pd.DataFrame"]

# Funciones realmente registradas, para el test de completitud: no es lo
# mismo que "está en INDICATORS.values()", porque los valores del registro
# son los adaptadores (closures), no las funciones originales.
_covered: set[Callable[..., object]] = set()


def _cover(*funcs: Callable[..., object]) -> None:
    _covered.update(funcs)


def _ma_adapter(func: Callable[[pd.Series], pd.Series]) -> Indicator:
    _cover(func)

    def adapter(ohlc: pd.DataFrame) -> pd.Series:
        return func(ohlc["close"])

    return adapter


def _distance_adapter(method: str) -> Indicator:
    _cover(distance)
    bound = partial(distance, method=method)  # fija el parámetro propio "method"

    def adapter(ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        ma = sma(close)
        if method == "atr":
            return bound(close, ma, high=ohlc["high"], low=ohlc["low"])
        return bound(close, ma)

    return adapter


def _bollinger_adapter(ohlc: pd.DataFrame) -> pd.DataFrame:
    upper, central, lower = bollinger(ohlc["close"])
    return pd.DataFrame({"upper": upper, "central": central, "lower": lower})


def _keltner_adapter(ohlc: pd.DataFrame) -> pd.DataFrame:
    upper, central, lower = keltner(ohlc["close"], ohlc["high"], ohlc["low"])
    return pd.DataFrame({"upper": upper, "central": central, "lower": lower})


def _adx_adapter(ohlc: pd.DataFrame) -> pd.DataFrame:
    adx_series, plus_di, minus_di = adx(ohlc["high"], ohlc["low"], ohlc["close"])
    return pd.DataFrame({"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di})


_cover(bollinger, keltner, adx)

INDICATORS: dict[str, Indicator] = {
    **{name: _ma_adapter(func) for name, func in MOVING_AVERAGES.items()},
    "distance_std": _distance_adapter("std"),
    "distance_atr": _distance_adapter("atr"),
    "bollinger": _bollinger_adapter,
    "keltner": _keltner_adapter,
    "adx": _adx_adapter,
}
"""Nombre -> adaptador (DataFrame OHLC -> Series | DataFrame)."""

COVERED_FUNCTIONS: frozenset[Callable[..., object]] = frozenset(_covered)
"""Funciones de indicador realmente representadas en `INDICATORS`, para que
`tests/indicators/test_lookahead.py` pueda comprobar que ninguna función
pública del paquete se ha quedado sin registrar."""
