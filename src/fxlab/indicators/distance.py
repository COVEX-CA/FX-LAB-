"""Distancia normalizada del precio a una media.

Una distancia de 20 pips no significa lo mismo en un mercado tranquilo que
en uno agitado, ni en EUR/USD que en GBP/JPY. Sin normalizar por la
volatilidad propia de cada serie y cada época, "el precio está lejos de la
media" no es una magnitud comparable entre pares ni a lo largo del tiempo.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from fxlab.indicators._true_range import DEFAULT_ATR_PERIOD, atr

METHODS = ("std", "atr")


def distance(
    price: pd.Series,
    ma: pd.Series,
    method: str = "atr",
    *,
    period: int = DEFAULT_ATR_PERIOD,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> pd.Series:
    """Distancia normalizada de `price` a `ma`.

    El signo indica el lado: **positivo = precio por encima de la media,
    negativo = precio por debajo**. La magnitud es adimensional (no está en
    precio, pips ni porcentaje) y comparable entre pares y entre épocas,
    porque en ambos métodos se divide por una medida de volatilidad de la
    propia serie.

    Dos métodos de normalización:

    - `"std"`: distancia en unidades de desviación típica de los retornos
      logarítmicos.
        retornos_t = ln(price_t / price_{t-1})
        sigma_t = std_movil(retornos, period)
        distancia_t = ((price_t - ma_t) / ma_t) / sigma_t
      Es decir, "a cuántas desviaciones típicas de retorno equivale el
      porcentaje de distancia a la media".

    - `"atr"`: distancia en unidades de rango verdadero medio (ATR) del
      propio precio (ver `fxlab.indicators._true_range.atr`).
        distancia_t = (price_t - ma_t) / ATR_t

    Args:
        price: serie de precios (cierre).
        ma: serie de media (cualquiera de `fxlab.indicators.moving_averages`).
        method: `"std"` o `"atr"`.
        period: ventana de la desviación típica de retornos, o periodo del
            ATR, según el método. Por defecto 14, igual que el ATR clásico
            de Wilder.
        high: serie de máximos, requerida si `method="atr"`.
        low: serie de mínimos, requerida si `method="atr"`.

    Returns:
        `pd.Series` del mismo índice que `price`, con `NaN` mientras no haya
        suficientes datos para el denominador.
    """
    if method not in METHODS:
        raise ValueError(f"method desconocido: {method!r}, debe ser uno de {METHODS}")

    if method == "std":
        log_returns = cast(pd.Series, np.log(price / price.shift(1)))
        sigma = log_returns.rolling(window=period, min_periods=period).std()
        pct_distance = (price - ma) / ma
        return pct_distance / sigma

    # method == "atr"
    if high is None or low is None:
        raise ValueError('method="atr" requiere pasar high y low')
    atr_series = atr(high, low, price, period=period)
    return (price - ma) / atr_series
