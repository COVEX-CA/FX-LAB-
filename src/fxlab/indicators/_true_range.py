"""Rango verdadero (True Range) y ATR: cálculo compartido por Keltner
(`bands.py`), `distance.py` (método "atr") y ADX (`trend_strength.py`).

No es un indicador nuevo añadido por su cuenta: es la pieza de cálculo que
esos tres indicadores ya pedidos necesitan, factorizada una sola vez para no
triplicar la fórmula (y el riesgo de que diverjan).

El suavizado usado es el de Wilder (`wilder_smooth`, alpha=1/period), no una
media simple ni una EMA "clásica" (alpha=2/(period+1)), y es el mismo para
los tres consumidores — nunca una copia paralela. Dos razones, no solo una
preferencia de estilo:

1. Es la definición histórica: "ATR" significa, por convención desde 1978,
   rango verdadero suavizado a la manera de Wilder. Suavizarlo con una media
   simple produciría un número distinto que ya no sería un ATR real, sino
   otro indicador con el mismo nombre.
2. En ADX específicamente, no es opcional: Wilder diseñó +DI/-DI con
   `100 * suavizado(+DM) / suavizado(TR)`. Si TR y +DM/-DM se suavizaran con
   operadores distintos (p.ej. TR con una media simple y +DM con el
   suavizado de Wilder), el cociente dejaría de significar "proporción del
   rango verdadero explicada por el movimiento direccional" — perdería la
   interpretación que le da sentido al indicador, aunque el cálculo siguiera
   siendo numéricamente válido. Por eso `adx()` en `trend_strength.py`
   importa `wilder_smooth` de aquí y lo aplica igual a TR, +DM y -DM (y
   luego a DX->ADX), en vez de tener su propio suavizado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ATR_PERIOD = 14


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Rango verdadero de Wilder: el mayor movimiento intrabarra, contando gaps.

    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)

    `close_{t-1}` no existe en `t=0`, así que TR_0 se define como
    `high_0 - low_0` (no hay barra anterior con la que medir un gap).

    Fuente: J. Welles Wilder, "New Concepts in Technical Trading Systems"
    (1978).
    """
    prev_close = close.shift(1)
    high_low = high - low
    high_prev_close = (high - prev_close).abs()
    low_prev_close = (low - prev_close).abs()

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    tr.iloc[0] = high_low.iloc[0]
    return tr


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Suavizado de Wilder: EMA con alpha = 1/period, sembrada con la SMA de
    los primeros `period` valores válidos de `series`.

    Es el suavizado que Wilder usa para ATR, +DM, -DM y DX->ADX — distinto
    del alpha=2/(period+1) de una EMA "clásica". Debe ser el mismo para los
    tres en ADX: el cociente `100 * suavizado(+DM) / suavizado(TR)` solo
    significa "proporción del rango verdadero explicada por el movimiento
    direccional" si numerador y denominador se suavizan con el mismo
    operador (ver docstring del módulo). Robusto a que `series` empiece con
    `NaN` (p.ej. `true_range` no está definida hasta que hay suficientes
    barras previas para otros cálculos encadenados).
    """
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = series.dropna()
    if len(valid) < period:
        return result

    alpha = 1.0 / period
    seed = valid.iloc[:period].mean()
    tail = valid.iloc[period - 1 :].copy()
    tail.iloc[0] = seed
    smoothed = tail.ewm(alpha=alpha, adjust=False).mean()
    result.loc[smoothed.index] = smoothed.to_numpy()
    return result


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = DEFAULT_ATR_PERIOD
) -> pd.Series:
    """ATR (Average True Range): suavizado de Wilder del rango verdadero.

    ATR_t = suavizado_de_Wilder(TR, period)_t

    `period=14` es el valor original de Wilder, que se reutiliza como
    convención en Keltner, en `distance()` y en ADX.
    """
    return wilder_smooth(true_range(high, low, close), period)
