"""Doce medias móviles, implementadas desde cero (sin `pandas-ta` ni `TA-Lib`).

La pregunta central del proyecto es si el *tipo* de media importa para las
hipótesis de reversión a la media y de pullback de continuación, así que las
fórmulas viven aquí, no ocultas en una dependencia de terceros.

Convención común a las doce funciones:

- Firma homogénea: reciben una `pd.Series` de precios (más los parámetros
  propios de cada una, con nombre y valor por defecto) y devuelven una
  `pd.Series` del mismo índice y longitud.
- Las primeras posiciones para las que no hay datos suficientes son `NaN`.
  Nunca se rellena hacia atrás ni hacia delante.
- Son funciones puras: misma entrada, misma salida, sin estado ni efectos
  laterales. No miran al futuro — cada valor en `t` se calcula solo con
  datos de `t` y anteriores.

Todas las medias que se basan en EMA (DEMA, TEMA, ZLEMA, T3) usan la misma
convención de arranque que `ema`: la primera media se siembra con la SMA de
los primeros `period` valores válidos, y la recursión exponencial continúa a
partir de ahí. Esto es lo que se calcula a mano en los tests — no es la
única convención posible (algunas plataformas siembran con el primer precio
directamente), pero es una elección explícita, documentada y consistente en
todo el módulo.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = [
    "MOVING_AVERAGES",
    "alma",
    "dema",
    "ema",
    "hma",
    "kama",
    "mcginley",
    "sma",
    "t3",
    "tema",
    "vidya",
    "wma",
    "zlema",
]


def sma(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil simple: media aritmética de las últimas `period` barras.

    SMA_t = (1/n) * sum(price_{t-n+1 .. t})

    `period` no tiene un valor "correcto" derivado de ninguna fórmula: es un
    parámetro libre. Se usa 20 como convención por defecto, común a las
    medias de esta familia, sin más significado que ese.
    """
    return price.rolling(window=period, min_periods=period).mean()


def ema(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil exponencial, sembrada con la SMA de los primeros `period` valores.

    EMA_{t0} = SMA(price_0..t0, period), con t0 el índice del primer valor
    válido tras `period` observaciones.
    EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}, alpha = 2 / (period + 1)

    Robusta a que `price` tenga ya un prefijo de `NaN` (por ejemplo, al
    componer EMA de EMA en DEMA/TEMA/T3): el arranque se calcula a partir
    del primer valor no-NaN de `price`, no de la posición 0 del índice.

    Referencia: J. Welles Wilder / práctica estándar de medias exponenciales.
    """
    result = pd.Series(np.nan, index=price.index, dtype="float64")
    valid = price.dropna()
    if len(valid) < period:
        return result

    alpha = 2.0 / (period + 1)
    seed = valid.iloc[:period].mean()
    tail = valid.iloc[period - 1 :].copy()
    tail.iloc[0] = seed
    smoothed = tail.ewm(alpha=alpha, adjust=False).mean()
    result.loc[smoothed.index] = smoothed.to_numpy()
    return result


def _weighted_rolling(price: pd.Series, period: int, weights: np.ndarray) -> pd.Series:
    """Media móvil ponderada genérica: `weights` se aplica al orden [antiguo..reciente]."""
    weights_sum = weights.sum()

    def _dot(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / weights_sum)

    return price.rolling(window=period, min_periods=period).apply(_dot, raw=True)


def wma(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil ponderada linealmente: los pesos crecen 1, 2, ..., n hacia el presente.

    WMA_t = sum(i * price_{t-n+i}, i=1..n) / sum(i, i=1..n)
    """
    weights = np.arange(1, period + 1, dtype="float64")
    return _weighted_rolling(price, period, weights)


def hma(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil de Hull: reduce el retardo combinando dos WMA y volviendo a suavizar.

    HMA_t = WMA( 2 * WMA(price, n/2) - WMA(price, n), round(sqrt(n)) )

    `n/2` y `sqrt(n)` se redondean al entero más cercano (mínimo 1). Fuente:
    Alan Hull, "Hull Moving Average" (2005).
    """
    half_period = max(1, round(period / 2))
    sqrt_period = max(1, round(np.sqrt(period)))
    raw = 2 * wma(price, half_period) - wma(price, period)
    return wma(raw, sqrt_period)


def dema(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil exponencial doble: reduce el retardo respecto a la EMA simple.

    DEMA_t = 2 * EMA(price, n)_t - EMA(EMA(price, n), n)_t

    Fuente: Patrick Mulloy, "Smoothing Data with Faster Moving Averages"
    (Technical Analysis of Stocks & Commodities, 1994).
    """
    ema1 = ema(price, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def tema(price: pd.Series, period: int = 20) -> pd.Series:
    """Media móvil exponencial triple: reduce el retardo aún más que la DEMA.

    TEMA_t = 3*EMA1_t - 3*EMA2_t + EMA3_t, donde
    EMA1 = EMA(price, n), EMA2 = EMA(EMA1, n), EMA3 = EMA(EMA2, n)

    Fuente: Patrick Mulloy, "Smoothing Data with Faster Moving Averages"
    (Technical Analysis of Stocks & Commodities, 1994).
    """
    ema1 = ema(price, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def zlema(price: pd.Series, period: int = 20) -> pd.Series:
    """EMA de retardo cero: aplica la EMA sobre una serie "des-retardada".

    lag = round((period - 1) / 2)
    des_retardada_t = price_t + (price_t - price_{t-lag}) = 2*price_t - price_{t-lag}
    ZLEMA_t = EMA(des_retardada, period)_t

    Fuente: John Ehlers y Ric Way, "Zero Lag (Well, Almost)" (2010).
    """
    lag = round((period - 1) / 2)
    de_lagged = 2 * price - price.shift(lag)
    return ema(de_lagged, period)


def kama(
    price: pd.Series,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """Media adaptativa de Kaufman: se acelera en tendencia, se frena en ruido.

    ER_t = |price_t - price_{t-er_period}| / sum(|price_i - price_{i-1}|, i en la ventana)
    SC_t = (ER_t * (fastSC - slowSC) + slowSC)^2, con
        fastSC = 2/(fast_period+1), slowSC = 2/(slow_period+1)
    KAMA_{er_period} = price_{er_period}  (siembra)
    KAMA_t = KAMA_{t-1} + SC_t * (price_t - KAMA_{t-1})   para t > er_period

    `er_period=10`, `fast_period=2` y `slow_period=30` son los valores
    originales de Perry J. Kaufman, "Trading Systems and Methods" (1998) —
    no son una elección nuestra, son la definición del indicador.

    Fórmula intrínsecamente recursiva: se implementa con un bucle explícito
    en vez de un truco vectorizado.
    """
    n = len(price)
    if n <= er_period:
        return pd.Series(np.nan, index=price.index, dtype="float64")

    values = price.to_numpy()
    diffs = np.abs(np.diff(values))  # diffs[i] = |values[i+1] - values[i]|
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)

    kama_values = np.full(n, np.nan)
    kama_values[er_period] = values[er_period]

    volatility = np.convolve(diffs, np.ones(er_period), mode="valid")  # ventanas de er_period diffs

    for t in range(er_period + 1, n):
        change = abs(values[t] - values[t - er_period])
        # volatility[t - er_period] = sum(diffs[t-er_period .. t-1])
        vol = volatility[t - er_period]
        er = 0.0 if vol == 0 else change / vol
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_values[t] = kama_values[t - 1] + sc * (values[t] - kama_values[t - 1])

    return pd.Series(kama_values, index=price.index, dtype="float64")


def alma(
    price: pd.Series,
    period: int = 20,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pd.Series:
    """Media móvil de Arnaud Legoux: pesos gaussianos desplazados hacia el presente.

    m = offset * (period - 1)
    s = period / sigma
    w_i = exp(-(i - m)^2 / (2 * s^2)),  i = 0..period-1  (0 = barra más antigua)
    ALMA_t = sum(w_i * price_{t-period+1+i}) / sum(w_i)

    `offset=0.85` y `sigma=6` son los valores recomendados en el paper
    original: Arnaud Legoux y Dimitrios Kouzis-Loukas, "ALMA — Arnaud Legoux
    Moving Average" (2009).
    """
    m = offset * (period - 1)
    s = period / sigma
    i = np.arange(period, dtype="float64")
    weights = np.exp(-((i - m) ** 2) / (2 * s**2))
    return _weighted_rolling(price, period, weights)


def t3(price: pd.Series, period: int = 20, v: float = 0.7) -> pd.Series:
    """T3 de Tillson: seis EMA anidadas combinadas para suavizar sin apenas retardo.

    e1 = EMA(price, n); e2 = EMA(e1, n); ...; e6 = EMA(e5, n)
    c1 = -v^3
    c2 = 3v^2 + 3v^3
    c3 = -6v^2 - 3v - 3v^3
    c4 = 1 + 3v + v^3 + 3v^2
    T3_t = c1*e6_t + c2*e5_t + c3*e4_t + c4*e3_t

    Nótese que c1+c2+c3+c4 = 1 siempre (se cancelan los términos en v): sobre
    una serie constante, T3 reproduce exactamente esa constante — es la
    comprobación de coherencia usada en los tests.

    `v=0.7` es el "volume factor" que recomienda el propio Tillson. Fuente:
    Tim Tillson, "Smoothing Techniques For More Accurate Signals" (Technical
    Analysis of Stocks & Commodities, 1998).
    """
    e1 = ema(price, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    c1 = -(v**3)
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v**2

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def vidya(price: pd.Series, period: int = 14) -> pd.Series:
    """Media dinámica de índice variable: EMA cuya velocidad depende del momentum (CMO).

    CMO_t (Chande Momentum Oscillator) sobre una ventana de `period` cambios:
        up = suma de subidas positivas, down = suma de bajadas (en valor absoluto)
        CMO_t = 100 * (up - down) / (up + down)
    alpha = 2 / (period + 1)
    VIDYA_{period} = price_{period}  (siembra)
    VIDYA_t = VIDYA_{t-1} + alpha * |CMO_t / 100| * (price_t - VIDYA_{t-1})

    El periodo del CMO se fija igual a `period` (un único parámetro, en vez
    de un hiperparámetro adicional) — es la simplificación más directa de la
    definición y evita introducir un segundo periodo sin fuente canónica que
    lo fije. Fuente: Tushar Chande, "Adapting Moving Averages to Market
    Volatility" (Technical Analysis of Stocks & Commodities, 1992).

    Fórmula intrínsecamente recursiva: se implementa con un bucle explícito.
    """
    n = len(price)
    if n <= period:
        return pd.Series(np.nan, index=price.index, dtype="float64")

    values = price.to_numpy()
    diffs = np.diff(values)  # diffs[i] = values[i+1] - values[i]
    alpha = 2.0 / (period + 1)

    vidya_values = np.full(n, np.nan)
    vidya_values[period] = values[period]

    for t in range(period + 1, n):
        window = diffs[t - period : t]  # los `period` cambios que terminan en t
        up = window[window > 0].sum()
        down = -window[window < 0].sum()
        cmo = 0.0 if (up + down) == 0 else 100.0 * (up - down) / (up + down)
        eff_alpha = alpha * abs(cmo) / 100.0
        vidya_values[t] = vidya_values[t - 1] + eff_alpha * (values[t] - vidya_values[t - 1])

    return pd.Series(vidya_values, index=price.index, dtype="float64")


def mcginley(price: pd.Series, period: int = 14) -> pd.Series:
    """McGinley Dynamic: se auto-ajusta a la velocidad del precio para reducir el "whipsaw".

    Sembrada con la SMA de los primeros `period` valores (misma convención
    que `ema`, ya que la fórmula original de McGinley no fija una siembra):
        MD_{t0} = SMA(price_0..t0, period)
    MD_t = MD_{t-1} + (price_t - MD_{t-1}) / (N * (price_t / MD_{t-1})^4), con N = period

    `period=14` sigue la convención más citada de John R. McGinley
    ("Study Aid: Technical Analysis Course", Market Technicians Association).

    Fórmula intrínsecamente recursiva: se implementa con un bucle explícito.
    Nótese que si price_t == MD_{t-1}, el numerador es 0 y MD no cambia —
    en una serie constante, McGinley reproduce exactamente esa constante.
    """
    n = len(price)
    if n < period:
        return pd.Series(np.nan, index=price.index, dtype="float64")

    values = price.to_numpy()
    md_values = np.full(n, np.nan)
    md_values[period - 1] = values[:period].mean()

    for t in range(period, n):
        prev = md_values[t - 1]
        md_values[t] = prev + (values[t] - prev) / (period * (values[t] / prev) ** 4)

    return pd.Series(md_values, index=price.index, dtype="float64")


MOVING_AVERAGES: dict[str, Callable[[pd.Series], pd.Series]] = {
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "hma": hma,
    "dema": dema,
    "tema": tema,
    "kama": kama,
    "alma": alma,
    "zlema": zlema,
    "t3": t3,
    "vidya": vidya,
    "mcginley": mcginley,
}
"""Registro nombre -> función, con los parámetros por defecto de cada una.

Las fases posteriores (barrido de parámetros, etc.) pueden iterar este
diccionario sin conocer los doce tipos de media uno a uno.
"""
