"""Bandas de Bollinger y Canales de Keltner.

Ambas aceptan cualquiera de las doce medias de
`fxlab.indicators.moving_averages` (o cualquier otra función `Series ->
Series`) como línea central: se le pasa `price` y nada más, usando los
parámetros por defecto de esa media. Si se quiere otro periodo para la
media central, se pasa ya aplicado, p.ej.:

    from functools import partial
    from fxlab.indicators.moving_averages import kama
    bollinger(price, ma_func=partial(kama, er_period=5))

El parámetro `period` de `bollinger`/`keltner` NO es el periodo de la media
central (eso lo decide `ma_func`): es la ventana de la desviación típica o
del ATR que define el ancho de las bandas. Son conceptualmente
independientes, por eso están desacoplados.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from fxlab.indicators._true_range import DEFAULT_ATR_PERIOD, atr
from fxlab.indicators.moving_averages import ema, sma

DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_K = 2.0


def bollinger(
    price: pd.Series,
    ma_func: Callable[[pd.Series], pd.Series] = sma,
    *,
    period: int = DEFAULT_BOLLINGER_PERIOD,
    k: float = DEFAULT_K,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger: línea central ± k desviaciones típicas del precio.

    superior_t = central_t + k * std_t
    inferior_t = central_t - k * std_t
    std_t = desviación típica móvil de `price` sobre `period` barras (no de
        la línea central: así el ancho de banda mide la volatilidad del
        precio en sí, sea cual sea la media usada como centro)

    Definición clásica (John Bollinger): `ma_func=sma`, `period=20`, `k=2`
    — la misma ventana de 20 para la SMA y para la desviación típica. Son
    los valores por defecto de esta función.

    Returns:
        (superior, central, inferior), todas del mismo índice que `price`.
    """
    central = ma_func(price)
    std = price.rolling(window=period, min_periods=period).std()
    upper = central + k * std
    lower = central - k * std
    return upper, central, lower


def keltner(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ma_func: Callable[[pd.Series], pd.Series] = ema,
    *,
    period: int = DEFAULT_ATR_PERIOD,
    k: float = DEFAULT_K,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Canales de Keltner: línea central ± k rangos verdaderos medios (ATR).

    superior_t = central_t + k * ATR_t
    inferior_t = central_t - k * ATR_t

    Definición clásica ORIGINAL (Chester Keltner, 1960, "10 Day Moving
    Average Rule"): SMA del precio típico ± el rango simple (máximo-mínimo),
    sin ATR. La versión que se usa casi universalmente hoy — y la que se
    toma aquí como valor por defecto — sustituye la media por una EMA y el
    rango simple por el ATR de Wilder: `ma_func=ema`, `period=14` (el
    periodo clásico de Wilder para el ATR), `k=2`.

    Returns:
        (superior, central, inferior), todas del mismo índice que `price`.
    """
    central = ma_func(price)
    atr_series = atr(high, low, price, period=period)
    upper = central + k * atr_series
    lower = central - k * atr_series
    return upper, central, lower
