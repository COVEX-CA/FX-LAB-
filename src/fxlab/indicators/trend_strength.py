"""ADX (Average Directional Index) y sus componentes +DI / -DI.

El ADX mide la **fuerza** de una tendencia, no su dirección: un ADX alto
significa "hay tendencia" tanto si el precio sube como si baja; +DI y -DI
son los que indican de qué lado. Es, conceptualmente, el indicador que
distingue el régimen favorable a cada una de las dos hipótesis del proyecto
(reversión a la media en rango lateral, pullback de continuación en
tendencia) — pero esta fase solo calcula el indicador. Qué umbral de ADX
separa "hay tendencia" de "no la hay", y qué hipótesis aplicar en cada caso,
es una decisión de una fase posterior (señales), fuera de alcance aquí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.indicators._true_range import DEFAULT_ATR_PERIOD, true_range, wilder_smooth

DEFAULT_ADX_PERIOD = DEFAULT_ATR_PERIOD  # 14, el periodo original de Wilder


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = DEFAULT_ADX_PERIOD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX y sus componentes direccionales +DI / -DI, según Wilder.

    up_t = high_t - high_{t-1}
    down_t = low_{t-1} - low_t
    +DM_t = up_t   si up_t > down_t y up_t > 0, si no 0
    -DM_t = down_t si down_t > up_t y down_t > 0, si no 0
    TR_t = rango verdadero (ver `fxlab.indicators._true_range.true_range`)

    +DI_t = 100 * suavizado_Wilder(+DM, period)_t / suavizado_Wilder(TR, period)_t
    -DI_t = 100 * suavizado_Wilder(-DM, period)_t / suavizado_Wilder(TR, period)_t
    DX_t  = 100 * |+DI_t - -DI_t| / (+DI_t + -DI_t)
    ADX_t = suavizado_Wilder(DX, period)_t

    El suavizado de Wilder (alpha=1/period) se aplica dos veces en cadena
    (una vez para TR/+DM/-DM, otra para DX->ADX), así que el ADX tarda
    aproximadamente `2*period` barras en producir su primer valor no-NaN.

    `period=14` es el valor original de Wilder. Fuente: J. Welles Wilder,
    "New Concepts in Technical Trading Systems" (1978).

    Args:
        high: serie de máximos.
        low: serie de mínimos.
        close: serie de cierres.
        period: periodo de suavizado, tanto para TR/+DM/-DM como para DX->ADX.

    Returns:
        (adx, plus_di, minus_di), todas del mismo índice que `high`.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # No hay barra anterior en la posición 0: el movimiento direccional no
    # está definido ahí (a diferencia del TR, que sí está definido en 0).
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr = true_range(high, low, close)

    smoothed_tr = wilder_smooth(tr, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    # Evita que un 0/0 real (+DI y -DI ambos exactamente 0: sin ningún
    # movimiento direccional) se propague como NaN igual que el hueco de
    # arranque. `di_sum != 0` ya es False solo cuando ambos son 0 de verdad
    # (di_sum NaN mantiene la condición en True y por tanto el NaN original).
    dx = dx.where(di_sum != 0, 0.0)

    adx_series = wilder_smooth(dx, period)

    return adx_series, plus_di, minus_di
