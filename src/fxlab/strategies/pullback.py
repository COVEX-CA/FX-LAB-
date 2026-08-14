"""Pullback de continuación: dentro de una tendencia, el precio retrocede
hasta una media rápida y, si confirma, se opera a favor de la tendencia.

Reglas (caso alcista; el bajista es la imagen especular exacta):

1. **Tendencia**: el cierre de la barra `t` está por encima de una media
   lenta: `close_t > slow_ma_t`.
2. **Pullback**: en `t`, el mínimo de la barra toca o cruza por debajo de
   una media rápida: `low_t <= fast_ma_t` (comparación contra el mínimo,
   no el cierre — así "toca" incluye el caso exacto).
3. **Confirmación**: la barra siguiente, `t+1`, cierra por encima de su
   propia apertura (vela alcista, `close_{t+1} > open_{t+1}`) y por encima
   de la media rápida en `t+1` (`close_{t+1} > fast_ma_{t+1}`).
4. **Entrada**: apertura de la barra posterior a la de confirmación
   (`t+2`).
5. **Salida**: apertura de la barra `entrada + n_bars`. Sin stop, sin
   objetivo, sin gestión — la posición se mantiene exactamente `n_bars`
   barras y se cierra en su apertura, simétrico con el mecanismo de
   entrada (que también ocurre en una apertura).
6. **Filtro de régimen (opcional)**: ADX en `t` por encima de un umbral.
   Se puede activar o desactivar (`use_adx_filter`); no tiene valor por
   defecto para el umbral — quien active el filtro debe decidir el
   número, no esta función. Es exactamente la pregunta abierta de la
   investigación: si este filtro aporta algo.

Sin lookahead por construcción: toda la lógica se expresa con
`Series.shift(k)` con `k >= 0` (nunca negativo), que solo trae valores de
posiciones *anteriores* a la actual. No hay ninguna ventana centrada ni
operación que recorra la serie hacia atrás desde el final.

Esta función devuelve señales — instantes de entrada/salida con dirección
— no órdenes ni tamaños. La conversión a operaciones con costes la hace
`fxlab.sweep`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from fxlab.indicators.trend_strength import DEFAULT_ADX_PERIOD, adx

MovingAverage = Callable[[pd.Series], pd.Series]


@dataclass(frozen=True)
class PullbackSignals:
    """Señales de entrada/salida, largas y cortas, alineadas al índice de entrada.

    Cada campo es una `pd.Series` booleana: `True` en las barras donde
    ocurre ese evento (entrada o salida), `False` en el resto. No son
    posiciones ni tamaños — el motor de barrido decide cómo ejecutarlas.
    """

    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series


def generate_signals(
    df: pd.DataFrame,
    slow_ma_func: MovingAverage,
    fast_ma_func: MovingAverage,
    n_bars: int,
    *,
    use_adx_filter: bool = False,
    adx_threshold: float | None = None,
    adx_period: int = DEFAULT_ADX_PERIOD,
) -> PullbackSignals:
    """Genera señales de pullback de continuación, largas y cortas.

    Args:
        df: DataFrame OHLC (columnas `open`, `high`, `low`, `close`), con
            el contrato de datos de `fxlab.data` (índice UTC ordenado).
        slow_ma_func: media "lenta" que define la tendencia. Cualquiera de
            las doce de `fxlab.indicators.moving_averages`, ya configurada
            con su período (p.ej. `functools.partial(ema, period=50)`) —
            se llama solo con la serie de cierre, igual que `ma_func` en
            `fxlab.indicators.bands`.
        fast_ma_func: media "rápida" contra la que se mide el pullback,
            misma convención que `slow_ma_func`.
        n_bars: barras que se mantiene la posición tras la entrada. Debe
            ser >= 1.
        use_adx_filter: si `True`, exige además que el ADX en la barra de
            pullback (`t`) esté por encima de `adx_threshold`.
        adx_threshold: umbral de ADX. Obligatorio si `use_adx_filter=True`
            — no tiene valor por defecto a propósito, para no decidir en
            silencio qué cuenta como "régimen de tendencia".
        adx_period: período del ADX, solo si `use_adx_filter=True`.

    Returns:
        `PullbackSignals` con las cuatro series booleanas, del mismo
        índice que `df`.

    Raises:
        ValueError: si `n_bars < 1`, o si `use_adx_filter=True` sin
            `adx_threshold`.
    """
    if n_bars < 1:
        raise ValueError(f"n_bars debe ser >= 1, se recibió {n_bars}")
    if use_adx_filter and adx_threshold is None:
        raise ValueError("adx_threshold es obligatorio cuando use_adx_filter=True")

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    slow_ma = slow_ma_func(close)
    fast_ma = fast_ma_func(close)

    uptrend = close > slow_ma
    downtrend = close < slow_ma

    pullback_up = low <= fast_ma  # toca o cruza por debajo (regla 2)
    pullback_down = high >= fast_ma  # simétrico: toca o cruza por encima

    bullish_bar = close > open_
    bearish_bar = close < open_

    confirm_long = (close > fast_ma) & bullish_bar
    confirm_short = (close < fast_ma) & bearish_bar

    setup_long = uptrend & pullback_up
    setup_short = downtrend & pullback_down

    if use_adx_filter:
        adx_value, _plus_di, _minus_di = adx(high, low, close, period=adx_period)
        regime_ok = adx_value > adx_threshold
        setup_long = setup_long & regime_ok
        setup_short = setup_short & regime_ok

    # la confirmación ocurre en t+1: se trae el setup de t (pasado) a la
    # posición t+1 con shift(1), nunca al revés.
    confirmation_long = setup_long.shift(1, fill_value=False) & confirm_long
    confirmation_short = setup_short.shift(1, fill_value=False) & confirm_short

    # entrada: apertura de la barra posterior a la de confirmación (t+2)
    long_entries = confirmation_long.shift(1, fill_value=False)
    short_entries = confirmation_short.shift(1, fill_value=False)

    # salida: apertura de la barra entrada + n_bars
    long_exits = long_entries.shift(n_bars, fill_value=False)
    short_exits = short_entries.shift(n_bars, fill_value=False)

    # una entrada cuya salida caería más allá de los datos disponibles no
    # se puede cerrar: se descarta en vez de fabricar una barra futura.
    if n_bars <= len(df):
        long_entries.iloc[-n_bars:] = False
        short_entries.iloc[-n_bars:] = False
    else:
        long_entries.iloc[:] = False
        short_entries.iloc[:] = False

    return PullbackSignals(
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )
