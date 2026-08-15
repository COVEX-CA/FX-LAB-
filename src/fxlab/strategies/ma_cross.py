"""Cruce de medias móviles: la estrategia seguidora de tendencia más simple.

Reglas (caso alcista; el bajista es la imagen especular exacta):

1. **Cruce alcista**: en la barra `t`, la media rápida cruza por ENCIMA de la
   lenta: `fast_t > slow_t` y `fast_{t-1} <= slow_{t-1}`. La comparación con
   la barra anterior es lo que distingue un *cruce* (evento puntual) de estar
   simplemente por encima (estado).
2. **Entrada larga**: en la barra siguiente, `t+1`. Nunca en la misma barra
   del cruce: el cruce solo se conoce al cierre de `t`, así que operar en `t`
   sería usar ese cierre en el mismo instante en que se forma. Entrar en `t+1`
   lo evita sin ambigüedad.
3. **Salida / reversa**: no hay stop ni objetivo ni tiempo fijo. La posición
   se mantiene hasta el cruce contrario, que **cierra el largo y abre un
   corto** a la vez (y viceversa). Es una estrategia siempre en mercado.

A diferencia de `fxlab.strategies.pullback`, aquí NO hay `n_bars`, ni
retroceso, ni confirmación de vela: es el cruce puro, el del indicador clásico.

Sin lookahead por construcción: el cruce se calcula con `Series.shift(k)` con
`k >= 0` (solo trae valores de posiciones anteriores) y las entradas se
desplazan a `t+1`. Ninguna señal en `t` depende de datos de `t+1` o posterior.

Esta función devuelve señales — instantes de entrada/salida con dirección — no
órdenes ni tamaños. La conversión a operaciones con costes la hace `fxlab.sweep`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

MovingAverage = Callable[[pd.Series], pd.Series]


@dataclass(frozen=True)
class CrossSignals:
    """Señales de entrada/salida, largas y cortas, del cruce de medias.

    Cada campo es una `pd.Series` booleana: `True` en las barras donde ocurre
    ese evento, `False` en el resto. Como la estrategia se da la vuelta en cada
    cruce, la salida de un lado coincide con la entrada del otro:
    `long_exits == short_entries` y `short_exits == long_entries`.
    """

    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series


def generate_signals(
    df: pd.DataFrame,
    fast_ma_func: MovingAverage,
    slow_ma_func: MovingAverage,
) -> CrossSignals:
    """Genera señales de cruce de medias, largas y cortas.

    Args:
        df: DataFrame OHLC (al menos la columna `close`), con el contrato de
            datos de `fxlab.data` (índice UTC ordenado).
        fast_ma_func: media "rápida", ya configurada con su período y tipo
            (p.ej. `functools.partial(ema, period=20)`). Se llama solo con la
            serie de cierre, misma convención que en `fxlab.strategies.pullback`.
        slow_ma_func: media "lenta", misma convención. Puede ser de un tipo
            distinto al de la rápida (p.ej. EMA rápida contra SMA lenta): tipo y
            período son independientes.

    Returns:
        `CrossSignals` con las cuatro series booleanas, del mismo índice que `df`.
    """
    close = df["close"]
    fast = fast_ma_func(close)
    slow = slow_ma_func(close)

    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    # Cruce estricto (como `ta.crossover`/`ta.crossunder`): la rápida pasa de
    # estar en un lado a estar ESTRICTAMENTE en el otro. Que solo se toquen
    # (`fast == slow`) no es un cruce y no emite señal. Las comparaciones con
    # NaN del calentamiento dan False, así que no hay señales espurias mientras
    # las medias aún no son válidas.
    cross_up = (fast > slow) & (prev_fast <= prev_slow)
    cross_down = (fast < slow) & (prev_fast >= prev_slow)

    # entrada en t+1: el cruce se confirma al cierre de t, se opera en la barra
    # siguiente (nunca en la misma barra del cruce).
    long_entries = cross_up.shift(1, fill_value=False)
    short_entries = cross_down.shift(1, fill_value=False)

    # reversa: el cruce contrario cierra la posición actual y abre la opuesta.
    long_exits = short_entries
    short_exits = long_entries

    return CrossSignals(
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )
