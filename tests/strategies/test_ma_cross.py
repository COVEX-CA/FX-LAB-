from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd

from fxlab.indicators.moving_averages import sma
from fxlab.strategies.ma_cross import generate_signals


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def _df(close: list[float]) -> pd.DataFrame:
    # ma_cross solo usa 'close'; el resto de columnas van por completitud.
    c = pd.Series(close, index=_index(len(close)), dtype="float64")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c})


def _fixed_ma(values: list[float]) -> Callable[[pd.Series], pd.Series]:
    """Media con valores fijados a mano, independiente del precio."""
    series = pd.Series(values, index=_index(len(values)), dtype="float64")
    return lambda _close: series


def test_crossover_and_crossunder_enter_one_bar_after_the_cross() -> None:
    # fast pasa por debajo->encima en t1 (cruce alcista), y encima->debajo en
    # t3 (cruce bajista). Las entradas van una barra despues del cruce.
    fast = _fixed_ma([1.0, 3.0, 3.0, 1.0, 1.0])
    slow = _fixed_ma([2.0, 2.0, 2.0, 2.0, 2.0])
    sig = generate_signals(_df([0.0] * 5), fast, slow)

    assert sig.long_entries.tolist() == [False, False, True, False, False]  # cruce t1 -> entra t2
    assert sig.short_entries.tolist() == [False, False, False, False, True]  # cruce t3 -> entra t4
    # reversa: la salida de un lado es la entrada del otro
    assert sig.long_exits.equals(sig.short_entries)
    assert sig.short_exits.equals(sig.long_entries)


def test_a_mere_touch_is_not_a_cross() -> None:
    # fast toca a slow por abajo en t1 (fast == slow) y vuelve a subir. Tocar no
    # es cruzar: no debe emitir ninguna senal bajista.
    fast = _fixed_ma([3.0, 2.0, 3.0])
    slow = _fixed_ma([2.0, 2.0, 2.0])
    sig = generate_signals(_df([0.0] * 3), fast, slow)

    assert sig.short_entries.sum() == 0
    assert sig.long_exits.sum() == 0


def test_no_signals_while_the_moving_averages_are_warming_up() -> None:
    # Con medias reales, las primeras barras son NaN: no debe salir ninguna
    # senal ahi (las comparaciones con NaN dan False).
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
    sig = generate_signals(_df(close), partial(sma, period=3), partial(sma, period=5))
    warmup = slice(0, 4)  # slow sma(5) aun no es valida
    assert not sig.long_entries[warmup].any()
    assert not sig.short_entries[warmup].any()


def test_no_lookahead_a_future_bar_never_changes_a_past_signal() -> None:
    # Prueba obligatoria (AGENTS.md §7): perturbar una barra futura no puede
    # alterar ninguna senal anterior. Si la hubiera con lookahead, esto falla.
    rng = np.random.default_rng(0)
    close = (100 + np.cumsum(rng.normal(0, 1.0, 60))).tolist()
    fast_f, slow_f = partial(sma, period=3), partial(sma, period=8)

    base = generate_signals(_df(close), fast_f, slow_f)

    j = 45  # barra futura que perturbamos
    perturbed = list(close)
    perturbed[j] *= 1.5
    other = generate_signals(_df(perturbed), fast_f, slow_f)

    for field in ("long_entries", "long_exits", "short_entries", "short_exits"):
        a = getattr(base, field).iloc[:j]
        b = getattr(other, field).iloc[:j]
        assert a.equals(b), f"{field}: una barra futura alteró señales anteriores (lookahead)"
