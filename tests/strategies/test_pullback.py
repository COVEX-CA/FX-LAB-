from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from fxlab.indicators.moving_averages import ema, sma
from fxlab.strategies.pullback import generate_signals


def _ohlc(
    open_: list[float], high: list[float], low: list[float], close: list[float]
) -> pd.DataFrame:
    n = len(open_)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def _const_ma(values: list[float]) -> pd.Series:
    """Serie de "media" con valores fijados a mano, independiente del precio."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


# Escenario base (alcista): 6 barras.
#   idx0: tendencia + pullback (setup)
#   idx1: confirmación
#   idx2: entrada (apertura)
#   idx3: (mantenida)
#   idx4: salida (apertura), con n_bars=2
#   idx5: barra de sobra
#
# slow_ma/fast_ma se fijan muy altas (1e6) fuera de idx0/idx1 para que
# ninguna otra barra dispare tendencia, pullback o confirmación por
# casualidad, en ningún lado (largo o corto).
_OPEN = [105.0, 101.0, 110.0, 111.0, 113.0, 115.0]
_HIGH = [106.0, 108.0, 111.0, 113.0, 115.0, 117.0]
_LOW = [100.0, 100.0, 109.0, 110.0, 112.0, 114.0]
_CLOSE = [104.0, 107.0, 110.5, 112.0, 114.0, 116.0]
_SLOW_MA = [100.0, 1e6, 1e6, 1e6, 1e6, 1e6]
_FAST_MA = [100.0, 105.0, 1e6, 1e6, 1e6, 1e6]


def _base_df() -> pd.DataFrame:
    return _ohlc(_OPEN, _HIGH, _LOW, _CLOSE)


def test_full_valid_sequence_produces_entry_and_exit_at_expected_bars() -> None:
    df = _base_df()
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    expected_entries = [False, False, True, False, False, False]
    expected_exits = [False, False, False, False, True, False]
    assert signals.long_entries.tolist() == expected_entries
    assert signals.long_exits.tolist() == expected_exits
    # el caso alcista no debe disparar ninguna señal corta
    assert not signals.short_entries.any()
    assert not signals.short_exits.any()


def test_exact_touch_of_fast_ma_counts_as_pullback() -> None:
    # low[0] == fast_ma[0] exactamente (100.0 == 100.0): "toca" es inclusive.
    df = _base_df()
    assert _LOW[0] == _FAST_MA[0]
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    assert signals.long_entries.iloc[2]


def test_pullback_without_trend_produces_no_signal() -> None:
    # slow_ma[0] por encima del cierre: no hay tendencia alcista en idx0,
    # aunque el resto de condiciones (pullback, confirmación) se cumplan.
    df = _base_df()
    slow_ma = _const_ma([200.0, 1e6, 1e6, 1e6, 1e6, 1e6])
    fast_ma = _const_ma(_FAST_MA)

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    assert not signals.long_entries.any()
    assert not signals.long_exits.any()


def test_failed_confirmation_bearish_candle_produces_no_signal() -> None:
    # idx1 pasa a ser una vela bajista (close < open): la tendencia y el
    # pullback en idx0 se cumplen, pero la confirmación exige vela alcista.
    open_ = list(_OPEN)
    close = list(_CLOSE)
    open_[1], close[1] = 109.0, 107.0  # close(107) < open(109): bajista
    df = _ohlc(open_, _HIGH, _LOW, close)
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    assert not signals.long_entries.any()


def test_failed_confirmation_close_below_fast_ma_produces_no_signal() -> None:
    # idx1 sigue siendo una vela alcista, pero su cierre no supera la
    # media rápida en t+1: la confirmación exige ambas condiciones.
    df = _base_df()
    slow_ma = _const_ma(_SLOW_MA)
    fast_ma = _const_ma([100.0, 200.0, 1e6, 1e6, 1e6, 1e6])  # fast_ma[1]=200 > close[1]=107

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    assert not signals.long_entries.any()


def test_short_side_is_the_mirror_image() -> None:
    # escenario bajista: abre/cierra invertidos respecto al caso alcista
    # base (idx1 pasa a ser una vela bajista, etc.)
    open_ = [104.0, 107.0, 110.5, 112.0, 114.0, 116.0]
    close = [105.0, 101.0, 110.0, 111.0, 113.0, 115.0]
    high = [106.0, 108.0, 111.0, 113.0, 115.0, 117.0]
    low = [100.0, 100.0, 109.0, 110.0, 112.0, 114.0]
    df = _ohlc(open_, high, low, close)

    # simétrico de _SLOW_MA/_FAST_MA: slow_ma baja (close < slow_ma en idx0
    # => downtrend), fast_ma toca el máximo en idx0 (pullback_down) y queda
    # por debajo del cierre en idx1 (close < fast_ma => confirmación bajista)
    slow_ma = _const_ma([1e6, -1e6, -1e6, -1e6, -1e6, -1e6])
    fast_ma = _const_ma([106.0, 105.0, -1e6, -1e6, -1e6, -1e6])

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)

    expected_entries = [False, False, True, False, False, False]
    expected_exits = [False, False, False, False, True, False]
    assert signals.short_entries.tolist() == expected_entries
    assert signals.short_exits.tolist() == expected_exits
    assert not signals.long_entries.any()
    assert not signals.long_exits.any()


def test_entries_near_end_of_data_are_dropped_not_fabricated() -> None:
    # el setup+confirmación caen justo al final: la salida (entrada+n_bars)
    # se saldría de los datos disponibles, así que la entrada se descarta.
    df = _base_df()
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    signals = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=10)

    assert not signals.long_entries.any()


def test_n_bars_must_be_at_least_one() -> None:
    df = _base_df()
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    with pytest.raises(ValueError, match="n_bars"):
        generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=0)


def test_adx_filter_requires_threshold() -> None:
    df = _base_df()
    slow_ma, fast_ma = _const_ma(_SLOW_MA), _const_ma(_FAST_MA)

    with pytest.raises(ValueError, match="adx_threshold"):
        generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2, use_adx_filter=True)


def test_adx_filter_blocks_signal_when_regime_not_trending() -> None:
    # 40 barras completamente laterales (sin tendencia real): ADX bajo en
    # todo momento. Aunque se fuerce un setup/confirmación de pullback con
    # medias constantes, el filtro debe bloquear la señal.
    n = 40
    rng = np.random.default_rng(0)
    flat = 100.0 + rng.normal(0, 0.05, n)
    open_ = flat.tolist()
    close = (flat + 0.01).tolist()
    high = (flat + 0.5).tolist()
    low = (flat - 0.5).tolist()
    # fuerza setup+confirmación en idx10/idx11 con medias constantes
    close[11] = open_[11] + 1.0
    df = _ohlc(open_, high, low, close)

    slow_values = [1e6] * n
    slow_values[10] = 90.0
    fast_values = [1e6] * n
    fast_values[10] = low[10]
    fast_values[11] = close[11] - 0.5
    slow_ma, fast_ma = _const_ma(slow_values), _const_ma(fast_values)

    signals_unfiltered = generate_signals(df, lambda _: slow_ma, lambda _: fast_ma, n_bars=2)
    assert signals_unfiltered.long_entries.any()  # el setup en sí sí dispara sin filtro

    signals_filtered = generate_signals(
        df,
        lambda _: slow_ma,
        lambda _: fast_ma,
        n_bars=2,
        use_adx_filter=True,
        adx_threshold=90.0,  # umbral inalcanzable en un mercado lateral
    )
    assert not signals_filtered.long_entries.any()


def test_no_lookahead_signal_at_t_ignores_data_after_t() -> None:
    n = 60
    cutoff = 40
    rng = np.random.default_rng(2)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close - rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)

    slow = partial(sma, period=10)
    fast = partial(ema, period=4)

    original = generate_signals(df, slow, fast, n_bars=3)

    modified_df = df.copy()
    tail = modified_df.iloc[cutoff + 1 :]
    modified_df.iloc[cutoff + 1 :] = tail * 1000.0 + 50_000.0
    modified = generate_signals(modified_df, slow, fast, n_bars=3)

    for field in ("long_entries", "long_exits", "short_entries", "short_exits"):
        prefix_original = getattr(original, field).iloc[: cutoff + 1]
        prefix_modified = getattr(modified, field).iloc[: cutoff + 1]
        pd.testing.assert_series_equal(prefix_original, prefix_modified)
