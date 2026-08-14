from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from fxlab.indicators.bands import bollinger, keltner
from fxlab.indicators.moving_averages import ema, kama, sma


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def _hand_ema(values: list[float], period: int) -> list[float]:
    alpha = 2.0 / (period + 1)
    out = [np.nan] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def _hand_atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [high[0] - low[0]]
    for t in range(1, len(high)):
        tr.append(max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1])))

    out = [np.nan] * len(tr)
    if len(tr) < period:
        return out
    seed = sum(tr[:period]) / period
    out[period - 1] = seed
    prev = seed
    for t in range(period, len(tr)):
        prev = prev + (tr[t] - prev) / period
        out[t] = prev
    return out


# --- Bollinger ---------------------------------------------------------


def test_bollinger_hand_calculated_default_sma() -> None:
    price = _series([1, 2, 3, 4, 5, 6, 7])
    # bollinger() llama a ma_func(price) solo con el precio (ver docstring
    # de bands.py): para que la SMA use el mismo periodo que la banda, se
    # preconfigura explícitamente con functools.partial.
    upper, central, lower = bollinger(price, ma_func=partial(sma, period=3), period=3, k=2.0)

    # central = SMA(3); std_movil(3) de una rampa de paso 1 siempre da 1.0
    # (valores a, a+1, a+2 -> media a+1, varianza muestral (1+0+1)/2 = 1)
    expected_central = [np.nan, np.nan, 2, 3, 4, 5, 6]
    expected_upper = [c + 2.0 if not np.isnan(c) else np.nan for c in expected_central]
    expected_lower = [c - 2.0 if not np.isnan(c) else np.nan for c in expected_central]

    np.testing.assert_allclose(central.to_numpy(), expected_central)
    np.testing.assert_allclose(upper.to_numpy(), expected_upper)
    np.testing.assert_allclose(lower.to_numpy(), expected_lower)


def test_bollinger_accepts_any_moving_average_as_centerline() -> None:
    price = _series([1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0, 10.0])

    _, central_sma, _ = bollinger(price, ma_func=partial(sma, period=3))
    _, central_ema, _ = bollinger(price, ma_func=partial(ema, period=3))
    _, central_kama, _ = bollinger(price, ma_func=partial(kama, er_period=3))

    # las tres centrales deben diferir en algún punto: no es un no-op que
    # ignore `ma_func`
    assert not central_sma.equals(central_ema)
    assert not central_sma.equals(central_kama)


def test_bollinger_band_ordering() -> None:
    rng = np.random.default_rng(2)
    price = _series(list(100 + np.cumsum(rng.normal(0, 1, 40))))
    upper, central, lower = bollinger(price, period=10)

    valid = central.notna()
    assert (upper[valid] >= central[valid]).all()
    assert (central[valid] >= lower[valid]).all()


def test_bollinger_output_shape_matches_input() -> None:
    price = _series([float(i) for i in range(15)])
    upper, central, lower = bollinger(price, period=5)

    for s in (upper, central, lower):
        assert len(s) == len(price)
        assert s.index.equals(price.index)


# El no-lookahead de bollinger() lo cubre el mecanismo genérico en
# tests/indicators/test_lookahead.py, vía fxlab.indicators.registry.INDICATORS
# ("bollinger"). Ver la misma nota en test_distance.py.


# --- Keltner -------------------------------------------------------------


def test_keltner_hand_calculated_default_ema() -> None:
    high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0]
    low = [9.0, 10.0, 10.0, 9.0, 11.0, 10.0, 12.0]
    close = [9.5, 10.5, 11.0, 10.0, 12.0, 11.0, 13.0]

    # keltner() llama a ma_func(price) solo con el precio: para que la EMA
    # use el mismo periodo que el ATR, se preconfigura con functools.partial.
    upper, central, lower = keltner(
        _series(close),
        _series(high),
        _series(low),
        ma_func=partial(ema, period=3),
        period=3,
        k=2.0,
    )

    expected_central = _hand_ema(close, 3)
    expected_atr = _hand_atr(high, low, close, 3)
    expected_upper = [
        c + 2.0 * a if not (np.isnan(c) or np.isnan(a)) else np.nan
        for c, a in zip(expected_central, expected_atr, strict=True)
    ]
    expected_lower = [
        c - 2.0 * a if not (np.isnan(c) or np.isnan(a)) else np.nan
        for c, a in zip(expected_central, expected_atr, strict=True)
    ]

    np.testing.assert_allclose(central.to_numpy(), expected_central, rtol=1e-6)
    np.testing.assert_allclose(upper.to_numpy(), expected_upper, rtol=1e-4)
    np.testing.assert_allclose(lower.to_numpy(), expected_lower, rtol=1e-4)


def test_keltner_accepts_any_moving_average_as_centerline() -> None:
    close = _series([1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0, 10.0])
    high = close + 1
    low = close - 1

    _, central_sma, _ = keltner(close, high, low, ma_func=partial(sma, period=3))
    _, central_ema, _ = keltner(close, high, low, ma_func=partial(ema, period=3))

    assert not central_sma.equals(central_ema)


def test_keltner_band_ordering() -> None:
    rng = np.random.default_rng(4)
    close = _series(list(100 + np.cumsum(rng.normal(0, 1, 40))))
    high = close + rng.uniform(0.1, 1.0, 40)
    low = close - rng.uniform(0.1, 1.0, 40)

    upper, central, lower = keltner(close, high, low, period=10)

    valid = central.notna()
    assert (upper[valid] >= central[valid]).all()
    assert (central[valid] >= lower[valid]).all()


def test_keltner_output_shape_matches_input() -> None:
    close = _series([float(i) for i in range(20)])
    high = close + 1
    low = close - 1
    upper, central, lower = keltner(close, high, low, period=5)

    for s in (upper, central, lower):
        assert len(s) == len(close)
        assert s.index.equals(close.index)


# El no-lookahead de keltner() lo cubre el mecanismo genérico en
# tests/indicators/test_lookahead.py, vía fxlab.indicators.registry.INDICATORS
# ("keltner"). Ver la misma nota en test_distance.py.
