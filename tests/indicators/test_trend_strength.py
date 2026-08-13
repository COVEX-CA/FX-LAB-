from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.indicators.trend_strength import adx


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_adx_components_hand_calculated_with_period_one() -> None:
    # Con period=1, el suavizado de Wilder es la identidad (alpha=1/1=1):
    # esto hace tratable verificar a mano toda la cadena +DM/-DM/TR/DI/DX.
    high = [10.0, 11.0, 13.0, 12.0, 15.0]
    low = [9.0, 10.0, 11.0, 10.0, 13.0]
    close = [9.5, 10.5, 12.0, 11.0, 14.0]

    adx_result, plus_di, minus_di = adx(_series(high), _series(low), _series(close), period=1)

    # up = high.diff() = [NaN, 1, 2, -1, 3]; down = -low.diff() = [NaN, -1, -1, 1, -2]
    # +DM = up si up>down y up>0, si no 0; -DM = down si down>up y down>0, si no 0
    plus_dm = [np.nan, 1, 2, 0, 3]
    minus_dm = [np.nan, 0, 0, 1, 0]
    tr = [1.0, 1.5, 2.5, 2.0, 4.0]  # TR0 = high0-low0; resto max(...)

    expected_plus_di = [np.nan] + [100 * plus_dm[t] / tr[t] for t in range(1, 5)]
    expected_minus_di = [np.nan] + [100 * minus_dm[t] / tr[t] for t in range(1, 5)]
    # en este ejemplo, en cada barra domina un solo lado -> DX=100 siempre
    expected_dx = [np.nan, 100.0, 100.0, 100.0, 100.0]

    np.testing.assert_allclose(plus_di.to_numpy(), expected_plus_di, rtol=1e-6)
    np.testing.assert_allclose(minus_di.to_numpy(), expected_minus_di, rtol=1e-6)
    # con period=1 el suavizado de DX->ADX también es la identidad
    np.testing.assert_allclose(adx_result.to_numpy(), expected_dx, rtol=1e-6)


def test_adx_output_shape_matches_input() -> None:
    n = 40
    close = _series(list(100 + np.cumsum(np.ones(n))))
    high = close + 0.5
    low = close - 0.5

    adx_result, plus_di, minus_di = adx(high, low, close, period=5)

    for s in (adx_result, plus_di, minus_di):
        assert len(s) == n
        assert s.index.equals(close.index)


def test_adx_nan_prefix_length() -> None:
    # El suavizado de Wilder se aplica dos veces en cadena: ADX tarda
    # aproximadamente 2*period barras en dar su primer valor no-NaN.
    n = 40
    period = 5
    close = _series(list(100 + np.cumsum(np.ones(n))))
    high = close + 0.5
    low = close - 0.5

    adx_result, _, _ = adx(high, low, close, period=period)

    assert adx_result.iloc[: 2 * period - 1].isna().all()
    assert adx_result.iloc[2 * period :].notna().all()


def test_adx_high_on_clear_trend_low_on_sideways() -> None:
    n = 80
    t = np.arange(n)

    trend_close = _series(list(100 + 0.5 * t))
    trend_high = trend_close + 0.3
    trend_low = trend_close - 0.3

    lateral_close = _series(list(100 + 2 * np.sin(2 * np.pi * t / 10)))
    lateral_high = lateral_close + 0.3
    lateral_low = lateral_close - 0.3

    adx_trend, _, _ = adx(trend_high, trend_low, trend_close, period=14)
    adx_lateral, _, _ = adx(lateral_high, lateral_low, lateral_close, period=14)

    assert adx_trend.dropna().mean() > 40
    assert adx_lateral.dropna().mean() < 25
    assert adx_trend.dropna().mean() > adx_lateral.dropna().mean()


def test_adx_measures_strength_not_direction() -> None:
    n = 80
    t = np.arange(n)

    up_close = _series(list(100 + 0.5 * t))
    up_high, up_low = up_close + 0.3, up_close - 0.3

    down_close = _series(list(100 - 0.5 * t))
    down_high, down_low = down_close + 0.3, down_close - 0.3

    adx_up, plus_di_up, minus_di_up = adx(up_high, up_low, up_close, period=14)
    adx_down, plus_di_down, minus_di_down = adx(down_high, down_low, down_close, period=14)

    # +DI/-DI sí distinguen dirección
    assert plus_di_up.dropna().mean() > minus_di_up.dropna().mean()
    assert minus_di_down.dropna().mean() > plus_di_down.dropna().mean()

    # el ADX en sí no: misma fuerza de tendencia en ambos sentidos
    assert adx_up.dropna().iloc[-1] == pytest.approx(adx_down.dropna().iloc[-1], rel=0.05)


def test_adx_no_lookahead() -> None:
    n, cutoff = 80, 60
    t = np.arange(n)
    close = _series(list(100 + np.cumsum(np.sin(t / 3.0))))
    high = close + 0.5
    low = close - 0.5

    adx_result, plus_di, minus_di = adx(high, low, close, period=10)

    modified_close = close.copy()
    modified_close.iloc[cutoff + 1 :] = modified_close.iloc[cutoff + 1 :] * 5 + 1000
    modified_high = modified_close + 0.5
    modified_low = modified_close - 0.5
    m_adx, m_plus_di, m_minus_di = adx(modified_high, modified_low, modified_close, period=10)

    for original, mod in ((adx_result, m_adx), (plus_di, m_plus_di), (minus_di, m_minus_di)):
        prefix_o, prefix_m = original.iloc[: cutoff + 1], mod.iloc[: cutoff + 1]
        assert prefix_o.notna().sum() > 0
        pd.testing.assert_series_equal(prefix_o, prefix_m, check_exact=False)
