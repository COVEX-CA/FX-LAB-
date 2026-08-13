"""Valores calculados a mano para las doce medias móviles.

Cada test construye una serie corta y compara contra valores derivados de la
fórmula documentada en `fxlab.indicators.moving_averages`, escritos aquí
como aritmética explícita — nunca contra la salida de la propia función ni
contra una librería externa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.indicators.moving_averages import (
    alma,
    dema,
    ema,
    hma,
    kama,
    mcginley,
    sma,
    t3,
    tema,
    vidya,
    wma,
    zlema,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


# --- SMA ---------------------------------------------------------------


def test_sma_hand_calculated() -> None:
    price = _series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = sma(price, period=3)

    # SMA(3): media de las 3 últimas barras
    expected = [
        np.nan,
        np.nan,
        (1 + 2 + 3) / 3,
        (2 + 3 + 4) / 3,
        (3 + 4 + 5) / 3,
        (4 + 5 + 6) / 3,
        (5 + 6 + 7) / 3,
        (6 + 7 + 8) / 3,
        (7 + 8 + 9) / 3,
        (8 + 9 + 10) / 3,
    ]
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_sma_no_nan_after_warmup() -> None:
    price = _series(list(range(1, 16)))
    result = sma(price, period=4)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].notna().all()


# --- EMA -----------------------------------------------------------------


def test_ema_hand_calculated() -> None:
    # alpha = 2/(3+1) = 0.5; siembra = media(1,2,3) = 2.0 en t=2
    price = _series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = ema(price, period=3)

    e2 = 2.0
    e3 = 0.5 * 4 + 0.5 * e2
    e4 = 0.5 * 5 + 0.5 * e3
    e5 = 0.5 * 6 + 0.5 * e4
    e6 = 0.5 * 7 + 0.5 * e5
    e7 = 0.5 * 8 + 0.5 * e6
    e8 = 0.5 * 9 + 0.5 * e7
    e9 = 0.5 * 10 + 0.5 * e8
    expected = [np.nan, np.nan, e2, e3, e4, e5, e6, e7, e8, e9]

    np.testing.assert_allclose(result.to_numpy(), expected)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[9] == pytest.approx(9.0)


# --- WMA -------------------------------------------------------------------


def test_wma_hand_calculated() -> None:
    price = _series([1, 2, 3, 4, 5])
    result = wma(price, period=3)

    # pesos [1,2,3], suma=6
    w2 = (1 * 1 + 2 * 2 + 3 * 3) / 6
    w3 = (1 * 2 + 2 * 3 + 3 * 4) / 6
    w4 = (1 * 3 + 2 * 4 + 3 * 5) / 6
    expected = [np.nan, np.nan, w2, w3, w4]

    np.testing.assert_allclose(result.to_numpy(), expected)
    assert result.iloc[2] == pytest.approx(14 / 6)
    assert result.iloc[3] == pytest.approx(20 / 6)
    assert result.iloc[4] == pytest.approx(26 / 6)


# --- HMA ---------------------------------------------------------------


def test_hma_hand_calculated_on_linear_ramp() -> None:
    # Sobre una rampa lineal perfecta, HMA converge exactamente al precio
    # (por diseño: es justo lo que busca reducir el retardo de Hull).
    price = _series([1, 2, 3, 4, 5, 6, 7, 8])
    result = hma(price, period=4)  # half=2, sqrt(4)=2: números redondos

    expected = [np.nan, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0]
    np.testing.assert_allclose(result.to_numpy(), expected)


# --- DEMA / TEMA -----------------------------------------------------------


def _hand_ema(values: list[float], period: int) -> list[float]:
    """Réplica manual y directa de la fórmula de EMA (siembra SMA), para
    construir los valores esperados de DEMA/TEMA sin llamar a `ema()`."""
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


def test_dema_hand_calculated() -> None:
    values = [1, 2, 4, 7, 11, 16]
    price = _series(values)
    result = dema(price, period=2)

    ema1 = _hand_ema(values, 2)
    ema2 = _hand_ema(ema1, 2)  # ema1[0] es NaN: _hand_ema ignora prefijo si len < period
    # _hand_ema no sabe saltarse NaN iniciales; se reconstruye a mano aquí:
    valid_ema1 = ema1[1:]  # ema1[0] es NaN
    ema2_valid = _hand_ema(valid_ema1, 2)
    ema2 = [np.nan] + ema2_valid

    expected = [
        2 * e1 - e2 if not (np.isnan(e1) or np.isnan(e2)) else np.nan
        for e1, e2 in zip(ema1, ema2, strict=True)
    ]
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


def test_tema_hand_calculated() -> None:
    values = [1, 2, 4, 7, 11, 16]
    price = _series(values)
    result = tema(price, period=2)

    ema1 = _hand_ema(values, 2)
    ema2 = [np.nan] + _hand_ema(ema1[1:], 2)
    ema3 = [np.nan, np.nan] + _hand_ema(ema2[2:], 2)

    expected = []
    for e1, e2, e3 in zip(ema1, ema2, ema3, strict=True):
        if np.isnan(e1) or np.isnan(e2) or np.isnan(e3):
            expected.append(np.nan)
        else:
            expected.append(3 * e1 - 3 * e2 + e3)

    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


# --- ZLEMA -----------------------------------------------------------------


def test_zlema_hand_calculated() -> None:
    price = _series([1, 2, 4, 7, 11])
    result = zlema(price, period=3)  # lag = round((3-1)/2) = 1

    # des-retardada_t = 2*price_t - price_{t-1}, definida desde t=1
    de_lagged = [np.nan, 2 * 2 - 1, 2 * 4 - 2, 2 * 7 - 4, 2 * 11 - 7]
    assert de_lagged == [np.nan, 3, 6, 10, 15]

    # ema(de_lagged, 3): siembra = media(3,6,10) = 19/3 en índice 3
    seed = (3 + 6 + 10) / 3
    alpha = 2 / 4
    e4 = alpha * 15 + (1 - alpha) * seed

    expected = [np.nan, np.nan, np.nan, seed, e4]
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


# --- ALMA --------------------------------------------------------------


def test_alma_hand_calculated() -> None:
    price = _series([1, 2, 3, 4, 5])
    result = alma(price, period=3, offset=0.85, sigma=6.0)

    m = 0.85 * 2  # 1.7
    s = 3 / 6.0  # 0.5
    w = [np.exp(-((i - m) ** 2) / (2 * s**2)) for i in range(3)]
    w_sum = sum(w)
    w_norm = [wi / w_sum for wi in w]

    a2 = w_norm[0] * 1 + w_norm[1] * 2 + w_norm[2] * 3
    a3 = w_norm[0] * 2 + w_norm[1] * 3 + w_norm[2] * 4
    a4 = w_norm[0] * 3 + w_norm[1] * 4 + w_norm[2] * 5

    expected = [np.nan, np.nan, a2, a3, a4]
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


# --- KAMA --------------------------------------------------------------


def test_kama_hand_calculated() -> None:
    price = _series([1, 2, 1, 3, 5, 4, 6])
    result = kama(price, er_period=2, fast_period=2, slow_period=5)

    fast_sc = 2 / 3
    slow_sc = 1 / 3

    # siembra en t=er_period=2
    k2 = 1.0  # price[2]

    # t=3: ER = |price[3]-price[1]| / (|price[3]-price[2]|+|price[2]-price[1]|)
    er3 = abs(3 - 2) / (abs(3 - 1) + abs(1 - 2))
    sc3 = (er3 * (fast_sc - slow_sc) + slow_sc) ** 2
    k3 = k2 + sc3 * (3 - k2)

    # t=4
    er4 = abs(5 - 1) / (abs(5 - 3) + abs(3 - 1))
    sc4 = (er4 * (fast_sc - slow_sc) + slow_sc) ** 2
    k4 = k3 + sc4 * (5 - k3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(k2)
    assert result.iloc[3] == pytest.approx(k3, abs=1e-6)
    assert result.iloc[4] == pytest.approx(k4, abs=1e-6)


# --- VIDYA -------------------------------------------------------------


def test_vidya_hand_calculated() -> None:
    price = _series([1, 2, 4, 3, 6, 5])
    result = vidya(price, period=3)

    assert result.iloc[:3].isna().all()
    # siembra en t=period=3
    assert result.iloc[3] == pytest.approx(3.0)
    # t=4: CMO sobre diffs[1:4]=[2,-1,3] -> up=5, down=1, CMO=100*4/6=66.667
    assert result.iloc[4] == pytest.approx(4.0, abs=1e-6)
    # t=5: CMO sobre diffs[2:5]=[-1,3,-1] -> up=3, down=2, CMO=100*1/5=20
    assert result.iloc[5] == pytest.approx(4.1, abs=1e-6)


# --- McGinley Dynamic ----------------------------------------------------


def test_mcginley_hand_calculated() -> None:
    price = _series([10, 11, 12, 11, 13])
    result = mcginley(price, period=3)

    # siembra: media(10,11,12) = 11 en t=2
    md2 = 11.0
    # t=3: price==md2, numerador 0 -> no cambia
    md3 = md2 + (11 - md2) / (3 * (11 / md2) ** 4)
    # t=4
    md4 = md3 + (13 - md3) / (3 * (13 / md3) ** 4)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(11.0)
    assert result.iloc[3] == pytest.approx(11.0)
    assert result.iloc[4] == pytest.approx(md4, abs=1e-4)


def test_mcginley_constant_series_stays_constant() -> None:
    # Si price == MD_prev, el numerador es 0 y MD no cambia: invariante
    # verificable a mano directamente desde la fórmula.
    price = _series([5.0] * 10)
    result = mcginley(price, period=3)

    assert result.iloc[:2].isna().all()
    assert (result.iloc[2:] == 5.0).all()


# --- T3 ------------------------------------------------------------------


def test_t3_coefficients_sum_to_one() -> None:
    # c1+c2+c3+c4 = 1 para cualquier v (los términos en v se cancelan):
    # identidad usada como comprobación de coherencia.
    v = 0.7
    c1 = -(v**3)
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v**2
    assert c1 + c2 + c3 + c4 == pytest.approx(1.0)


def test_t3_hand_calculated_via_chained_emas() -> None:
    # Sobre una serie constante, T3 reproduce la constante sea cual sea el
    # emparejamiento coeficiente<->nivel de EMA (con e3=e4=e5=e6, un error
    # de asignación no se detectaría). Aquí se encadenan las seis EMA a
    # mano con `_hand_ema` sobre una serie NO constante, para que un error
    # en qué coeficiente multiplica a qué nivel, o en el orden de
    # anidamiento, sí cambie el resultado y el test lo detecte.
    values = [1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0, 10.0, 12.0, 11.0, 14.0, 13.0, 16.0]
    period = 2
    v = 0.7
    price = _series(values)
    result = t3(price, period=period, v=v)

    e1 = _hand_ema(values, period)
    e2 = [np.nan] + _hand_ema(e1[1:], period)
    e3 = [np.nan, np.nan] + _hand_ema(e2[2:], period)
    e4 = [np.nan] * 3 + _hand_ema(e3[3:], period)
    e5 = [np.nan] * 4 + _hand_ema(e4[4:], period)
    e6 = [np.nan] * 5 + _hand_ema(e5[5:], period)

    c1 = -(v**3)
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v**2

    expected = []
    for e3_t, e4_t, e5_t, e6_t in zip(e3, e4, e5, e6, strict=True):
        if np.isnan(e3_t) or np.isnan(e4_t) or np.isnan(e5_t) or np.isnan(e6_t):
            expected.append(np.nan)
        else:
            expected.append(c1 * e6_t + c2 * e5_t + c3 * e4_t + c4 * e3_t)

    assert result.iloc[:6].isna().all()
    assert result.iloc[6:].notna().all()
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)
