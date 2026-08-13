from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.indicators.distance import distance


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_distance_std_method_hand_calculated() -> None:
    price = _series([100, 102, 101, 105, 103, 108, 107, 112, 110, 115])
    ma = _series([100.0] * 10)  # referencia constante: simplifica el numerador

    result = distance(price, ma, method="std", period=3)

    values = price.to_numpy()
    log_returns = np.log(values[1:] / values[:-1])  # r_1..r_9, calculado a mano (no vía distance())

    # std_movil(period=3) con ddof=1 (por defecto en pandas): primer valor
    # válido en el índice de retorno 2 (usa r0,r1,r2), que corresponde al
    # precio en el índice 3.
    sigma = [np.std(log_returns[i - 2 : i + 1], ddof=1) for i in range(2, len(log_returns))]
    pct_distance = (values[3:] - 100.0) / 100.0
    expected_tail = pct_distance / np.array(sigma)

    assert result.iloc[:3].isna().all()
    np.testing.assert_allclose(result.iloc[3:].to_numpy(), expected_tail, rtol=1e-8)


def test_distance_atr_method_hand_calculated() -> None:
    high = _series([10, 11, 12, 11, 13, 12, 14])
    low = _series([9, 10, 10, 9, 11, 10, 12])
    close = _series([9.5, 10.5, 11, 10, 12, 11, 13])
    ma = _series([10.0] * 7)

    result = distance(close, ma, method="atr", period=3, high=high, low=low)

    # True Range calculado a mano
    h, low_, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    tr = [h[0] - low_[0]]
    for t in range(1, len(h)):
        tr.append(max(h[t] - low_[t], abs(h[t] - c[t - 1]), abs(low_[t] - c[t - 1])))
    assert tr == [1, 1.5, 2, 2, 3, 2, 3]

    # Suavizado de Wilder (alpha=1/3), sembrado con la media de los 3 primeros TR
    atr_vals = [np.nan, np.nan, sum(tr[:3]) / 3]
    for t in range(3, len(tr)):
        atr_vals.append(atr_vals[-1] + (tr[t] - atr_vals[-1]) / 3)

    expected = [
        (c[t] - 10.0) / atr_vals[t] if not np.isnan(atr_vals[t]) else np.nan for t in range(len(c))
    ]

    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


def test_distance_sign_positive_above_negative_below() -> None:
    price = _series([100, 110, 90])
    ma = _series([100.0, 100.0, 100.0])
    high = _series([101, 111, 91])
    low = _series([99, 109, 89])

    result = distance(price, ma, method="atr", period=1, high=high, low=low)

    assert result.iloc[1] > 0  # precio por encima de la media
    assert result.iloc[2] < 0  # precio por debajo de la media


def test_distance_atr_without_high_low_raises() -> None:
    price = _series([1.0, 2.0, 3.0])
    ma = _series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="high"):
        distance(price, ma, method="atr")


def test_distance_unknown_method_raises() -> None:
    price = _series([1.0, 2.0, 3.0])
    ma = _series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="method"):
        distance(price, ma, method="bogus")


def test_distance_output_shape_matches_input() -> None:
    price = _series([float(i) for i in range(20)])
    ma = _series([float(i) for i in range(20)])
    high = price + 1
    low = price - 1

    result_std = distance(price, ma, method="std")
    result_atr = distance(price, ma, method="atr", high=high, low=low)

    assert len(result_std) == len(price)
    assert result_std.index.equals(price.index)
    assert len(result_atr) == len(price)
    assert result_atr.index.equals(price.index)


def test_distance_no_lookahead() -> None:
    n = 60
    cutoff = 40
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    price = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=index)
    ma = price.rolling(5, min_periods=5).mean()
    high = price + 1
    low = price - 1

    for method, kwargs in (("std", {}), ("atr", {"high": high, "low": low})):
        original = distance(price, ma, method=method, **kwargs)

        modified_price = price.copy()
        modified_price.iloc[cutoff + 1 :] = modified_price.iloc[cutoff + 1 :] * 5 + 1000
        modified_ma = modified_price.rolling(5, min_periods=5).mean()
        modified_kwargs = dict(kwargs)
        if method == "atr":
            modified_kwargs = {"high": modified_price + 1, "low": modified_price - 1}
        modified = distance(modified_price, modified_ma, method=method, **modified_kwargs)

        prefix_original = original.iloc[: cutoff + 1]
        prefix_modified = modified.iloc[: cutoff + 1]
        assert prefix_original.notna().sum() > 0
        pd.testing.assert_series_equal(prefix_original, prefix_modified, check_exact=False)
