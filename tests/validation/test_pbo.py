"""Propiedades estructurales del PBO/CSCV que no dependen de qué tan
"buenos" o "ruidosos" sean los retornos: número de combinaciones,
validaciones de entrada y forma del resultado."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fxlab.validation.pbo import probability_of_backtest_overfitting


def _returns(t: int, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(t, n))
    return pd.DataFrame(data, columns=[f"c{i}" for i in range(n)])


def test_s_equals_16_evaluates_exactly_c_16_8_combinations() -> None:
    returns = _returns(t=1600, n=8)
    result = probability_of_backtest_overfitting(returns, s=16)
    assert result.n_combinations == math.comb(16, 8) == 12870
    assert len(result.logits) == 12870
    assert len(result.is_oos_degradation_non_annualized) == 12870


def test_rejects_odd_s() -> None:
    returns = _returns(t=1600, n=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(returns, s=15)


def test_rejects_s_below_two() -> None:
    returns = _returns(t=1600, n=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(returns, s=0)


def test_rejects_t_not_multiple_of_s() -> None:
    returns = _returns(t=1601, n=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(returns, s=16)


def test_rejects_fewer_than_two_configurations() -> None:
    returns = _returns(t=1600, n=1)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(returns, s=16)


def test_pbo_is_between_zero_and_one() -> None:
    returns = _returns(t=1600, n=10)
    result = probability_of_backtest_overfitting(returns, s=16)
    assert 0.0 <= result.pbo <= 1.0
