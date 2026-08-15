from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxlab.sweep.costs import CostModel
from fxlab.sweep.engine import (
    MaCrossParams,
    iter_ma_cross_grid,
    run_ma_cross_sweep,
    run_ma_cross_trial,
)
from fxlab.sweep.registry import TrialRegistry


def _synthetic_bid_ask(
    n: int, start: str = "2004-01-01", seed: int = 0, spread: float = 0.0001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = close - rng.normal(0, 0.0001, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0002, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0002, n))
    bid = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)
    ask = bid + spread
    return bid, ask


# Cruce con tipos DISTINTOS a propósito (EMA rápida contra SMA lenta): la ruta
# tiene que admitirlo.
_BASE = MaCrossParams(fast_ma="ema", fast_period=5, slow_ma="sma", slow_period=20)


def test_iter_ma_cross_grid_is_the_full_cartesian_product() -> None:
    grid = {
        "fast_ma": ["sma", "ema"],
        "fast_period": [5, 10],
        "slow_ma": ["sma"],
        "slow_period": [50, 100],
    }
    combos = list(iter_ma_cross_grid(grid))
    assert len(combos) == 2 * 2 * 1 * 2
    assert len({c for c in combos}) == len(combos)


def test_iter_ma_cross_grid_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="faltan campos"):
        list(iter_ma_cross_grid({"fast_ma": ["ema"], "fast_period": [5]}))


def test_run_ma_cross_trial_returns_series_aligned_with_trades() -> None:
    bid, ask = _synthetic_bid_ask(300, seed=1)
    result = run_ma_cross_trial(bid, ask, _BASE, CostModel(commission=0.00007), freq="1h")

    assert isinstance(result.returns, pd.Series)
    assert result.returns.index.equals(bid.index)
    # un cruce rápido sobre 300 barras produce operaciones
    assert result.n_trades > 0
    assert result.total_return is not None


def test_run_ma_cross_sweep_persists_a_returns_matrix(tmp_path: Path) -> None:
    bid, ask = _synthetic_bid_ask(300, seed=7)  # empieza en 2004: todo desarrollo
    grid = {
        "fast_ma": ["sma", "ema"],
        "fast_period": [5, 10],
        "slow_ma": ["sma"],
        "slow_period": [20],
    }

    with TrialRegistry(tmp_path / "trials.db") as registry:
        total = run_ma_cross_sweep(
            bid,
            ask,
            grid,
            CostModel(commission=0.00007),
            registry,
            experiment_id="cross",
            symbol="EUR/USD",
            interval="1HOUR",
            freq="1h",
        )
        trials = registry.load_experiment("cross")
        matrix = registry.load_returns_matrix("cross")

    assert total == 4
    assert len(trials) == 4
    # biyección columnas == ids, mismo contrato que consume el informe
    assert list(matrix.columns) == [str(i) for i in trials["id"]]
    assert len(matrix) == len(bid)
    # los parámetros de cruce quedan expandidos en el registro
    assert "param_fast_ma" in trials.columns
    assert "param_slow_period" in trials.columns
