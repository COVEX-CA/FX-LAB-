"""Walk-forward: los pliegues nunca solapan entrenamiento con prueba, y
ninguno toca el holdout — se comprueba tanto en el generador de pliegues
(`_generate_folds`, directamente, para verificar los límites exactos) como
de punta a punta en `run_walk_forward` (con datos sintéticos que cruzan la
frontera de holdout, para verificar que el recorte a desarrollo realmente
se aplica antes de generar ningún pliegue)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.split import HOLDOUT_START
from fxlab.sweep.costs import CostModel
from fxlab.sweep.engine import SweepParams
from fxlab.validation.walk_forward import WindowMode, _generate_folds, run_walk_forward

_BASE_PARAMS = SweepParams(
    slow_ma="sma",
    slow_period=20,
    fast_ma="ema",
    fast_period=5,
    n_bars=5,
    use_adx_filter=False,
    adx_threshold=None,
    adx_period=14,
)
_GRID = {k: [v] for k, v in _BASE_PARAMS.as_dict().items()}


def _synthetic_bid_ask(
    n: int, start: str, seed: int = 0, spread: float = 0.0001
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


@pytest.mark.parametrize("mode", [WindowMode.ANCHORED, WindowMode.ROLLING])
def test_generate_folds_never_overlap_train_and_test(mode: WindowMode) -> None:
    index = pd.date_range("2010-01-01", periods=1000, freq="1h", tz="UTC")
    folds = list(_generate_folds(index, train_size=200, test_size=50, step=50, mode=mode))

    assert len(folds) > 0
    for fold in folds:
        assert fold.train_start <= fold.train_end < fold.test_start <= fold.test_end
        # ningún índice de prueba puede coincidir con uno de entrenamiento
        train_range = index[(index >= fold.train_start) & (index <= fold.train_end)]
        test_range = index[(index >= fold.test_start) & (index <= fold.test_end)]
        assert train_range.intersection(test_range).empty


def test_generate_folds_anchored_train_start_is_always_the_first_bar() -> None:
    index = pd.date_range("2010-01-01", periods=1000, freq="1h", tz="UTC")
    folds = list(
        _generate_folds(index, train_size=200, test_size=50, step=50, mode=WindowMode.ANCHORED)
    )
    assert all(fold.train_start == index[0] for fold in folds)
    # la ventana de entrenamiento crece de un pliegue al siguiente
    train_lengths = [len(index[(index >= f.train_start) & (index <= f.train_end)]) for f in folds]
    assert train_lengths == sorted(train_lengths)
    assert len(set(train_lengths)) > 1


def test_generate_folds_rolling_train_window_has_fixed_size() -> None:
    index = pd.date_range("2010-01-01", periods=1000, freq="1h", tz="UTC")
    train_size = 200
    folds = list(
        _generate_folds(
            index, train_size=train_size, test_size=50, step=50, mode=WindowMode.ROLLING
        )
    )
    assert len(folds) > 0
    for fold in folds:
        train_range = index[(index >= fold.train_start) & (index <= fold.train_end)]
        assert len(train_range) == train_size


def test_generate_folds_stops_before_running_out_of_test_bars() -> None:
    index = pd.date_range("2010-01-01", periods=310, freq="1h", tz="UTC")
    folds = list(
        _generate_folds(index, train_size=200, test_size=50, step=50, mode=WindowMode.ANCHORED)
    )
    for fold in folds:
        assert fold.test_end <= index[-1]


@pytest.mark.parametrize("mode", [WindowMode.ANCHORED, WindowMode.ROLLING])
def test_run_walk_forward_never_touches_holdout(mode: WindowMode) -> None:
    # arranca en desarrollo y cruza deliberadamente la frontera de holdout
    # (2020-01-01): si el recorte a Partition.DEVELOPMENT fallara, algunos
    # pliegues incluirían barras >= HOLDOUT_START.
    start = HOLDOUT_START - pd.Timedelta(hours=900)
    bid, ask = _synthetic_bid_ask(n=1500, start=str(start.date()))

    result = run_walk_forward(
        bid,
        ask,
        _GRID,
        CostModel(commission=0.00007),
        train_size=200,
        test_size=50,
        step=50,
        mode=mode,
        freq="1h",
    )

    assert len(result.folds) > 0
    for fold_result in result.folds:
        fold = fold_result.fold
        assert fold.train_start < HOLDOUT_START
        assert fold.train_end < HOLDOUT_START
        assert fold.test_start < HOLDOUT_START
        assert fold.test_end < HOLDOUT_START


def test_run_walk_forward_rejects_non_positive_window_sizes() -> None:
    bid, ask = _synthetic_bid_ask(n=500, start="2010-01-01")
    cost_model = CostModel(commission=0.00007)
    with pytest.raises(ValueError):
        run_walk_forward(
            bid,
            ask,
            _GRID,
            cost_model,
            train_size=0,
            test_size=50,
            step=50,
            mode=WindowMode.ANCHORED,
            freq="1h",
        )


def test_run_walk_forward_rejects_unknown_selection_metric() -> None:
    bid, ask = _synthetic_bid_ask(n=500, start="2010-01-01")
    cost_model = CostModel(commission=0.00007)
    with pytest.raises(ValueError):
        run_walk_forward(
            bid,
            ask,
            _GRID,
            cost_model,
            train_size=200,
            test_size=50,
            step=50,
            mode=WindowMode.ANCHORED,
            freq="1h",
            selection_metric="not_a_real_metric",
        )
