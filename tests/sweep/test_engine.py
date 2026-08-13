from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fxlab.split import HOLDOUT_START, Partition
from fxlab.sweep.costs import CostModel
from fxlab.sweep.engine import SweepParams, iter_grid, run_sweep, run_trial


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


class _FakeRegistry:
    """Sustituto de TrialRegistry que solo guarda en memoria lo que recibe."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_trial(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_iter_grid_produces_the_full_cartesian_product() -> None:
    grid = {
        "slow_ma": ["sma", "ema"],
        "slow_period": [20, 50, 100],
        "fast_ma": ["ema"],
        "fast_period": [5, 10],
        "n_bars": [5],
        "use_adx_filter": [False],
        "adx_threshold": [None],
        "adx_period": [14],
    }
    combos = list(iter_grid(grid))

    assert len(combos) == 2 * 3 * 1 * 2 * 1 * 1 * 1 * 1
    assert len({combo for combo in combos}) == len(combos)  # todas distintas


def test_run_trial_with_no_possible_exit_registers_zero_trades() -> None:
    # n_bars mayor que la longitud de los datos: ninguna entrada puede
    # cerrarse dentro de los datos disponibles, así que se descartan todas.
    bid, ask = _synthetic_bid_ask(100)
    params = SweepParams(
        slow_ma="sma",
        slow_period=20,
        fast_ma="ema",
        fast_period=5,
        n_bars=10_000,
        use_adx_filter=False,
        adx_threshold=None,
        adx_period=14,
    )

    result = run_trial(bid, ask, params, CostModel(commission=0.00007), freq="1h")

    assert result.n_trades == 0
    assert result.note == "sin operaciones"
    assert result.total_return is None
    assert result.sharpe is None
    assert result.max_drawdown is None
    assert result.profit_factor is None
    assert result.win_rate is None
    assert result.expectancy is None


def test_run_trial_with_trades_has_no_note() -> None:
    bid, ask = _synthetic_bid_ask(300, seed=1)
    result = run_trial(bid, ask, _BASE_PARAMS, CostModel(commission=0.00007), freq="1h")

    assert result.n_trades > 0
    assert result.note is None
    assert result.total_return is not None


def test_spread_reduces_return_even_with_zero_commission() -> None:
    # con comisión 0, la única diferencia entre las dos ejecuciones es el
    # spread real de los datos: si el spread se aplicara de verdad, el
    # resultado con spread debe ser peor (o igual, nunca mejor) que sin él.
    bid, ask_with_spread = _synthetic_bid_ask(300, seed=3, spread=0.0005)
    ask_no_spread = bid.copy()  # solo para esta comparación, no una ruta real del código

    zero_commission = CostModel(commission=0.0)
    result_with_spread = run_trial(bid, ask_with_spread, _BASE_PARAMS, zero_commission, freq="1h")
    result_no_spread = run_trial(bid, ask_no_spread, _BASE_PARAMS, zero_commission, freq="1h")

    assert result_with_spread.n_trades > 0
    assert result_with_spread.total_return is not None
    assert result_no_spread.total_return is not None
    assert result_with_spread.total_return < result_no_spread.total_return


def test_run_sweep_default_partition_never_sees_data_at_or_after_holdout() -> None:
    # datos que cruzan la partición: desde 2019-11 hasta bien entrado 2020.
    bid, ask = _synthetic_bid_ask(24 * 120, start="2019-11-01", seed=4)
    assert bid.index.max() >= HOLDOUT_START  # el fixture sí contiene holdout

    grid = {k: [v] for k, v in _BASE_PARAMS.as_dict().items()}
    registry = _FakeRegistry()

    run_sweep(
        bid,
        ask,
        grid,
        CostModel(commission=0.00007),
        registry,  # type: ignore[arg-type]
        experiment_id="e2e",
        symbol="EUR/USD",
        interval="1HOUR",
        freq="1h",
    )

    assert len(registry.calls) == 1
    call = registry.calls[0]
    assert call["partition"] == Partition.DEVELOPMENT.value
    assert call["end_date"] < HOLDOUT_START


def test_run_sweep_holdout_requires_explicit_partition_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bid, ask = _synthetic_bid_ask(24 * 60, start="2019-12-15", seed=5)
    grid = {k: [v] for k, v in _BASE_PARAMS.as_dict().items()}
    registry = _FakeRegistry()

    import logging

    with caplog.at_level(logging.WARNING, logger="fxlab.split"):
        run_sweep(
            bid,
            ask,
            grid,
            CostModel(commission=0.00007),
            registry,  # type: ignore[arg-type]
            experiment_id="holdout-run",
            symbol="EUR/USD",
            interval="1HOUR",
            freq="1h",
            partition=Partition.HOLDOUT,
        )

    assert any("holdout" in r.message.lower() for r in caplog.records)
    assert registry.calls[0]["partition"] == Partition.HOLDOUT.value
    assert registry.calls[0]["start_date"] >= HOLDOUT_START
