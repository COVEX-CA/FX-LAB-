from __future__ import annotations

import inspect

import pandas as pd
import pytest

from fxlab.sweep.costs import CostModel, execution_prices


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_cost_model_commission_has_no_default() -> None:
    # commission es obligatorio: quien construya un CostModel debe decidir
    # el número, nunca heredarlo de un valor por omisión en el código.
    sig = inspect.signature(CostModel)
    assert sig.parameters["commission"].default is inspect.Parameter.empty


def test_cost_model_rejects_negative_commission() -> None:
    with pytest.raises(ValueError, match="commission"):
        CostModel(commission=-0.0001)


def test_cost_model_allows_zero_commission() -> None:
    # cero es una decisión legítima (bróker sin comisión aparte); lo que no
    # puede pasar es que sea el valor por defecto silencioso.
    model = CostModel(commission=0.0)
    assert model.commission == 0.0


def test_execution_prices_buys_at_ask_sells_at_bid() -> None:
    bid = _series([1.1000, 1.1010, 1.1020, 1.1030, 1.1040])
    ask = _series([1.1002, 1.1012, 1.1022, 1.1032, 1.1042])

    long_entries = _series([1, 0, 0, 0, 0]).astype(bool)
    long_exits = _series([0, 0, 1, 0, 0]).astype(bool)
    short_entries = _series([0, 0, 0, 1, 0]).astype(bool)
    short_exits = _series([0, 0, 0, 0, 1]).astype(bool)

    price = execution_prices(long_entries, long_exits, short_entries, short_exits, bid, ask)

    assert price.iloc[0] == ask.iloc[0]  # compra: abrir largo
    assert price.iloc[2] == bid.iloc[2]  # venta: cerrar largo
    assert price.iloc[3] == bid.iloc[3]  # venta: abrir corto
    assert price.iloc[4] == ask.iloc[4]  # compra: cerrar corto


def test_execution_prices_never_equal_when_spread_is_nonzero() -> None:
    # con spread real (bid != ask) y comisión cero, el precio de compra y
    # el de venta en el mismo instante nunca pueden coincidir: el coste de
    # spread no depende de la comisión ni se puede anular junto con ella.
    bid = _series([1.1000] * 4)
    ask = _series([1.1002] * 4)
    long_entries = _series([1, 0, 0, 0]).astype(bool)
    long_exits = _series([0, 1, 0, 0]).astype(bool)
    short_entries = _series([0, 0, 0, 0]).astype(bool)
    short_exits = _series([0, 0, 0, 0]).astype(bool)

    price = execution_prices(long_entries, long_exits, short_entries, short_exits, bid, ask)

    buy_price = price.iloc[0]
    sell_price = price.iloc[1]
    assert buy_price > sell_price


def test_run_trial_signature_has_no_cost_bypass() -> None:
    # ni un flag booleano de "sin costes" ni una forma de pasar comisión
    # fuera del CostModel obligatorio.
    from fxlab.sweep.engine import run_trial

    sig = inspect.signature(run_trial)
    assert "commission" not in sig.parameters
    assert "cost_model" in sig.parameters
    assert sig.parameters["cost_model"].default is inspect.Parameter.empty
    suspicious = [
        name
        for name in sig.parameters
        if ("cost" in name.lower() or "fee" in name.lower()) and name != "cost_model"
    ]
    assert suspicious == []
