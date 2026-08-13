"""Test de no-lookahead genérico para el registro de medias móviles.

Recorre automáticamente `fxlab.indicators.moving_averages.MOVING_AVERAGES`:
cualquier media que se añada al registro en el futuro queda cubierta sin
tocar este test. Es, con diferencia, el test más importante de esta fase —
una media que "mire al futuro" produciría resultados que parecen buenos en
un backtest y son falsos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.indicators.moving_averages import MOVING_AVERAGES

_N = 200
_CUTOFF = 150  # posición t: todo lo posterior a esta barra se va a alterar


def _make_price_series() -> pd.Series:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=_N, freq="1D", tz="UTC")
    values = 100.0 + np.cumsum(rng.normal(0, 1, size=_N))
    return pd.Series(values, index=index, dtype="float64")


@pytest.mark.parametrize("name", sorted(MOVING_AVERAGES))
def test_moving_average_value_at_t_ignores_data_after_t(name: str) -> None:
    func = MOVING_AVERAGES[name]
    price = _make_price_series()

    original = func(price)

    # altera radicalmente todo lo posterior a _CUTOFF, dejando [0, _CUTOFF] igual
    modified_price = price.copy()
    tail = modified_price.iloc[_CUTOFF + 1 :]
    modified_price.iloc[_CUTOFF + 1 :] = tail * 1000.0 + 50_000.0
    modified = func(modified_price)

    prefix_original = original.iloc[: _CUTOFF + 1]
    prefix_modified = modified.iloc[: _CUTOFF + 1]

    # si esto fuera 0, el test no comprobaría nada real para `name`
    assert prefix_original.notna().sum() > 0, (
        f"{name}: ningún valor no-NaN en el prefijo con los parámetros de prueba, "
        "el test no está ejercitando nada"
    )

    pd.testing.assert_series_equal(prefix_original, prefix_modified, check_exact=False)


@pytest.mark.parametrize("name", sorted(MOVING_AVERAGES))
def test_moving_average_output_shape_matches_input(name: str) -> None:
    func = MOVING_AVERAGES[name]
    price = _make_price_series()

    result = func(price)

    assert len(result) == len(price)
    assert result.index.equals(price.index)
