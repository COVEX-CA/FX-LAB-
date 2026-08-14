"""Test de no-lookahead genérico, sobre el registro `INDICATORS` completo.

Recorre automáticamente `fxlab.indicators.registry.INDICATORS`, que cubre
las doce medias (`moving_averages`) y también `distance`, `bands` y
`trend_strength`: cualquier indicador que se añada en el futuro, en
cualquiera de los cuatro módulos, queda cubierto en cuanto se registre en
`INDICATORS` — y si no se registra, `test_registry_covers_every_public_indicator_function`
falla y lo dice por su nombre.

Es, con diferencia, el test más importante de esta fase — un indicador que
"mire al futuro" produciría resultados que parecen buenos en un backtest y
son falsos.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import fxlab.indicators as indicators_package
from fxlab.indicators.registry import COVERED_FUNCTIONS, INDICATORS

_N = 200
_CUTOFF = 150  # posición t: todo lo posterior a esta barra se va a alterar


def _make_ohlc_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=_N, freq="1D", tz="UTC")
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=_N)), index=index, dtype="float64")
    return pd.DataFrame({"high": close + 1.0, "low": close - 1.0, "close": close})


def _as_frame(result: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_value_at_t_ignores_data_after_t(name: str) -> None:
    func = INDICATORS[name]
    ohlc = _make_ohlc_frame()

    original = _as_frame(func(ohlc))

    # altera radicalmente todo lo posterior a _CUTOFF, dejando [0, _CUTOFF] igual
    modified_close = ohlc["close"].copy()
    tail = modified_close.iloc[_CUTOFF + 1 :]
    modified_close.iloc[_CUTOFF + 1 :] = tail * 1000.0 + 50_000.0
    modified_ohlc = pd.DataFrame(
        {"high": modified_close + 1.0, "low": modified_close - 1.0, "close": modified_close}
    )
    modified = _as_frame(func(modified_ohlc))

    prefix_original = original.iloc[: _CUTOFF + 1]
    prefix_modified = modified.iloc[: _CUTOFF + 1]

    # si esto fuera 0, el test no comprobaría nada real para `name`
    assert prefix_original.notna().to_numpy().sum() > 0, (
        f"{name}: ningún valor no-NaN en el prefijo con los parámetros de prueba, "
        "el test no está ejercitando nada"
    )

    pd.testing.assert_frame_equal(prefix_original, prefix_modified, check_exact=False)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_output_shape_matches_input(name: str) -> None:
    func = INDICATORS[name]
    ohlc = _make_ohlc_frame()

    result = _as_frame(func(ohlc))

    assert len(result) == len(ohlc)
    assert result.index.equals(ohlc.index)


def _public_indicator_functions() -> dict[str, Callable[..., object]]:
    """Todas las funciones públicas definidas en los módulos de `fxlab.indicators`.

    Excluye módulos privados (nombre con `_` inicial, como `_true_range`) y
    el propio `registry`. Excluye también los nombres importados de otro
    módulo (p.ej. `sma`/`ema` re-expuestos en `bands.py` como valor por
    defecto de `ma_func`): solo cuentan las funciones definidas ahí.
    """
    functions: dict[str, Callable[..., object]] = {}
    assert indicators_package.__path__ is not None
    for module_info in pkgutil.iter_modules(
        indicators_package.__path__, prefix=f"{indicators_package.__name__}."
    ):
        leaf_name = module_info.name.rsplit(".", 1)[-1]
        if leaf_name.startswith("_") or leaf_name == "registry":
            continue
        module = importlib.import_module(module_info.name)
        for func_name, obj in inspect.getmembers(module, inspect.isfunction):
            if func_name.startswith("_") or obj.__module__ != module.__name__:
                continue
            functions[f"{module_info.name}.{func_name}"] = obj
    return functions


def test_registry_covers_every_public_indicator_function() -> None:
    """Ninguna función pública de indicador puede quedarse sin registrar.

    Sin este test, `INDICATORS` solo protege a quien se acuerda de añadir
    su indicador nuevo al registro — que es precisamente el fallo que este
    mecanismo existe para evitar.
    """
    all_functions = _public_indicator_functions()
    missing = sorted(
        qualname for qualname, func in all_functions.items() if func not in COVERED_FUNCTIONS
    )
    assert not missing, f"funciones públicas de indicador sin registrar en INDICATORS: {missing}"
