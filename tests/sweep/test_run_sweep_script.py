"""Resolución y validación del rango de fechas del CLI de barrido.

Se prueba `resolve_range` de forma aislada (sin red, sin cargar datos): la
regla de oro que fija este módulo es que un barrido **nunca** puede cruzar
el corte de holdout, y que pedirlo falla con un error explícito en vez de
recortar en silencio.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from fxlab.split import DEVELOPMENT_START, HOLDOUT_START, Partition
from fxlab.sweep.engine import run_ma_cross_sweep, run_sweep

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_sweep.py"
_NOW = pd.Timestamp("2024-01-01", tz="UTC")


def _load_script() -> ModuleType:
    """`scripts/` no es un paquete importable: se carga por ruta."""
    spec = importlib.util.spec_from_file_location("run_sweep", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sweep"] = module
    spec.loader.exec_module(module)
    return module


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_resolve_range_defaults_to_the_whole_development_partition() -> None:
    script = _load_script()
    start, end = script.resolve_range(Partition.DEVELOPMENT, None, None, now=_NOW)
    assert start == DEVELOPMENT_START
    assert end == HOLDOUT_START


def test_resolve_range_restricts_to_a_requested_development_subperiod() -> None:
    script = _load_script()
    requested_start, requested_end = _ts("2015-01-01"), _ts("2017-01-01")
    start, end = script.resolve_range(
        Partition.DEVELOPMENT, requested_start, requested_end, now=_NOW
    )
    assert (start, end) == (requested_start, requested_end)


def test_resolve_range_allows_end_exactly_at_the_holdout_cut() -> None:
    # --end es exclusivo, así que HOLDOUT_START como fin cubre todo desarrollo
    # sin tocar ni una barra reservada: es el límite permitido, no un cruce.
    script = _load_script()
    start, end = script.resolve_range(Partition.DEVELOPMENT, _ts("2019-01-01"), HOLDOUT_START)
    assert end == HOLDOUT_START


def test_resolve_range_rejects_a_development_end_that_crosses_the_holdout() -> None:
    # La prueba obligatoria: un --end que alcanza la partición reservada debe
    # fallar explícitamente, nunca recortarse en silencio.
    script = _load_script()
    with pytest.raises(ValueError, match="holdout"):
        script.resolve_range(Partition.DEVELOPMENT, _ts("2019-01-01"), _ts("2020-06-01"))


def test_resolve_range_rejects_a_development_range_fully_inside_the_holdout() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="holdout"):
        script.resolve_range(Partition.DEVELOPMENT, _ts("2020-06-01"), _ts("2020-12-01"))


def test_resolve_range_rejects_an_empty_range() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="vac"):
        script.resolve_range(Partition.DEVELOPMENT, _ts("2017-01-01"), _ts("2015-01-01"))


def test_resolve_range_holdout_defaults_run_from_the_cut_to_now() -> None:
    script = _load_script()
    start, end = script.resolve_range(Partition.HOLDOUT, None, None, now=_NOW)
    assert start == HOLDOUT_START
    assert end == _NOW


def test_resolve_range_holdout_cannot_reach_back_into_development() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="holdout"):
        script.resolve_range(Partition.HOLDOUT, _ts("2019-06-01"), None, now=_NOW)


def test_select_sweep_dispatches_by_strategy() -> None:
    script = _load_script()
    assert script.select_sweep("pullback") is run_sweep
    assert script.select_sweep("ma_cross") is run_ma_cross_sweep


def test_select_sweep_rejects_an_unknown_strategy() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="desconocida"):
        script.select_sweep("bollinger")


def test_parser_accepts_start_and_end_and_defaults_them_to_none() -> None:
    script = _load_script()
    parser = script._build_parser()

    without_range = parser.parse_args(["cfg.yaml"])
    assert without_range.start is None
    assert without_range.end is None

    with_range = parser.parse_args(["cfg.yaml", "--start", "2015-01-01", "--end", "2017-01-01"])
    assert with_range.start == _ts("2015-01-01")
    assert with_range.end == _ts("2017-01-01")
