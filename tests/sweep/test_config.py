from __future__ import annotations

from pathlib import Path

import pytest

from fxlab.sweep.config import load_config
from fxlab.sweep.engine import iter_ma_cross_grid

_REPO = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_parses_a_valid_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1HOUR"
        commission: 0.00007
        grid:
          slow_ma: [sma]
          slow_period: [50]
          fast_ma: [ema]
          fast_period: [10]
          n_bars: [5]
          use_adx_filter: [false]
          adx_threshold: [null]
          adx_period: [14]
        """,
    )

    config = load_config(path)

    assert config.experiment_id == "exp1"
    assert config.symbol == "EUR/USD"
    assert config.cost_model.commission == pytest.approx(0.00007)
    assert config.grid["slow_period"] == [50]


def test_load_config_requires_commission_with_no_implicit_default(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1HOUR"
        grid:
          slow_ma: [sma]
        """,
    )

    with pytest.raises(ValueError, match="commission"):
        load_config(path)


def test_load_config_requires_grid(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1HOUR"
        commission: 0.0001
        """,
    )

    with pytest.raises(ValueError, match="grid"):
        load_config(path)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="mapeo"):
        load_config(path)


def test_load_config_defaults_strategy_to_pullback(tmp_path: Path) -> None:
    # Sin `strategy`, se asume pullback (los configs originales no lo declaran).
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1HOUR"
        commission: 0.00007
        grid:
          slow_ma: [sma]
        """,
    )
    assert load_config(path).strategy == "pullback"


def test_load_config_parses_an_explicit_strategy(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1DAY"
        strategy: ma_cross
        commission: 0.00007
        grid:
          fast_ma: [ema]
        """,
    )
    assert load_config(path).strategy == "ma_cross"


def test_load_config_rejects_an_unknown_strategy(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        experiment_id: exp1
        symbol: "EUR/USD"
        interval: "1HOUR"
        strategy: bollinger
        commission: 0.00007
        grid:
          x: [1]
        """,
    )
    with pytest.raises(ValueError, match="strategy"):
        load_config(path)


def test_pre_registered_ma_cross_configs_load_and_expand_to_300() -> None:
    # Fija el grid bloqueado: cualquier edición accidental que cambie el nº de
    # combinaciones (o rompa el YAML) salta aquí.
    for name in ("ma_cross_eurusd_d1", "ma_cross_eurusd_4h"):
        config = load_config(_REPO / "configs" / f"{name}.yaml")
        assert config.strategy == "ma_cross"
        assert len(list(iter_ma_cross_grid(config.grid))) == 300
