from __future__ import annotations

from pathlib import Path

import pytest

from fxlab.sweep.config import load_config


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
