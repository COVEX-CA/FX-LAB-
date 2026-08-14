"""El script de línea de comandos, de extremo a extremo sobre un registro
SQLite real.

Cubre el cableado que los tests de `build_report` no tocan: leer del
registro tanto las métricas como la matriz de retornos, montar la evidencia
y llamar a validation. Ya no hay ningún fichero externo de retornos: el
barrido los persiste en el propio `trials.db` y el informe los lee de ahí.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from fxlab.sweep.costs import CostModel
from fxlab.sweep.engine import run_sweep
from fxlab.sweep.registry import TrialRegistry

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_report.py"


def _load_script() -> ModuleType:
    """`scripts/` no es un paquete importable: se carga por ruta."""
    spec = importlib.util.spec_from_file_location("generate_report", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_report"] = module
    spec.loader.exec_module(module)
    return module


def _build_fixtures(tmp_path: Path) -> tuple[Path, str]:
    """Registro con retornos ya persistidos; devuelve (db, id de trial objetivo)."""
    records = [
        {"slow_period": slow, "fast_period": fast, "slow_ma": "sma"}
        for slow, fast in itertools.product((10, 20, 30, 40, 50), (2, 4))
    ]
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, size=(1600, len(records)))

    db_path = tmp_path / "trials.db"
    start = pd.Timestamp("2010-01-01", tz="UTC")
    end = pd.Timestamp("2010-03-01", tz="UTC")
    returns_index = pd.date_range("2010-01-01", periods=1600, freq="1h", tz="UTC")
    with TrialRegistry(db_path) as registry:
        for position, record in enumerate(records):
            series = pd.Series(data[:, position], index=returns_index)
            registry.record_trial(
                experiment_id="exp-cli",
                symbol="EUR/USD",
                interval="1HOUR",
                partition="development",
                start_date=start,
                end_date=end,
                data_hash="h",
                params=record,
                n_trades=10,
                total_return=float(series.sum()),
                sharpe_annualized=float(series.mean() / series.std(ddof=1)),
                max_drawdown=-0.05,
                profit_factor=1.2,
                win_rate=0.55,
                expectancy=0.01,
                note=None,
                returns=series,
            )
        target = str(registry.load_experiment("exp-cli")["id"].iloc[0])

    return db_path, target


def _synthetic_ohlc(
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


def test_script_generates_a_report_end_to_end(tmp_path: Path) -> None:
    db_path, target = _build_fixtures(tmp_path)
    output = tmp_path / "informe.html"
    script = _load_script()

    script.main(
        [
            "exp-cli",
            "--registry",
            str(db_path),
            "--output",
            str(output),
            "--metric",
            "sharpe_annualized",
            "--target",
            target,
            "--distance-threshold",
            "0.3",
            "--pbo-s",
            "4",
            "--min-effective-trials",
            "5",
        ]
    )

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    body = html.split("<body>", 1)[1]
    assert 'id="verdict"' in body
    assert body.index('id="verdict"') < body.index("plotly-graph-div")
    # Autocontenido: ninguna etiqueta del documento carga un recurso externo.
    # (No basta con buscar URLs en el texto: el bundle de plotly.js lleva
    # dentro la URL por defecto de topojson como valor de configuración, que
    # solo se usaría en gráficos geográficos, y este informe no dibuja
    # ninguno. Lo que importa es que el marcado no referencie nada de fuera.)
    assert not re.search(r"<script[^>]+\bsrc\s*=", html)
    assert not re.search(r"<link[^>]+\bhref\s*=", html)
    assert not re.search(r"<img[^>]+\bsrc\s*=", html)
    assert "Plotly.newPlot" in html  # el bundle sí está embebido y se usa


def test_report_generates_end_to_end_from_a_real_sweep(tmp_path: Path) -> None:
    # La tubería entera sin pasos manuales: un barrido real persiste sus
    # retornos, y el informe se genera leyéndolos del mismo registro.
    bid, ask = _synthetic_ohlc(1000, seed=11)
    grid = {
        "slow_ma": ["sma", "ema"],
        "slow_period": [20],
        "fast_ma": ["ema"],
        "fast_period": [5, 10, 15],
        "n_bars": [5],
        "use_adx_filter": [False],
        "adx_threshold": [None],
        "adx_period": [14],
    }

    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as registry:
        run_sweep(
            bid,
            ask,
            grid,
            CostModel(commission=0.00007),
            registry,
            experiment_id="real",
            symbol="EUR/USD",
            interval="1HOUR",
            freq="1h",
        )
        target = str(registry.load_experiment("real")["id"].iloc[0])

    output = tmp_path / "informe.html"
    script = _load_script()
    script.main(
        [
            "real",
            "--registry",
            str(db_path),
            "--output",
            str(output),
            "--metric",
            "sharpe_annualized",
            "--target",
            target,
            "--distance-threshold",
            "0.3",
            "--pbo-s",
            "4",
            "--min-effective-trials",
            "5",
        ]
    )

    assert output.exists()
    body = output.read_text(encoding="utf-8").split("<body>", 1)[1]
    assert 'id="verdict"' in body
    # el barrido sintético cubre unos días de 2004: es un subperíodo, y el
    # aviso de cobertura debe aparecer arriba del todo
    assert 'id="data-coverage"' in body
    assert body.index('id="data-coverage"') < body.index('id="verdict"')


def test_script_requires_the_quality_parameters(tmp_path: Path) -> None:
    # Sin valores por defecto: omitir --min-effective-trials debe fallar,
    # no elegir un mínimo en silencio.
    script = _load_script()
    with pytest.raises(SystemExit):
        script._parse_args(
            [
                "exp-cli",
                "--registry",
                str(tmp_path / "trials.db"),
                "--output",
                str(tmp_path / "out.html"),
                "--metric",
                "sharpe_annualized",
                "--target",
                "x",
                "--distance-threshold",
                "0.3",
                "--pbo-s",
                "4",
            ]
        )
