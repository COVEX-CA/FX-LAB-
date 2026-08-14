"""El script de línea de comandos, de extremo a extremo sobre un registro
SQLite real y un parquet de retornos sintéticos.

Cubre el cableado que los tests de `build_report` no tocan: leer del
registro, leer el parquet, montar la evidencia y llamar a validation.
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


def _build_fixtures(tmp_path: Path) -> tuple[Path, Path, str]:
    records = [
        {"slow_period": slow, "fast_period": fast, "slow_ma": "sma"}
        for slow, fast in itertools.product((10, 20, 30, 40, 50), (2, 4))
    ]
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, size=(1600, len(records)))

    db_path = tmp_path / "trials.db"
    start = pd.Timestamp("2010-01-01", tz="UTC")
    end = pd.Timestamp("2010-03-01", tz="UTC")
    with TrialRegistry(db_path) as registry:
        for position, record in enumerate(records):
            series = pd.Series(data[:, position])
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
            )
        # Las columnas de la matriz de retornos son los id que acaba de
        # asignar el registro: es el contrato de ReportInputs.
        ids = [str(v) for v in registry.load_experiment("exp-cli")["id"]]

    returns = pd.DataFrame(
        data,
        index=pd.date_range("2010-01-01", periods=1600, freq="1h", tz="UTC"),
        columns=ids,
    )
    returns_path = tmp_path / "returns.parquet"
    returns.to_parquet(returns_path)
    return db_path, returns_path, ids[0]


def test_script_generates_a_report_end_to_end(tmp_path: Path) -> None:
    db_path, returns_path, target = _build_fixtures(tmp_path)
    output = tmp_path / "informe.html"
    script = _load_script()

    script.main(
        [
            "exp-cli",
            "--registry",
            str(db_path),
            "--returns",
            str(returns_path),
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
                "--returns",
                str(tmp_path / "returns.parquet"),
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
