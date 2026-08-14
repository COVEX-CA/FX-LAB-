"""Informe HTML: estructura, restricciones y casos límite de rejilla.

Todo sobre datos sintéticos. Se comprueban las restricciones que la fase
declara tan importantes como el contenido: el veredicto va antes que
cualquier gráfico, ninguna fecha del holdout llega a la salida, un veredicto
negativo se informa con el mismo detalle que uno positivo, y ningún objeto
público expone una configuración "ganadora" suelta.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxlab.reporting.report import ReportInputs, build_report
from fxlab.reporting.sections import RankedConfigurations, rank_configurations
from fxlab.split import HOLDOUT_START
from fxlab.sweep.engine import TrialResult
from fxlab.validation.pbo import probability_of_backtest_overfitting
from fxlab.validation.report import (
    MIN_EFFECTIVE_TRIALS,
    ExperimentEvidence,
    Verdict,
    VerdictReport,
    evaluate_experiment,
)
from fxlab.validation.walk_forward import (
    Fold,
    WalkForwardFoldResult,
    WalkForwardResult,
    WindowMode,
)

_T = 1600
_PBO_S = 4  # C(4,2)=6 combinaciones: rápido. La estructura de s=16 ya está
# cubierta en tests/validation/test_pbo.py.
_DISTANCE_THRESHOLD = 0.3
_METRIC = "sharpe_annualized"


def _index(periods: int = _T, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC")


def _params_records(n_slow: int, n_fast: int) -> list[dict[str, object]]:
    """Rejilla cartesiana de dos parámetros que varían, más uno fijo."""
    return [
        {"slow_period": slow, "fast_period": fast, "slow_ma": "sma"}
        for slow, fast in itertools.product(
            range(10, 10 + n_slow * 10, 10), range(2, 2 + n_fast * 2, 2)
        )
    ]


def _trials_frame(records: list[dict[str, object]], metric_values: np.ndarray) -> pd.DataFrame:
    """Imita la forma que devuelve `TrialRegistry.load_experiment`."""
    return pd.DataFrame(
        {
            "id": range(1, len(records) + 1),
            "experiment_id": "exp-test",
            "symbol": "EUR/USD",
            "interval": "1HOUR",
            "partition": "development",
            "start_date": "2010-01-01T00:00:00+00:00",
            "end_date": "2010-03-01T00:00:00+00:00",
            "data_hash": "h",
            "code_version": "abc123",
            **{f"param_{key}": [r[key] for r in records] for key in sorted(records[0])},
            "n_trades": 10,
            _METRIC: metric_values,
            "total_return": 0.1,
            "max_drawdown": -0.05,
            "profit_factor": 1.2,
            "win_rate": 0.55,
            "expectancy": 0.01,
            "note": None,
        }
    )


def _returns_frame(data: np.ndarray, records: list[dict[str, object]]) -> pd.DataFrame:
    """Columnas = id de trial como cadena, que es el contrato de ReportInputs."""
    return pd.DataFrame(
        data,
        index=_index(data.shape[0]),
        columns=[str(i) for i in range(1, len(records) + 1)],
    )


def _trial_result(sharpe: float) -> TrialResult:
    return TrialResult(
        n_trades=10,
        total_return=0.1,
        sharpe_annualized=sharpe,
        max_drawdown=-0.05,
        profit_factor=1.2,
        win_rate=0.55,
        expectancy=0.01,
        note=None,
        returns=pd.Series(
            [0.0], index=pd.date_range("2010-01-01", periods=1, freq="1h", tz="UTC")
        ),
    )


def _walk_forward(test_sharpe: float, n_folds: int = 3) -> WalkForwardResult:
    index = _index()
    folds = [
        WalkForwardFoldResult(
            fold=Fold(
                train_start=index[0],
                train_end=index[199 + 200 * i],
                test_start=index[200 + 200 * i],
                test_end=index[399 + 200 * i],
            ),
            best_params=None,
            train_result=_trial_result(test_sharpe + 0.5),
            test_result=_trial_result(test_sharpe),
            sharpe_degradation_annualized=0.5,
            note=None,
        )
        for i in range(n_folds)
    ]
    return WalkForwardResult(
        mode=WindowMode.ANCHORED, train_size=200, test_size=200, step=200, folds=folds
    )


def _verdict_report(verdict: Verdict, *, collapsed: bool = False) -> VerdictReport:
    """VerdictReport construido a mano: aquí se prueba la presentación, no
    los criterios de veredicto (cubiertos en tests/validation/)."""
    return VerdictReport(
        verdict=verdict,
        dsr=0.97 if verdict is Verdict.CANDIDATO else 0.30,
        pbo=0.05 if verdict is Verdict.CANDIDATO else 0.60,
        n_trials_raw=10,
        n_trials_effective=2 if collapsed else 8,
        grid_collapsed=collapsed,
        walk_forward_mean_degradation_annualized=0.5,
        walk_forward_mean_test_sharpe_annualized=1.0,
        reasons=[f"motivo de prueba para {verdict.value}"],
    )


def _inputs(
    *,
    verdict: VerdictReport,
    n_slow: int = 5,
    n_fast: int = 2,
    seed: int = 0,
    redundant: bool = False,
) -> ReportInputs:
    records = _params_records(n_slow, n_fast)
    n_configs = len(records)
    rng = np.random.default_rng(seed)
    if redundant:
        base = rng.normal(0.0, 0.01, _T)
        data = np.column_stack([base + rng.normal(0.0, 0.003, _T) for _ in range(n_configs)])
    else:
        data = rng.normal(0.0, 0.01, size=(_T, n_configs))

    all_returns = _returns_frame(data, records)
    trials = _trials_frame(records, data.mean(axis=0) / data.std(axis=0, ddof=1))
    evidence = ExperimentEvidence(
        target_returns=all_returns.iloc[:, 0],
        all_returns=all_returns,
        walk_forward=_walk_forward(1.0),
        distance_threshold=_DISTANCE_THRESHOLD,
        pbo_s=_PBO_S,
        min_effective_trials=MIN_EFFECTIVE_TRIALS,
    )
    pbo = probability_of_backtest_overfitting(all_returns, _PBO_S) if n_configs >= 2 else None
    return ReportInputs(
        experiment_id="exp-test",
        trials=trials,
        evidence=evidence,
        verdict=verdict,
        pbo=pbo,
        metric=_METRIC,
    )


def _body(html: str) -> str:
    """Cuerpo del documento, sin el bundle de plotly.js que va en <head>."""
    return html.split("<body>", 1)[1]


# --- los tres veredictos ---------------------------------------------------


@pytest.mark.parametrize("verdict", list(Verdict))
def test_report_is_generated_for_every_verdict(verdict: Verdict, tmp_path: Path) -> None:
    output = tmp_path / f"{verdict.value}.html"
    build_report(_inputs(verdict=_verdict_report(verdict)), output)

    html = output.read_text(encoding="utf-8")
    body = _body(html)
    assert verdict.value.upper() in body

    # Mismo detalle sea cual sea el veredicto: no hay modo degradado.
    for section_id in (
        "verdict",
        "grid-diagnostics",
        "parameter-surface",
        "walk-forward",
        "pbo-lambda",
        "equity",
    ):
        assert f'id="{section_id}"' in body, f"falta la sección {section_id} en {verdict}"
    assert body.count("plotly-graph-div") >= 5


def test_every_verdict_produces_a_report_of_comparable_detail(tmp_path: Path) -> None:
    sizes = {}
    for verdict in Verdict:
        output = tmp_path / f"{verdict.value}.html"
        build_report(_inputs(verdict=_verdict_report(verdict)), output)
        sizes[verdict] = len(_body(output.read_text(encoding="utf-8")))
    # Ninguno puede ser una versión recortada de otro.
    assert min(sizes.values()) > 0.8 * max(sizes.values()), sizes


# --- estructura obligatoria ------------------------------------------------


def test_verdict_appears_before_any_chart(tmp_path: Path) -> None:
    output = tmp_path / "order.html"
    build_report(_inputs(verdict=_verdict_report(Verdict.CANDIDATO)), output)
    body = _body(output.read_text(encoding="utf-8"))

    verdict_position = body.index('id="verdict"')
    first_chart_position = body.index("plotly-graph-div")
    assert verdict_position < first_chart_position

    # Y el orden completo de secciones es el que impone la fase.
    order = [
        body.index(f'id="{name}"')
        for name in (
            "verdict",
            "grid-diagnostics",
            "parameter-surface",
            "walk-forward",
            "pbo-lambda",
            "equity",
        )
    ]
    assert order == sorted(order)


def test_report_contains_no_date_beyond_the_holdout_cut(tmp_path: Path) -> None:
    output = tmp_path / "holdout.html"
    build_report(_inputs(verdict=_verdict_report(Verdict.CANDIDATO)), output)

    # Se escanea el cuerpo, no el documento entero: el bundle de plotly.js va
    # en <head> y lleva dentro constantes de calendario con años lejanos
    # (2500-01-01, 5000-01-01...) que no son datos del informe. Todo el
    # contenido —incluidos los datos de cada figura, que Plotly embebe como
    # JSON dentro del <body>— sí queda cubierto.
    body = _body(output.read_text(encoding="utf-8"))
    # Sin \b al final: en un ISO datetime ("2010-01-01T00:00:00+00:00") no hay
    # frontera de palabra entre el día y la T, y el patrón no encontraría nada.
    found = {
        pd.Timestamp(match, tz="UTC")
        for match in re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", body)
    }
    assert found, "el escaneo no encontró ninguna fecha: el test no estaría comprobando nada"
    beyond = sorted(d for d in found if d >= HOLDOUT_START)
    assert not beyond, f"el informe contiene fechas de la partición reservada: {beyond[:5]}"


def test_build_report_refuses_returns_that_reach_the_holdout(tmp_path: Path) -> None:
    inputs = _inputs(verdict=_verdict_report(Verdict.CANDIDATO))
    holdout_index = pd.date_range(
        HOLDOUT_START - pd.Timedelta(hours=_T - 1), periods=_T, freq="1h", tz="UTC"
    )
    poisoned = inputs.evidence.all_returns.set_axis(holdout_index)
    inputs = ReportInputs(
        experiment_id=inputs.experiment_id,
        trials=inputs.trials,
        evidence=ExperimentEvidence(
            target_returns=poisoned.iloc[:, 0],
            all_returns=poisoned,
            walk_forward=inputs.evidence.walk_forward,
            distance_threshold=_DISTANCE_THRESHOLD,
            pbo_s=_PBO_S,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        ),
        verdict=inputs.verdict,
        pbo=inputs.pbo,
        metric=_METRIC,
    )
    with pytest.raises(ValueError, match="holdout"):
        build_report(inputs, tmp_path / "nope.html")


# --- rejilla colapsada -----------------------------------------------------


def test_collapsed_grid_warning_appears_in_the_output(tmp_path: Path) -> None:
    # Camino completo: el veredicto lo calcula validation, no se construye a mano.
    inputs = _inputs(verdict=_verdict_report(Verdict.RUIDO), redundant=True)
    verdict = evaluate_experiment(inputs.evidence)
    assert verdict.grid_collapsed, "la rejilla redundante debería colapsar"

    output = tmp_path / "collapsed.html"
    build_report(
        ReportInputs(
            experiment_id=inputs.experiment_id,
            trials=inputs.trials,
            evidence=inputs.evidence,
            verdict=verdict,
            pbo=inputs.pbo,
            metric=_METRIC,
        ),
        output,
    )

    body = _body(output.read_text(encoding="utf-8"))
    assert "colapso de rejilla" in body.lower() or "rejilla colapsada" in body.lower()
    assert body.count("collapsed-warning") >= 2  # veredicto y diagnóstico de rejilla


# --- tamaños extremos de rejilla -------------------------------------------


def test_single_configuration_grid_does_not_break(tmp_path: Path) -> None:
    records = _params_records(1, 1)
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, size=(_T, 1))
    all_returns = _returns_frame(data, records)
    trials = _trials_frame(records, np.array([0.5]))

    inputs = ReportInputs(
        experiment_id="exp-single",
        trials=trials,
        evidence=ExperimentEvidence(
            target_returns=all_returns.iloc[:, 0],
            all_returns=all_returns,
            walk_forward=_walk_forward(1.0),
            distance_threshold=_DISTANCE_THRESHOLD,
            pbo_s=_PBO_S,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        ),
        verdict=_verdict_report(Verdict.NO_CONCLUYENTE, collapsed=True),
        pbo=None,  # el PBO no está definido con una sola configuración
        metric=_METRIC,
    )
    output = build_report(inputs, tmp_path / "single.html")
    body = _body(output.read_text(encoding="utf-8"))
    assert 'id="pbo-lambda"' in body
    assert "una sola configuración" in body


def test_large_grid_does_not_break(tmp_path: Path) -> None:
    output = tmp_path / "large.html"
    build_report(_inputs(verdict=_verdict_report(Verdict.RUIDO), n_slow=30, n_fast=10), output)
    body = _body(output.read_text(encoding="utf-8"))
    assert 'id="grid-diagnostics"' in body
    # 300 configuraciones > el tope de la vista de correlación: debe decirlo.
    assert "muestra determinista" in body


def test_walk_forward_section_survives_an_empty_walk_forward(tmp_path: Path) -> None:
    inputs = _inputs(verdict=_verdict_report(Verdict.RUIDO))
    empty = WalkForwardResult(mode=WindowMode.ANCHORED, train_size=0, test_size=0, step=0, folds=[])
    inputs = ReportInputs(
        experiment_id=inputs.experiment_id,
        trials=inputs.trials,
        evidence=ExperimentEvidence(
            target_returns=inputs.evidence.target_returns,
            all_returns=inputs.evidence.all_returns,
            walk_forward=empty,
            distance_threshold=_DISTANCE_THRESHOLD,
            pbo_s=_PBO_S,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        ),
        verdict=inputs.verdict,
        pbo=inputs.pbo,
        metric=_METRIC,
    )
    output = build_report(inputs, tmp_path / "empty_wf.html")
    assert 'id="walk-forward"' in _body(output.read_text(encoding="utf-8"))


# --- nada devuelve "la mejor estrategia" -----------------------------------


def test_ranked_configurations_always_carries_the_verdict() -> None:
    inputs = _inputs(verdict=_verdict_report(Verdict.CANDIDATO))
    ranked = rank_configurations(inputs.trials, _METRIC, inputs.verdict, top_n=10)

    assert isinstance(ranked, RankedConfigurations)
    names = {f.name for f in fields(RankedConfigurations)}
    # La tabla nunca viaja sin su contexto: no se puede obtener un ranking
    # sin el veredicto y el recuento efectivo de pruebas.
    assert {"verdict", "n_trials_raw", "n_trials_effective", "grid_collapsed"} <= names
    assert ranked.verdict is inputs.verdict.verdict
    assert ranked.n_trials_effective == inputs.verdict.n_trials_effective


def test_ranking_table_in_the_html_is_accompanied_by_the_verdict(tmp_path: Path) -> None:
    output = tmp_path / "ranking.html"
    build_report(_inputs(verdict=_verdict_report(Verdict.CANDIDATO)), output)
    body = _body(output.read_text(encoding="utf-8"))

    ranking_position = body.index("Configuraciones mejor puntuadas")
    context_position = body.index("ranking-context")
    table_position = body.index('class="metrics ranked"')
    # El contexto (veredicto + recuento efectivo) va entre el título y la tabla.
    assert ranking_position < context_position < table_position
    assert "no dice cuál es" in body


def test_no_public_reporting_function_returns_a_bare_configuration() -> None:
    import fxlab.reporting as reporting_package
    import fxlab.reporting.sections as sections

    exported = [
        getattr(sections, name)
        for name in dir(sections)
        if not name.startswith("_") and callable(getattr(sections, name))
    ]
    returning_configs = [
        f for f in exported if getattr(f, "__name__", "") in {"rank_configurations"}
    ]
    assert returning_configs, "rank_configurations debería estar entre las públicas"
    # El único punto de entrada del paquete es build_report, que escribe un
    # informe completo; no hay ninguna función que devuelva una configuración.
    assert set(reporting_package.__all__) == {"ReportInputs", "build_report"}
