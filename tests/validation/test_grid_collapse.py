"""Regresión del suelo de pruebas efectivas.

El modo de fallo que fija este módulo: cuando la rejilla es redundante
(muchas variantes de una misma señal), el clustering la colapsa a muy pocos
clusters, `SR0` cae a cero y el DSR deja de deflactar — precisamente cuando
más falta hace. Sobre ruido puro eso produce DSR≈0.98, por encima del umbral
de significación. Los dos tests de aquí fijan las dos mitades del contrato:

1. Rejilla redundante sobre ruido puro: NUNCA `CANDIDATO`, en ninguna semilla.
2. Rejilla diversa con ventaja real: sigue dando `CANDIDATO`. Sin este
   segundo test, el suelo podría "aprobar" el primero simplemente rompiendo
   la detección legítima.

El `WalkForwardResult` se construye a mano en vez de ejecutar un
walk-forward real: lo que se prueba aquí es la lógica del veredicto, no el
motor de pliegues (que tiene sus propios tests en `test_walk_forward.py`),
y un walk-forward real por semilla haría el test inviablemente lento.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fxlab.sweep.engine import TrialResult
from fxlab.validation.deflated_sharpe import effective_n_trials
from fxlab.validation.report import (
    MIN_EFFECTIVE_TRIALS,
    ExperimentEvidence,
    Verdict,
    evaluate_experiment,
)
from fxlab.validation.walk_forward import (
    Fold,
    WalkForwardFoldResult,
    WalkForwardResult,
    WindowMode,
)

_T = 1600  # múltiplo de 16, para poder usar s=16 en PBO
_DISTANCE_THRESHOLD = 0.3


def _trial_result(sharpe_annualized: float) -> TrialResult:
    return TrialResult(
        n_trades=10,
        total_return=0.1,
        sharpe_annualized=sharpe_annualized,
        max_drawdown=-0.05,
        profit_factor=1.2,
        win_rate=0.55,
        expectancy=0.01,
        note=None,
        returns=pd.Series([0.0], index=pd.date_range("2010-01-01", periods=1, freq="1h", tz="UTC")),
    )


def _walk_forward_with_test_sharpe(sharpe_annualized: float) -> WalkForwardResult:
    """Walk-forward sintético de dos pliegues, con el Sharpe de prueba dado."""
    index = pd.date_range("2010-01-01", periods=400, freq="1h", tz="UTC")
    folds = [
        WalkForwardFoldResult(
            fold=Fold(
                train_start=index[0],
                train_end=index[99 + 100 * i],
                test_start=index[100 + 100 * i],
                test_end=index[199 + 100 * i],
            ),
            best_params=None,
            train_result=_trial_result(sharpe_annualized + 0.5),
            test_result=_trial_result(sharpe_annualized),
            sharpe_degradation_annualized=0.5,
            note=None,
        )
        for i in range(2)
    ]
    return WalkForwardResult(
        mode=WindowMode.ANCHORED, train_size=100, test_size=100, step=100, folds=folds
    )


def _redundant_noise_grid(n_columns: int, seed: int) -> pd.DataFrame:
    """`n_columns` variantes ruidosas de UNA sola señal de fondo, todas ruido
    puro (media cero). Es la forma de una rejilla que solo varía un parámetro
    en pasos pequeños: correlación mutua ~0.9."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, _T)
    return pd.DataFrame(
        np.column_stack([base + rng.normal(0.0, 0.003, _T) for _ in range(n_columns)]),
        columns=[f"c{i}" for i in range(n_columns)],
    )


def test_redundant_grid_on_pure_noise_never_yields_candidato() -> None:
    for seed in range(12):
        returns = _redundant_noise_grid(n_columns=30, seed=seed)
        best_col = returns.columns[int(np.argmax(returns.mean().to_numpy()))]

        report = evaluate_experiment(
            ExperimentEvidence(
                target_returns=returns[best_col],
                all_returns=returns,
                walk_forward=_walk_forward_with_test_sharpe(1.0),
                distance_threshold=_DISTANCE_THRESHOLD,
                pbo_s=16,
                min_effective_trials=MIN_EFFECTIVE_TRIALS,
            )
        )

        assert report.verdict is not Verdict.CANDIDATO, (
            f"semilla {seed}: rejilla redundante sobre ruido puro dio CANDIDATO "
            f"(DSR={report.dsr:.4f}, PBO={report.pbo:.4f}, "
            f"n_efectivo={report.n_trials_effective})"
        )
        assert report.grid_collapsed
        assert report.n_trials_effective < MIN_EFFECTIVE_TRIALS


def test_diverse_grid_with_real_edge_still_yields_candidato() -> None:
    # Contrapartida obligatoria: el suelo no debe conseguir su objetivo
    # rompiendo la detección legítima.
    n_columns = 10
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        data = rng.normal(0.0, 0.01, size=(_T, n_columns))
        data[:, 0] += 0.003  # ventaja real y persistente en c0
        returns = pd.DataFrame(data, columns=[f"c{i}" for i in range(n_columns)])

        report = evaluate_experiment(
            ExperimentEvidence(
                target_returns=returns["c0"],
                all_returns=returns,
                walk_forward=_walk_forward_with_test_sharpe(1.0),
                distance_threshold=_DISTANCE_THRESHOLD,
                pbo_s=16,
                min_effective_trials=MIN_EFFECTIVE_TRIALS,
            )
        )

        assert report.verdict is Verdict.CANDIDATO, (
            f"semilla {seed}: ventaja real en rejilla diversa NO dio CANDIDATO "
            f"(DSR={report.dsr:.4f}, PBO={report.pbo:.4f}, "
            f"n_efectivo={report.n_trials_effective}, motivos={report.reasons})"
        )
        assert not report.grid_collapsed
        assert report.n_trials_effective >= MIN_EFFECTIVE_TRIALS


def test_effective_n_trials_warns_when_the_grid_collapses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    returns = _redundant_noise_grid(n_columns=30, seed=0)
    with caplog.at_level(logging.WARNING, logger="fxlab.validation.deflated_sharpe"):
        n_effective = effective_n_trials(
            returns,
            distance_threshold=_DISTANCE_THRESHOLD,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        )

    assert n_effective < MIN_EFFECTIVE_TRIALS
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    # el aviso debe indicar recuento bruto, efectivo y correlación media
    assert "30 configuraciones brutas" in message
    assert f"-> {n_effective} pruebas efectivas" in message
    assert f"mínimo {MIN_EFFECTIVE_TRIALS}" in message
    assert "correlación media |r| = 0." in message


def test_effective_n_trials_does_not_warn_on_a_healthy_grid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, size=(_T, 30)))
    with caplog.at_level(logging.WARNING, logger="fxlab.validation.deflated_sharpe"):
        n_effective = effective_n_trials(
            returns,
            distance_threshold=_DISTANCE_THRESHOLD,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        )

    assert n_effective >= MIN_EFFECTIVE_TRIALS
    assert caplog.records == []


def test_collapsed_grid_does_not_block_the_ruido_verdict() -> None:
    # El suelo retira CANDIDATO, no RUIDO: el PBO no depende de n_efectivo,
    # y un DSR bajo SIN deflación es aún más concluyente que uno con ella.
    # Se fuerza RUIDO con un walk-forward irrelevante y una rejilla colapsada
    # cuya combinación evaluada es claramente mala.
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, _T)
    data = np.column_stack([base + rng.normal(0.0, 0.003, _T) for _ in range(30)])
    data[:, 0] -= 0.004  # la evaluada es sistemáticamente peor
    returns = pd.DataFrame(data, columns=[f"c{i}" for i in range(30)])

    report = evaluate_experiment(
        ExperimentEvidence(
            target_returns=returns["c0"],
            all_returns=returns,
            walk_forward=_walk_forward_with_test_sharpe(-1.0),
            distance_threshold=_DISTANCE_THRESHOLD,
            pbo_s=16,
            min_effective_trials=MIN_EFFECTIVE_TRIALS,
        )
    )

    assert report.grid_collapsed
    assert report.verdict is Verdict.RUIDO
