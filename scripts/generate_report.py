#!/usr/bin/env python
"""CLI para generar el informe HTML de un experimento ya registrado.

Sin lógica de decisión: lee del registro tanto las métricas como la matriz
de retornos (`TrialRegistry.load_returns_matrix`), delega el veredicto en
`fxlab.validation` y el dibujo en `fxlab.reporting`. Este script no decide
nada, no ejecuta backtests y no toca el holdout. No necesita ningún fichero
externo de retornos: el barrido ya los persiste en el propio `trials.db`.

Limitación conocida: el walk-forward no lo persiste ninguna fase del
proyecto, así que desde línea de comandos el informe se genera con la
sección de walk-forward vacía (y lo dice en la propia sección). Para
incluirla hay que llamar a `fxlab.reporting.build_report` desde código,
pasando el `WalkForwardResult` en el `ExperimentEvidence`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fxlab.reporting import ReportInputs, build_report
from fxlab.sweep.registry import TrialRegistry
from fxlab.validation.pbo import probability_of_backtest_overfitting
from fxlab.validation.report import ExperimentEvidence, evaluate_experiment
from fxlab.validation.walk_forward import WalkForwardResult, WindowMode

logger = logging.getLogger("fxlab.generate_report")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="identificador del experimento en el registro")
    parser.add_argument("--registry", type=Path, required=True, help="ruta a trials.db")
    parser.add_argument("--output", type=Path, required=True, help="ruta del HTML de salida")
    parser.add_argument(
        "--metric",
        required=True,
        help="columna del registro que dibujan las superficies de parámetros",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="id de trial (columna de la matriz de retornos) de la configuración evaluada",
    )
    # Sin valores por defecto: son criterios de calidad, y el proyecto exige
    # que cada experimento los declare explícitamente (ver AGENTS.md y los
    # docstrings de fxlab.validation).
    parser.add_argument("--distance-threshold", type=float, required=True)
    parser.add_argument("--pbo-s", type=int, required=True)
    parser.add_argument("--min-effective-trials", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    with TrialRegistry(args.registry) as registry:
        trials = registry.load_experiment(args.experiment_id)
        all_returns = registry.load_returns_matrix(args.experiment_id)

    if args.target not in all_returns.columns:
        raise SystemExit(
            f"--target {args.target!r} no es un id de trial del experimento "
            f"{args.experiment_id!r}: elige uno de las columnas de la matriz de retornos"
        )

    logger.warning(
        "el walk-forward no está persistido: el informe se genera con esa sección "
        "vacía. Para incluirla, llama a fxlab.reporting.build_report desde código."
    )
    evidence = ExperimentEvidence(
        target_returns=all_returns[args.target],
        all_returns=all_returns,
        walk_forward=WalkForwardResult(
            mode=WindowMode.ANCHORED, train_size=0, test_size=0, step=0, folds=[]
        ),
        distance_threshold=args.distance_threshold,
        pbo_s=args.pbo_s,
        min_effective_trials=args.min_effective_trials,
    )

    verdict = evaluate_experiment(evidence)
    pbo = (
        probability_of_backtest_overfitting(all_returns, args.pbo_s)
        if all_returns.shape[1] >= 2
        else None
    )

    path = build_report(
        ReportInputs(
            experiment_id=args.experiment_id,
            trials=trials,
            evidence=evidence,
            verdict=verdict,
            pbo=pbo,
            metric=args.metric,
        ),
        args.output,
    )
    logger.info("informe escrito en %s (veredicto: %s)", path, verdict.verdict.value)


if __name__ == "__main__":
    main()
