"""Composición del informe HTML autocontenido.

## De dónde salen los datos, y por qué el llamador los inyecta

El registro de la fase 3 guarda métricas agregadas por combinación, no la
serie de retornos barra a barra. Pero el diagnóstico de rejilla, el
histograma de λ y las curvas de equity necesitan la matriz T×N de retornos,
que no está en el registro y no se puede reconstruir desde un Sharpe ya
agregado. Recalcularla aquí exigiría ejecutar backtests, que es justo lo que
este paquete no hace.

Por eso `ReportInputs` recibe el `ExperimentEvidence` y el `VerdictReport`
ya construidos por quien orquesta la evaluación, y el `experiment_id` solo
para leer del registro la rejilla de parámetros y sus métricas agregadas.
Las capas quedan separadas: `validation` decide, `reporting` dibuja.

El `PBOResult` también entra como parámetro en vez de salir del
`VerdictReport`. No es una elección de diseño sino una consecuencia:
`evaluate_experiment` calcula el PBO internamente pero solo devuelve el
escalar, y `fxlab.validation` es zona prohibida (ver `AGENTS.md`, §9), así
que no se le puede añadir el campo. El coste es que el llamador calcula el
PBO una segunda vez para obtener la distribución completa de λ.

## Holdout

`build_report` valida activamente que ninguna entrada contenga fechas
posteriores al corte, y falla si las hay. No basta con "no dibujarlas": si
llegan hasta aquí, algo ha ido mal aguas arriba y el informe no debe
disimularlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.offline as pyo

from fxlab.reporting.sections import (
    equity_curves_section,
    grid_diagnostics_section,
    parameter_surface_section,
    pbo_lambda_section,
    rank_configurations,
    verdict_section,
    walk_forward_section,
)
from fxlab.split import HOLDOUT_START
from fxlab.validation.pbo import PBOResult
from fxlab.validation.report import ExperimentEvidence, VerdictReport

TOP_CONFIGURATIONS = 10

_STYLE = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0 auto; max-width: 1100px; padding: 2rem 1.5rem; color: #0f172a;
       line-height: 1.55; }
h1 { font-size: 1.7rem; border-bottom: 2px solid #0f172a; padding-bottom: .4rem; }
h2 { font-size: 1.3rem; margin-top: 2.5rem; }
h3 { font-size: 1.05rem; margin-top: 1.6rem; }
section { margin-bottom: 2rem; }
.verdict-box { padding: 1.2rem 1.5rem; border-radius: 8px; margin: 1rem 0; }
.verdict-label { font-size: 2rem; font-weight: 700; letter-spacing: .06em; }
.collapsed-warning { background: #fef3c7; border-left: 5px solid #b45309;
                     padding: .9rem 1.1rem; margin: 1rem 0; }
table.metrics { border-collapse: collapse; margin: 1rem 0; font-size: .93rem; }
table.metrics th, table.metrics td { border: 1px solid #cbd5e1; padding: .35rem .7rem;
                                     text-align: left; }
table.metrics th { background: #f1f5f9; }
table.ranked { display: block; overflow-x: auto; max-width: 100%; }
table.folds { display: block; overflow-x: auto; max-width: 100%; }
.note { color: #475569; font-size: .92rem; }
.reading { background: #f8fafc; border-left: 4px solid #64748b; padding: .8rem 1.1rem; }
.ranking-context { background: #f1f5f9; padding: .8rem 1.1rem; border-radius: 6px; }
ul.reasons { font-size: .93rem; color: #334155; }
"""


@dataclass(frozen=True)
class ReportInputs:
    """Todo lo que el informe necesita, ya calculado por otras capas.

    Args:
        experiment_id: identificador del experimento en el registro.
        trials: `TrialRegistry.load_experiment(experiment_id)`.
        evidence: la misma evidencia que se pasó a `evaluate_experiment`.
        verdict: su resultado.
        pbo: `probability_of_backtest_overfitting(evidence.all_returns,
            evidence.pbo_s)`, para la distribución completa de λ (ver el
            docstring del módulo). `None` si la rejilla tiene una sola
            configuración, en cuyo caso el PBO no está definido.
        metric: columna de `trials` que dibujan las superficies de
            parámetros. Sin valor por defecto, siguiendo el patrón del
            proyecto: qué métrica se mira cambia qué se concluye.

    Las columnas de `evidence.all_returns` deben ser los `id` de trial de
    `trials`, como cadenas y en el mismo orden: es lo que permite ligar una
    celda de un mapa de calor con su curva de equity. Se liga por `id` y no
    por parámetros porque `TrialRegistry.load_experiment` ya expande
    `params_json` en columnas `param_*` y descarta el JSON original, y
    reconstruirlo perdería información (un parámetro `None` vuelve como
    `NaN`). Se valida en `build_report` y falla si no coincide, en vez de
    emparejar por posición y arriesgar un desalineamiento silencioso.
    """

    experiment_id: str
    trials: pd.DataFrame
    evidence: ExperimentEvidence
    verdict: VerdictReport
    pbo: PBOResult | None
    metric: str


_PARAM_PREFIX = "param_"


def _expand_params(trials: pd.DataFrame) -> pd.DataFrame:
    """Columnas de parámetros del registro, sin el prefijo `param_`.

    `TrialRegistry.load_experiment` ya hace la expansión desde JSON; aquí
    solo se recorta el prefijo para que los ejes de los mapas de calor
    lleven el nombre real del parámetro.
    """
    columns = [c for c in trials.columns if c.startswith(_PARAM_PREFIX)]
    if not columns:
        raise ValueError(
            "el registro no trae ninguna columna param_*: ¿se cargó con "
            "TrialRegistry.load_experiment?"
        )
    renamed = {c: c[len(_PARAM_PREFIX) :] for c in columns}
    return trials[columns].rename(columns=renamed).reset_index(drop=True)


def _validate(inputs: ReportInputs) -> None:
    if inputs.trials.empty:
        raise ValueError(f"el experimento {inputs.experiment_id!r} no tiene ninguna prueba")
    if inputs.metric not in inputs.trials.columns:
        raise ValueError(
            f"metric={inputs.metric!r} no es una columna del registro; "
            f"disponibles: {sorted(inputs.trials.columns)}"
        )

    if "id" not in inputs.trials.columns:
        raise ValueError("el registro cargado no trae la columna 'id'")
    expected = [str(v) for v in inputs.trials["id"]]
    actual = [str(c) for c in inputs.evidence.all_returns.columns]
    if actual != expected:
        raise ValueError(
            "las columnas de evidence.all_returns deben ser los id de trial de "
            "trials, como cadenas y en el mismo orden, para poder ligar cada "
            f"configuración con su curva de equity (recibidas {len(actual)}, "
            f"esperadas {len(expected)})"
        )

    index = inputs.evidence.all_returns.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("evidence.all_returns necesita un DatetimeIndex timezone-aware en UTC")
    _assert_no_holdout(inputs)


def _assert_no_holdout(inputs: ReportInputs) -> None:
    """El informe nunca muestra datos del holdout. Si llegan hasta aquí, se
    falla en vez de recortarlos en silencio: recortarlos ocultaría que algo
    aguas arriba ya los ha mirado."""
    index = inputs.evidence.all_returns.index
    if (index >= HOLDOUT_START).any():
        raise ValueError(
            f"evidence.all_returns contiene fechas >= {HOLDOUT_START.date()} "
            "(partición reservada): el informe no muestra holdout"
        )

    for column in ("start_date", "end_date"):
        if column in inputs.trials.columns:
            dates = pd.to_datetime(inputs.trials[column], utc=True, errors="coerce")
            if (dates >= HOLDOUT_START).any():
                raise ValueError(
                    f"trials.{column} contiene fechas >= {HOLDOUT_START.date()}: "
                    "el informe no muestra holdout"
                )

    for fold_result in inputs.evidence.walk_forward.folds:
        fold = fold_result.fold
        if max(fold.train_end, fold.test_end) >= HOLDOUT_START:
            raise ValueError(
                f"un pliegue del walk-forward llega a {fold.test_end}, en la "
                "partición reservada: el informe no muestra holdout"
            )


def build_report(inputs: ReportInputs, output_path: Path) -> Path:
    """Genera el informe HTML autocontenido y devuelve su ruta.

    El orden de las secciones es fijo y no depende del veredicto: un
    resultado negativo se informa con el mismo detalle que uno positivo,
    porque un resultado negativo es un resultado. No hay modo degradado.

    Raises:
        ValueError: si falta la métrica pedida, si las columnas de retornos
            no casan con el registro, o si alguna entrada contiene fechas de
            la partición reservada.
    """
    _validate(inputs)

    trials = inputs.trials.reset_index(drop=True)
    params = _expand_params(trials)
    metric_values = pd.to_numeric(trials[inputs.metric], errors="coerce")
    best_index = int(metric_values.idxmax()) if metric_values.notna().any() else 0

    ranked = rank_configurations(trials, inputs.metric, inputs.verdict, TOP_CONFIGURATIONS)
    min_effective = inputs.evidence.min_effective_trials

    # El orden de esta lista es la estructura obligatoria del informe: el
    # veredicto va antes que cualquier figura, las curvas de equity al final.
    body = "".join(
        [
            verdict_section(inputs.verdict, min_effective),
            grid_diagnostics_section(
                inputs.evidence.all_returns,
                inputs.evidence.distance_threshold,
                inputs.verdict,
                min_effective,
            ),
            parameter_surface_section(trials, params, inputs.metric, best_index),
            walk_forward_section(inputs.evidence.walk_forward, inputs.verdict),
            pbo_lambda_section(inputs.pbo),
            equity_curves_section(inputs.evidence.all_returns, params, ranked, best_index),
        ]
    )

    code_versions = sorted({str(v) for v in trials.get("code_version", pd.Series(dtype=str)) if v})
    provenance = f"código: {', '.join(code_versions)}" if code_versions else "código: n/d"

    document = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe de experimento {inputs.experiment_id}</title>
<style>{_STYLE}</style>
<script type="text/javascript">{pyo.get_plotlyjs()}</script>
</head>
<body>
<h1>Experimento {inputs.experiment_id}</h1>
<p class="note">Partición de desarrollo únicamente. {provenance}.</p>
{body}
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
