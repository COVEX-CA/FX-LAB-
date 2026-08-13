"""Veredicto: combina DSR, PBO y la degradación walk-forward en una
conclusión categórica — nunca en "la mejor estrategia".

## Por qué este módulo no lee directamente el registro de la fase 3

`fxlab.sweep.registry` guarda, por combinación, métricas agregadas (Sharpe,
retorno total, etc.) — no la serie de retornos barra a barra. El PBO
necesita la matriz T×N de retornos de todas las combinaciones, y el DSR
necesita la asimetría y curtosis de la distribución completa de retornos de
la combinación evaluada: ninguna de las dos se puede reconstruir a partir
de un Sharpe ya agregado. Por eso este módulo opera sobre `ExperimentEvidence`
(series de retornos), no sobre filas de SQLite. Reconstruir esas series a
partir de un experimento real (recomputando con
`fxlab.sweep.engine.run_trial`) es responsabilidad de quien orquesta la
evaluación — fuera de alcance de esta fase, que no ejecuta ningún barrido
real.

## Los umbrales — fijados ahora, antes de ver ningún resultado

Es la razón de ser de que esta fase preceda a la ejecución del barrido: si
los umbrales se eligieran después de ver los números, dejarían de ser un
criterio y pasarían a ser una racionalización. Cada uno se justifica por
separado más abajo, junto a su constante.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from fxlab.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    effective_n_trials,
    sharpe_stats_from_returns,
)
from fxlab.validation.pbo import probability_of_backtest_overfitting
from fxlab.validation.walk_forward import WalkForwardResult

# --- Umbrales -------------------------------------------------------------

DSR_SIGNIFICANT = 0.95
"""DSR >= este valor: el Sharpe observado, ya corregido por sesgo de
selección, no-normalidad y número efectivo de pruebas, supera SR0 con al
menos un 95% de probabilidad. Es la convención de confianza al 95%
unilateral estándar en estadística, y es literalmente el umbral que Bailey
y López de Prado (2014) usan en los ejemplos numéricos de su propio paper.
No es una elección arbitraria de este proyecto."""

DSR_NOISE_FLOOR = 0.5
"""DSR < este valor: es MENOS probable que no que el Sharpe verdadero
supere lo esperable por puro azar — el punto de indiferencia de una
apuesta justa. Por debajo de él no hay ninguna base para hablar de ventaja,
solo para descartarla."""

PBO_CANDIDATE_MAX = 0.2
"""PBO < este valor para poder hablar de candidato. PBO=0.5 es el punto en
el que la selección de parámetros no aporta nada sobre el azar (mitad de
las veces gana la mitad superior out-of-sample, mitad la inferior). 0.2 es
un margen deliberadamente conservador por debajo de ese punto medio: no
basta con estar "algo mejor que el azar", tiene que ser holgado, porque el
propio PBO es una estimación con varianza (ver limitaciones documentadas en
`fxlab.validation.pbo`). Es el mismo margen que usan implementaciones de
referencia de CSCV en la práctica, no un punto exacto derivado de la teoría
del método — el método en sí solo fija con precisión el 0.5, no el 0.2."""

PBO_NOISE_FLOOR = 0.5
"""PBO >= este valor: la configuración ganadora in-sample tiene la misma
probabilidad (o menos) de acabar en la mitad superior que en la inferior
out-of-sample. Selección indistinguible del azar, por definición del
método (Bailey, Borwein, López de Prado y Zhu, 2015)."""


class Verdict(Enum):
    RUIDO = "ruido"
    NO_CONCLUYENTE = "no_concluyente"
    CANDIDATO = "candidato"


@dataclass(frozen=True)
class ExperimentEvidence:
    """Entrada necesaria para evaluar un experimento.

    Args:
        target_returns: retornos (sin anualizar) de la combinación bajo
            evaluación.
        all_returns: matriz T (tiempo) × N (configuraciones) de retornos
            de TODAS las combinaciones probadas en el experimento,
            incluida `target_returns` como una de sus columnas.
        walk_forward: resultado de `fxlab.validation.walk_forward.run_walk_forward`
            sobre la misma combinación.
        distance_threshold: umbral de corte para `effective_n_trials` (ver
            su docstring). Sin valor por defecto.
        pbo_s: número de bloques para `probability_of_backtest_overfitting`.
            Sin valor por defecto.
    """

    target_returns: pd.Series
    all_returns: pd.DataFrame
    walk_forward: WalkForwardResult
    distance_threshold: float
    pbo_s: int


@dataclass(frozen=True)
class VerdictReport:
    verdict: Verdict
    dsr: float
    pbo: float
    n_trials_raw: int
    n_trials_effective: int
    walk_forward_mean_degradation: float | None
    walk_forward_mean_test_sharpe: float | None
    reasons: list[str]
    """Explicación legible de por qué se llegó a este veredicto, un
    elemento por cada condición evaluada."""


def evaluate_experiment(evidence: ExperimentEvidence) -> VerdictReport:
    """Combina DSR, PBO y degradación walk-forward en un veredicto categórico.

    No selecciona ni ordena combinaciones: opina solo sobre
    `evidence.target_returns`, dado el conjunto completo de pruebas
    (`evidence.all_returns`) en el que se enmarca.

    Reglas (evaluadas en este orden):

    - `RUIDO` si `DSR < DSR_NOISE_FLOOR` o `PBO >= PBO_NOISE_FLOOR`: basta
      con que una de las dos señales sea indistinguible del azar, o peor.
    - `CANDIDATO` si a la vez `DSR >= DSR_SIGNIFICANT`,
      `PBO < PBO_CANDIDATE_MAX`, y el Sharpe medio de prueba (out-of-sample)
      del walk-forward es positivo: las tres señales, independientes entre
      sí, tienen que apuntar en la misma dirección.
    - `NO_CONCLUYENTE` en cualquier otro caso — el default ante evidencia
      mixta o insuficiente, no un tercio automático.
    """
    target_stats = sharpe_stats_from_returns(evidence.target_returns)
    all_sharpes = [
        sharpe_stats_from_returns(evidence.all_returns[col]).sharpe
        for col in evidence.all_returns.columns
    ]

    n_effective = effective_n_trials(evidence.all_returns, evidence.distance_threshold)
    dsr_result = deflated_sharpe_ratio(evidence.target_returns, all_sharpes, n_effective)
    pbo_result = probability_of_backtest_overfitting(evidence.all_returns, evidence.pbo_s)

    fold_degradations = [
        f.degradation for f in evidence.walk_forward.folds if f.degradation is not None
    ]
    fold_test_sharpes = [
        f.test_result.sharpe
        for f in evidence.walk_forward.folds
        if f.test_result is not None and f.test_result.sharpe is not None
    ]
    mean_degradation = (
        sum(fold_degradations) / len(fold_degradations) if fold_degradations else None
    )
    mean_test_sharpe = (
        sum(fold_test_sharpes) / len(fold_test_sharpes) if fold_test_sharpes else None
    )

    reasons: list[str] = [
        f"Sharpe (sin anualizar) de la combinación evaluada: "
        f"{target_stats.sharpe:.4f} (n={target_stats.n})",
        f"DSR = {dsr_result.dsr:.4f} (umbral significativo >= {DSR_SIGNIFICANT}, "
        f"suelo de ruido < {DSR_NOISE_FLOOR})",
        f"PBO = {pbo_result.pbo:.4f} sobre {pbo_result.n_combinations} combinaciones "
        f"(umbral candidato < {PBO_CANDIDATE_MAX}, suelo de ruido >= {PBO_NOISE_FLOOR})",
        f"pruebas: {dsr_result.n_trials_raw} brutas, {dsr_result.n_trials_effective} efectivas",
    ]
    if mean_test_sharpe is not None:
        reasons.append(
            f"Sharpe medio de prueba walk-forward (out-of-sample): {mean_test_sharpe:.4f}"
        )
    else:
        reasons.append(
            "walk-forward sin ningún pliegue evaluable (sin operaciones en entrenamiento)"
        )

    if dsr_result.dsr < DSR_NOISE_FLOOR or pbo_result.pbo >= PBO_NOISE_FLOOR:
        verdict = Verdict.RUIDO
        if dsr_result.dsr < DSR_NOISE_FLOOR:
            reasons.append(f"RUIDO: DSR {dsr_result.dsr:.4f} < suelo {DSR_NOISE_FLOOR}")
        if pbo_result.pbo >= PBO_NOISE_FLOOR:
            reasons.append(f"RUIDO: PBO {pbo_result.pbo:.4f} >= suelo {PBO_NOISE_FLOOR}")
    elif (
        dsr_result.dsr >= DSR_SIGNIFICANT
        and pbo_result.pbo < PBO_CANDIDATE_MAX
        and mean_test_sharpe is not None
        and mean_test_sharpe > 0
    ):
        verdict = Verdict.CANDIDATO
        reasons.append(
            "CANDIDATO: DSR, PBO y Sharpe de prueba walk-forward cumplen los tres umbrales"
        )
    else:
        verdict = Verdict.NO_CONCLUYENTE
        reasons.append(
            "NO_CONCLUYENTE: no se cumplen todas las condiciones de CANDIDATO ni de RUIDO"
        )

    return VerdictReport(
        verdict=verdict,
        dsr=dsr_result.dsr,
        pbo=pbo_result.pbo,
        n_trials_raw=dsr_result.n_trials_raw,
        n_trials_effective=dsr_result.n_trials_effective,
        walk_forward_mean_degradation=mean_degradation,
        walk_forward_mean_test_sharpe=mean_test_sharpe,
        reasons=reasons,
    )
