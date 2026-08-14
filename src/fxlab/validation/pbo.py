"""Probability of Backtest Overfitting (PBO) mediante Combinatorially
Symmetric Cross-Validation (CSCV).

Bailey, D. H., Borwein, J., López de Prado, M. y Zhu, Q. J. (2015), "The
Probability of Backtest Overfitting", Journal of Computational Finance.

Algoritmo (CSCV):

1. Matriz `M` de retornos: `T` filas (tiempo) × `N` columnas (una por
   configuración probada).
2. Se parten las `T` filas en `S` bloques disjuntos del mismo tamaño (`S`
   par).
3. Para cada una de las `C(S, S/2)` formas de elegir `S/2` bloques como
   in-sample (el resto es out-of-sample):
   a. `n*` = la configuración (columna) con mayor Sharpe in-sample.
   b. rango de `n*` entre las `N` configuraciones, medido en out-of-sample.
   c. `ω = rango / (N+1)` (rango relativo); `λ = ln(ω / (1-ω))` (logit).
4. `PBO = P(λ < 0)`: fracción de combinaciones en las que la configuración
   ganadora in-sample queda en la mitad inferior out-of-sample.

Interpretación: si el proceso de selección de parámetros no distingue nada
del azar, la configuración ganadora in-sample tiene la misma probabilidad
de acabar en la mitad superior que en la inferior out-of-sample, y
`PBO -> 0.5`. Un PBO bajo (la ganadora in-sample sigue ganando out-of-sample
la mayoría de las veces) es evidencia de que la selección captura algo real
y no solo ruido ajustado al histórico de entrenamiento.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: list[float]
    """Distribución completa de λ, una por combinación in-sample/out-of-sample."""
    is_oos_degradation_non_annualized: list[float]
    """Sharpe in-sample menos Sharpe out-of-sample de la configuración
    ganadora, una por combinación (positivo = peor fuera de muestra).

    Sin anualizar: ambos lados los calcula `_column_sharpe` a partir de los
    retornos, así que el módulo es internamente consistente. El sufijo va
    en el nombre para que no se compare por error con
    `fxlab.sweep.engine.TrialResult.sharpe_annualized`."""
    n_combinations: int
    s: int
    n_configurations: int


def _column_sharpe(returns: np.ndarray) -> np.ndarray:
    """Sharpe sin anualizar de cada columna. `NaN` si la desviación típica es 0."""
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(std > 0, mean / std, np.nan)
    return sharpe


def probability_of_backtest_overfitting(returns: pd.DataFrame, s: int) -> PBOResult:
    """Calcula el PBO de un conjunto de configuraciones probadas.

    Args:
        returns: matriz T (tiempo) × N (configuraciones) de retornos, sin
            anualizar, mismo periodo para todas las columnas.
        s: número de bloques en los que se parte el eje temporal. Debe ser
            par y >= 2 (valor habitual en la literatura: 16, que da
            `C(16,8) = 12870` combinaciones). Sin valor por defecto: la
            elección de `s` cambia cuántas combinaciones se evalúan y qué
            tan finas son las particiones temporales.

    Returns:
        `PBOResult` con el PBO, la distribución completa de logits, y la
        degradación in-sample/out-of-sample de la configuración ganadora
        en cada combinación.

    Raises:
        ValueError: si `s` no es par y >= 2, si `T` no es múltiplo exacto
            de `s` (los bloques deben ser del mismo tamaño, no aproximado),
            o si hay menos de 2 configuraciones.
    """
    if s < 2 or s % 2 != 0:
        raise ValueError(f"s debe ser par y >= 2, se recibió {s}")

    t, n = returns.shape
    if n < 2:
        raise ValueError(f"se necesitan al menos 2 configuraciones, se recibieron {n}")
    if t % s != 0:
        raise ValueError(
            f"T={t} filas no se puede dividir en {s} bloques del mismo tamaño "
            f"(T debe ser múltiplo exacto de s)"
        )

    values = returns.to_numpy(dtype="float64")
    block_size = t // s
    blocks = [values[i * block_size : (i + 1) * block_size] for i in range(s)]

    half = s // 2
    logits: list[float] = []
    degradations: list[float] = []

    for is_block_idx in combinations(range(s), half):
        oos_block_idx = [b for b in range(s) if b not in is_block_idx]

        is_returns = np.concatenate([blocks[b] for b in is_block_idx], axis=0)
        oos_returns = np.concatenate([blocks[b] for b in oos_block_idx], axis=0)

        is_sharpe = _column_sharpe(is_returns)
        oos_sharpe = _column_sharpe(oos_returns)

        if np.all(np.isnan(is_sharpe)):
            raise ValueError(
                "todas las configuraciones tienen desviación típica nula en un "
                "tramo in-sample: no se puede identificar la configuración ganadora"
            )
        best_idx = int(np.nanargmax(is_sharpe))

        # rango de la ganadora in-sample dentro de out-of-sample: 1=peor,
        # N=mejor (empates -> rango medio, vía scipy.stats.rankdata).
        # Un Sharpe OOS de NaN (columna sin variación) se trata como el
        # peor resultado posible: no hay evidencia de que fuera de muestra
        # replicara ninguna ventaja.
        oos_sharpe_for_rank = np.where(np.isnan(oos_sharpe), -np.inf, oos_sharpe)
        ranks = rankdata(oos_sharpe_for_rank, method="average")
        rank_of_best = float(ranks[best_idx])

        omega = rank_of_best / (n + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)  # evita log(0) / log(inf) en los extremos
        logit = math.log(omega / (1 - omega))
        logits.append(logit)

        oos_sharpe_best = oos_sharpe[best_idx]
        if np.isnan(oos_sharpe_best):
            degradation = float("nan")
        else:
            degradation = float(is_sharpe[best_idx] - oos_sharpe_best)
        degradations.append(degradation)

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0))

    return PBOResult(
        pbo=pbo,
        logits=logits,
        is_oos_degradation_non_annualized=degradations,
        n_combinations=len(logits),
        s=s,
        n_configurations=n,
    )
