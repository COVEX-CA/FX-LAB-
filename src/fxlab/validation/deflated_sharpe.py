"""Probabilistic Sharpe Ratio (PSR) y Deflated Sharpe Ratio (DSR).

Bailey, D. H. y López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management, 40(5).

## Sharpe sin anualizar — consistencia obligatoria con el registro de fase 3

Todo este módulo trabaja con el Sharpe **sin anualizar**:
`media(retornos) / desviación_típica(retornos)`, sobre la serie de
retornos barra a barra, sin multiplicar por `sqrt(periodos/año)`.

`fxlab.sweep.engine.run_trial` guarda en el registro el Sharpe que devuelve
`vectorbt.Portfolio.sharpe_ratio()`, que sí está anualizado (multiplica
por `sqrt(periodos/año)` derivado de `freq`). Mezclar ese Sharpe anualizado
con las fórmulas de este módulo las rompería sin ningún error visible —
un Sharpe anualizado en H1 es `sqrt(8760)` (~93.6) veces mayor que el
mismo Sharpe sin anualizar. Por eso `sharpe_stats_from_returns` calcula su
propio Sharpe directamente a partir de los retornos, en vez de aceptar un
Sharpe ya calculado por otro sitio: es la única forma de garantizar que
nunca se cuela uno anualizado por error. La relación exacta entre ambos
está verificada explícitamente en
`tests/validation/test_deflated_sharpe.py::test_sharpe_is_not_annualized_and_matches_vectorbt_relationship`.

## Número efectivo de pruebas

`N` en la fórmula de `SR0` debe ser el número de pruebas *efectivamente
independientes*, no el recuento bruto de combinaciones — dos períodos de
media vecinos (20 y 21) están casi perfectamente correlacionados y no son,
a efectos de sesgo de selección, dos pruebas distintas. `effective_n_trials`
lo estima agrupando las series de retornos por clustering jerárquico de
correlación (ver su docstring para el método exacto y sus limitaciones).
`deflated_sharpe_ratio` no calcula esta estimación por su cuenta: recibe
`n_trials_effective` ya calculado, para no acoplar la fórmula del DSR a un
método concreto de estimación.

### El modo de fallo asimétrico de esa estimación

`SR0` crece como `sqrt(2·ln N)`: por encima de N≈5 es casi plano (pasar de
30 a 1000 pruebas solo sube el listón un 57%), pero por debajo de 3 hay un
acantilado, y en `N=1` la deflación es exactamente cero. Es decir:
**sobreestimar N sale barato y subestimarlo sale carísimo**, y subestimarlo
es justo lo que ocurre cuando la rejilla es más redundante — una rejilla de
30 variantes de una misma señal (correlación ~0.9) se colapsa a un único
cluster con cualquier umbral entre 0.1 y 0.5. Ahí el DSR degenera en un
test de significación al 95% sin ninguna corrección por selección, y da
DSR≈0.98 sobre ruido puro.

Por eso `effective_n_trials` avisa (`logging.warning`) cuando el resultado
cae por debajo del mínimo, y `fxlab.validation.report` retira el veredicto
`CANDIDATO` en ese caso. Esta función informa; no corrige el número, porque
inventarle un suelo al valor devuelto falsearía la estimación en vez de
señalar que la rejilla no sostiene la conclusión.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class SharpeStats:
    """Estadísticos de una serie de retornos, sin anualizar."""

    sharpe_non_annualized: float
    """Sufijo explícito: `fxlab.sweep.engine.TrialResult.sharpe_annualized`
    es el mismo estadístico multiplicado por `sqrt(periodos/año)`. Ver el
    docstring del módulo."""
    n: int
    skewness: float
    """Asimetría (g3), estimador de momentos (`scipy.stats.skew(bias=True)`)."""
    kurtosis: float
    """Curtosis NO en exceso (g4): 3.0 para una normal, no 0.0."""


def sharpe_stats_from_returns(returns: pd.Series | np.ndarray) -> SharpeStats:
    """Calcula Sharpe, n, asimetría y curtosis de una serie de retornos.

    El Sharpe se calcula sin anualizar (ver docstring del módulo), con
    desviación típica muestral (`ddof=1`, consistente con el resto del
    proyecto). La curtosis se calcula NO en exceso (`fisher=False`): una
    distribución normal da 3.0, no 0.0 — así coincide directamente con la
    variable `g4` de la fórmula de Bailey y López de Prado.

    Args:
        returns: retornos periodo a periodo (no acumulados, no anualizados).

    Raises:
        ValueError: si hay menos de 2 retornos.
    """
    values = np.asarray(returns, dtype="float64")
    n = len(values)
    if n < 2:
        raise ValueError(f"se necesitan al menos 2 retornos, se recibieron {n}")

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sharpe = mean / std if std > 0 else float("nan")
    skewness = float(stats.skew(values, bias=True))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=True))
    return SharpeStats(sharpe_non_annualized=sharpe, n=n, skewness=skewness, kurtosis=kurtosis)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float,
) -> float:
    """PSR(SR*): probabilidad de que el Sharpe verdadero supere `benchmark_sharpe`.

    PSR(SR*) = Φ[ (SR - SR*) · sqrt(n-1) / sqrt(1 - g3·SR + ((g4-1)/4)·SR²) ]

    Args:
        sharpe: Sharpe observado, SIN anualizar (ver docstring del módulo).
        n: número de retornos usados para calcular `sharpe`.
        skewness: asimetría (g3) de esos mismos retornos.
        kurtosis: curtosis NO en exceso (g4) de esos mismos retornos.
        benchmark_sharpe: umbral SR* contra el que se compara. Para el DSR,
            es `SR0` (ver `expected_max_sharpe_under_luck`); para un PSR
            "simple" sin corregir por selección, suele ser 0.

    Returns:
        Un valor en [0, 1]. `NaN` si el denominador no es un número
        positivo finito (por ejemplo, varianza nula de retornos).

    Raises:
        ValueError: si `n < 2`.
    """
    if n < 2:
        raise ValueError(f"se necesitan al menos 2 retornos, se recibió n={n}")

    denom_sq = 1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe**2
    if not math.isfinite(denom_sq) or denom_sq <= 0:
        return float("nan")

    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe_under_luck(sharpe_variance: float, n_trials: int) -> float:
    """SR0: Sharpe máximo esperable por puro azar tras `n_trials` pruebas independientes.

    SR0 = sqrt(Var[SR_n]) · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]

    Caso especial `n_trials <= 1`: la fórmula general degenera (
    Φ⁻¹(1 - 1/1) = Φ⁻¹(0) = -∞) porque con una sola prueba no hay ningún
    efecto de selección que corregir — no se está eligiendo la mejor entre
    varias. Por definición, `SR0 = 0.0` en ese caso: el DSR se reduce
    entonces al PSR frente a un benchmark de 0, sin deflactar.

    Args:
        sharpe_variance: varianza (muestral, ddof=1) de los Sharpe
            observados entre TODAS las pruebas del experimento — no la
            varianza de los retornos de una sola prueba.
        n_trials: número de pruebas (típicamente `n_trials_effective`, ver
            docstring del módulo).

    Raises:
        ValueError: si `n_trials < 1` o `sharpe_variance < 0`.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials debe ser >= 1, se recibió {n_trials}")
    if sharpe_variance < 0:
        raise ValueError(f"sharpe_variance no puede ser negativa: {sharpe_variance}")

    if n_trials <= 1:
        return 0.0

    std_sr = math.sqrt(sharpe_variance)
    term1 = (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
    term2 = EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(std_sr * (term1 + term2))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    dsr: float
    sr0: float
    sharpe_non_annualized: float
    """Sharpe (sin anualizar) de la combinación evaluada."""
    n_trials_raw: int
    n_trials_effective: int
    sharpe_variance_across_trials: float


def deflated_sharpe_ratio(
    target_returns: pd.Series | np.ndarray,
    all_trial_sharpes: Sequence[float],
    n_trials_effective: int,
) -> DeflatedSharpeResult:
    """DSR de una combinación concreta, dentro de un conjunto de `N` pruebas.

    El DSR es `PSR(SR0)`: la probabilidad de que el Sharpe verdadero de
    `target_returns` supere `SR0`, el Sharpe máximo esperable por azar tras
    `n_trials_effective` pruebas independientes.

    Args:
        target_returns: retornos (sin anualizar) de la combinación evaluada.
        all_trial_sharpes: Sharpe (sin anualizar, mismo criterio — ver
            `sharpe_stats_from_returns`) de TODAS las combinaciones
            probadas en el experimento, incluida la evaluada. Se usa solo
            para `Var[SR_n]`.
        n_trials_effective: número efectivo de pruebas independientes, ya
            estimado (ver `effective_n_trials`).

    Returns:
        `DeflatedSharpeResult` con el DSR, `SR0`, el Sharpe usado, y tanto
        el recuento bruto (`len(all_trial_sharpes)`) como el efectivo.
    """
    target_stats = sharpe_stats_from_returns(target_returns)
    sharpes = np.asarray(all_trial_sharpes, dtype="float64")
    n_raw = len(sharpes)
    sr_variance = float(np.var(sharpes, ddof=1)) if n_raw > 1 else 0.0

    sr0 = expected_max_sharpe_under_luck(sr_variance, n_trials_effective)
    dsr = probabilistic_sharpe_ratio(
        target_stats.sharpe_non_annualized,
        target_stats.n,
        target_stats.skewness,
        target_stats.kurtosis,
        sr0,
    )

    return DeflatedSharpeResult(
        dsr=dsr,
        sr0=sr0,
        sharpe_non_annualized=target_stats.sharpe_non_annualized,
        n_trials_raw=n_raw,
        n_trials_effective=n_trials_effective,
        sharpe_variance_across_trials=sr_variance,
    )


def effective_n_trials(
    returns: pd.DataFrame, distance_threshold: float, min_effective_trials: int
) -> int:
    """Número efectivo de pruebas independientes, por clustering de correlación.

    Método: se calcula la matriz de correlación de Pearson entre las
    columnas de `returns` (cada columna = la serie de retornos de una
    configuración probada); se define una distancia `1 - |correlación|`
    entre cada par; se agrupa con clustering jerárquico aglomerativo de
    enlace promedio (average linkage) sobre esa matriz de distancias; se
    corta el dendrograma en `distance_threshold` y se cuentan los clusters
    resultantes. Configuraciones que caen en el mismo cluster cuentan como
    una sola prueba efectiva.

    Args:
        returns: matriz T (tiempo) × N (configuraciones) de retornos.
        distance_threshold: distancia de corte del dendrograma, en
            `[0, 1]` (0 = solo se fusionan series idénticamente
            correlacionadas: N_efectivo = N; 1 = todo cae en un único
            cluster: N_efectivo = 1). Sin valor por defecto a propósito:
            no hay un umbral "correcto" universal, es una decisión de
            cuánta correlación hace que dos pruebas cuenten como la misma.
        min_effective_trials: mínimo por debajo del cual se emite un
            `logging.warning`. No modifica el valor devuelto — esta función
            informa, no corrige. Quien decide qué hacer con una rejilla
            colapsada es `fxlab.validation.report.evaluate_experiment`.
            Sin valor por defecto: ver `report.MIN_EFFECTIVE_TRIALS`.

    Returns:
        Número de clusters (pruebas efectivas), entre 1 y N.

    Limitaciones (documentadas explícitamente, no resueltas por este método):

    - El resultado depende de `distance_threshold`, elegido por quien
      llama, no por los datos.
    - El enlace promedio es una elección entre varias razonables (enlace
      simple, completo, Ward...); cambia la forma de los clusters en los
      bordes.
    - Solo captura correlación LINEAL (Pearson) entre retornos: dos
      configuraciones podrían depender la una de la otra de forma no
      lineal y aun así quedar en clusters distintos.
    - Con series cortas, las correlaciones estimadas son ruidosas y pueden
      fragmentar en varios clusters lo que en población infinita sería uno
      solo (sobreestimando N_efectivo), o al revés.
    """
    n = returns.shape[1]
    if n < 2:
        _warn_if_collapsed(
            n_effective=n,
            n_raw=n,
            mean_abs_corr=float("nan"),
            min_effective_trials=min_effective_trials,
        )
        return n

    corr = returns.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)  # columnas de varianza nula -> sin correlación
    distance = 1 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2  # fuerza simetría exacta (redondeo de floats)

    condensed = squareform(distance, checks=False)
    linkage_matrix = hierarchy.linkage(condensed, method="average")
    cluster_labels = hierarchy.fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    n_effective = int(len(set(cluster_labels.tolist())))

    off_diagonal = corr[~np.eye(n, dtype=bool)]
    _warn_if_collapsed(
        n_effective=n_effective,
        n_raw=n,
        mean_abs_corr=float(np.abs(off_diagonal).mean()),
        min_effective_trials=min_effective_trials,
    )
    return n_effective


def _warn_if_collapsed(
    n_effective: int, n_raw: int, mean_abs_corr: float, min_effective_trials: int
) -> None:
    """Avisa cuando la rejilla se colapsa por debajo del mínimo de pruebas
    efectivas: es el caso en que el DSR deja de deflactar y se vuelve
    permisivo justo cuando la rejilla es más redundante."""
    if n_effective >= min_effective_trials:
        return
    logger.warning(
        "rejilla colapsada: %d configuraciones brutas -> %d pruebas efectivas "
        "(mínimo %d, correlación media |r| = %.3f). Con tan pocas pruebas "
        "independientes el término de deflación del DSR es nulo o casi nulo, "
        "así que un DSR alto NO corrige el sesgo de selección",
        n_raw,
        n_effective,
        min_effective_trials,
        mean_abs_corr,
    )
