"""El corazón de la fase de validación: si estos dos tests no se cumplen,
DSR y PBO no distinguen ruido de ventaja real y todo lo demás en este
paquete es ruido con apariencia de rigor.

- Ruido puro (sin ninguna ventaja verdadera): PBO debe rondar 0.5 y el DSR
  de la mejor configuración encontrada por azar no debe alcanzar el umbral
  de significación.
- Ventaja real e inyectada (una configuración con desplazamiento de media
  persistente frente al resto, ruido puro): PBO debe caer claramente por
  debajo de 0.5 y el DSR de esa configuración debe ser alto.

Un solo sorteo de ruido puro tiene mucha varianza en el PBO (las 12870
combinaciones de CSCV con S=16 son sorteos muy dependientes entre sí, no
muestras independientes): por eso el test de ruido promedia sobre varios
conjuntos sintéticos independientes en vez de fiarse de una sola semilla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    effective_n_trials,
    sharpe_stats_from_returns,
)
from fxlab.validation.pbo import probability_of_backtest_overfitting
from fxlab.validation.report import DSR_SIGNIFICANT

_T = 1600  # múltiplo de 16, para poder usar s=16 en PBO
_DISTANCE_THRESHOLD = 0.3


def _noise_returns(n_columns: int, seed: int, std: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, std, size=(_T, n_columns))
    return pd.DataFrame(data, columns=[f"c{i}" for i in range(n_columns)])


def test_pure_noise_pbo_averages_near_one_half_and_best_dsr_is_not_significant() -> None:
    n_columns = 30
    n_seeds = 15

    pbos: list[float] = []
    dsrs: list[float] = []
    for seed in range(n_seeds):
        returns = _noise_returns(n_columns, seed)

        pbo_result = probability_of_backtest_overfitting(returns, s=16)
        pbos.append(pbo_result.pbo)

        sharpes = [sharpe_stats_from_returns(returns[col]).sharpe for col in returns.columns]
        best_col = returns.columns[int(np.argmax(sharpes))]
        n_effective = effective_n_trials(returns, distance_threshold=_DISTANCE_THRESHOLD)
        dsr_result = deflated_sharpe_ratio(returns[best_col], sharpes, n_effective)
        dsrs.append(dsr_result.dsr)

    mean_pbo = float(np.mean(pbos))
    assert 0.35 < mean_pbo < 0.65, f"PBO medio en ruido puro debería rondar 0.5, dio {mean_pbo}"

    # Si el módulo dice que la mejor configuración de puro ruido es
    # significativa (DSR >= umbral de "candidato"), está roto.
    assert all(dsr < DSR_SIGNIFICANT for dsr in dsrs), (
        f"DSR de la mejor configuración de ruido puro alcanzó el umbral de "
        f"significación en al menos una semilla: {dsrs}"
    )


def test_real_persistent_edge_gives_low_pbo_and_high_dsr() -> None:
    n_columns = 10
    edge_mean = 0.003

    for seed in (1, 7, 42):
        returns = _noise_returns(n_columns, seed)
        returns["c0"] = returns["c0"] + edge_mean

        pbo_result = probability_of_backtest_overfitting(returns, s=16)
        assert pbo_result.pbo < 0.2, (
            f"PBO debería caer claramente por debajo de 0.5 con una ventaja "
            f"real inyectada (semilla {seed}), dio {pbo_result.pbo}"
        )

        sharpes = [sharpe_stats_from_returns(returns[col]).sharpe for col in returns.columns]
        n_effective = effective_n_trials(returns, distance_threshold=_DISTANCE_THRESHOLD)
        dsr_result = deflated_sharpe_ratio(returns["c0"], sharpes, n_effective)
        assert dsr_result.dsr > 0.9, (
            f"DSR de la configuración con ventaja real debería ser alto "
            f"(semilla {seed}), dio {dsr_result.dsr}"
        )
