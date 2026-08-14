"""PSR contra su forma cerrada normal (g3=0, g4=3) y casos límite del DSR."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from scipy.stats import norm

from fxlab.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_luck,
    probabilistic_sharpe_ratio,
    sharpe_stats_from_returns,
)


@pytest.mark.parametrize(
    "sharpe,n,benchmark_sharpe",
    [
        (0.5, 100, 0.0),
        (1.2, 250, 0.5),
        (-0.3, 50, 0.0),
        (0.0, 10, 0.0),
    ],
)
def test_psr_matches_normal_closed_form_when_skew_zero_kurtosis_three(
    sharpe: float, n: int, benchmark_sharpe: float
) -> None:
    # Para una distribución normal (asimetría 0, curtosis no-exceso 3), el
    # denominador de PSR se reduce a sqrt(1 + 0.5*SR^2): se calcula aquí
    # directamente con scipy.stats.norm, sin pasar por probabilistic_sharpe_ratio,
    # como verificación independiente de la fórmula general.
    denom = math.sqrt(1 + 0.5 * sharpe**2)
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / denom
    expected = float(norm.cdf(z))

    actual = probabilistic_sharpe_ratio(
        sharpe, n, skewness=0.0, kurtosis=3.0, benchmark_sharpe=benchmark_sharpe
    )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_psr_raises_below_two_observations() -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(1.0, 1, skewness=0.0, kurtosis=3.0, benchmark_sharpe=0.0)


def test_psr_is_nan_when_denominator_is_not_positive() -> None:
    # skewness muy grande junto a un Sharpe grande puede anular o volver
    # negativo el término bajo la raíz: PSR no está definido ahí.
    result = probabilistic_sharpe_ratio(
        10.0, 100, skewness=10.0, kurtosis=1.0, benchmark_sharpe=0.0
    )
    assert math.isnan(result)


def test_sr0_is_zero_with_a_single_trial() -> None:
    # Sin selección entre varias pruebas no hay sesgo de selección que corregir.
    assert expected_max_sharpe_under_luck(sharpe_variance=1.0, n_trials=1) == 0.0


def test_sr0_is_zero_when_sharpe_variance_is_zero() -> None:
    # Todas las pruebas tuvieron exactamente el mismo Sharpe: no hay
    # dispersión de la que "el mejor por azar" pueda beneficiarse.
    assert expected_max_sharpe_under_luck(sharpe_variance=0.0, n_trials=1000) == 0.0


def test_sr0_grows_monotonically_and_stays_finite_for_very_large_n() -> None:
    small = expected_max_sharpe_under_luck(sharpe_variance=1.0, n_trials=10)
    medium = expected_max_sharpe_under_luck(sharpe_variance=1.0, n_trials=1_000)
    large = expected_max_sharpe_under_luck(sharpe_variance=1.0, n_trials=1_000_000)
    assert 0.0 < small < medium < large
    assert math.isfinite(large)


def test_sr0_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe_under_luck(sharpe_variance=1.0, n_trials=0)
    with pytest.raises(ValueError):
        expected_max_sharpe_under_luck(sharpe_variance=-1.0, n_trials=10)


def test_deflated_sharpe_ratio_with_a_single_trial_has_no_selection_correction() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))

    result = deflated_sharpe_ratio(
        returns,
        all_trial_sharpes=[sharpe_stats_from_returns(returns).sharpe_non_annualized],
        n_trials_effective=1,
    )

    assert result.sr0 == 0.0
    assert result.n_trials_raw == 1
    assert result.n_trials_effective == 1
    # Sin deflactar por selección, el DSR se reduce al PSR frente a benchmark 0.
    stats = sharpe_stats_from_returns(returns)
    expected = probabilistic_sharpe_ratio(
        stats.sharpe_non_annualized, stats.n, stats.skewness, stats.kurtosis, 0.0
    )
    assert result.dsr == pytest.approx(expected)


def test_sharpe_is_not_annualized_and_matches_vectorbt_relationship() -> None:
    # fxlab.sweep.engine.run_trial guarda el Sharpe anualizado que devuelve
    # vectorbt.Portfolio.sharpe_ratio() (raw_sharpe * sqrt(periodos/año)).
    # Este módulo debe calcular el Sharpe SIN anualizar directamente de los
    # retornos: se verifica aquí la relación exacta entre ambos.
    rng = np.random.default_rng(1)
    n = 500
    index = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    close_series = pd.Series(close, index=index)

    freq = "1h"
    portfolio = vbt.Portfolio.from_holding(close_series, freq=freq)
    annualized_sharpe = float(portfolio.sharpe_ratio())

    # portfolio.returns() (no close.pct_change()) es la serie exacta que
    # vectorbt anualiza: incluye la barra inicial con retorno 0.0, que
    # pct_change().dropna() no tiene, y desplazaría ligeramente la media y
    # la desviación típica si se usara en su lugar.
    raw_stats = sharpe_stats_from_returns(portfolio.returns())

    ann_factor = math.sqrt(pd.Timedelta("365D") / pd.Timedelta(freq))
    assert annualized_sharpe == pytest.approx(
        raw_stats.sharpe_non_annualized * ann_factor, rel=1e-6
    )
    assert raw_stats.sharpe_non_annualized != pytest.approx(annualized_sharpe, rel=1e-3)
