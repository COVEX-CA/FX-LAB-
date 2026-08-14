"""Motor de barrido: ejecuta una rejilla de parámetros de `pullback` sobre
VectorBT, con costes siempre activos, y registra cada resultado.

No selecciona nada, no ordena por rentabilidad, no imprime "el mejor
resultado": eso es exactamente lo que la fase de validación estadística
existe para hacer con cuidado. Este módulo solo genera y registra números.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from functools import partial

import pandas as pd
import vectorbt as vbt

from fxlab.indicators.moving_averages import MOVING_AVERAGES
from fxlab.split import Partition, filter_partition
from fxlab.strategies.pullback import generate_signals
from fxlab.sweep.costs import CostModel, execution_prices
from fxlab.sweep.registry import TrialRegistry, hash_dataset


@dataclass(frozen=True)
class SweepParams:
    """Una combinación de parámetros de `fxlab.strategies.pullback`."""

    slow_ma: str
    slow_period: int
    fast_ma: str
    fast_period: int
    n_bars: int
    use_adx_filter: bool
    adx_threshold: float | None
    adx_period: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrialResult:
    """Resultado de evaluar una `SweepParams`. `n_trades=0` es un resultado
    válido, no un error: `total_return`..`expectancy` quedan en `None` y
    `note` explica el motivo.

    El Sharpe lleva el sufijo `_annualized` en el nombre a propósito: es el
    que devuelve `vectorbt.Portfolio.sharpe_ratio()`, ya multiplicado por
    `sqrt(periodos/año)` según `freq`. `fxlab.validation` trabaja con el
    Sharpe SIN anualizar (sufijo `_non_annualized`), y mezclar ambos no
    produce ningún error visible, solo un resultado equivocado: en H1 el
    factor es `sqrt(8760)` ≈ 93.6. La convención va en el nombre para que
    la mezcla sea imposible de cometer en silencio.
    """

    n_trades: int
    total_return: float | None
    sharpe_annualized: float | None
    max_drawdown: float | None
    profit_factor: float | None
    win_rate: float | None
    expectancy: float | None
    note: str | None


def _build_ma(name: str, period: int) -> Callable[[pd.Series], pd.Series]:
    if name not in MOVING_AVERAGES:
        raise ValueError(f"media desconocida: {name!r}, debe ser una de {sorted(MOVING_AVERAGES)}")
    # MOVING_AVERAGES está tipado como Callable[[pd.Series], pd.Series] (la
    # firma común del registro), pero cada función real acepta además su
    # propio `period` con nombre; eso es justo lo que se fija aquí.
    return partial(MOVING_AVERAGES[name], period=period)  # type: ignore[call-arg]


def iter_grid(grid: Mapping[str, list[object]]) -> Iterator[SweepParams]:
    """Expande una rejilla (nombre de campo -> lista de valores) en todas
    las combinaciones de `SweepParams`, por producto cartesiano.

    No elimina combinaciones redundantes (p.ej. distintos `adx_threshold`
    cuando `use_adx_filter=False` dan el mismo resultado): cada punto de la
    rejilla configurada es una prueba distinta y se registra como tal —
    optimizar qué se prueba está fuera de alcance de esta fase.
    """
    fields = (
        "slow_ma",
        "slow_period",
        "fast_ma",
        "fast_period",
        "n_bars",
        "use_adx_filter",
        "adx_threshold",
        "adx_period",
    )
    missing = [f for f in fields if f not in grid]
    if missing:
        raise ValueError(f"faltan campos en la rejilla: {missing}")

    values = [grid[f] for f in fields]
    for combo in itertools.product(*values):
        # `fields` recorre exactamente los campos de SweepParams, en orden;
        # mypy no puede verificarlo porque `grid` es un Mapping genérico.
        yield SweepParams(**dict(zip(fields, combo, strict=True)))  # type: ignore[arg-type]


def run_trial(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    params: SweepParams,
    cost_model: CostModel,
    *,
    freq: str,
    init_cash: float = 10_000.0,
) -> TrialResult:
    """Genera las señales de `params`, simula con VectorBT y devuelve las métricas.

    Las señales se generan sobre el precio bid (`bid` es el OHLC usado por
    los indicadores y las reglas de la estrategia); el ask solo se usa para
    el precio de ejecución de las compras (ver `fxlab.sweep.costs`).

    Args:
        bid: OHLC bid, partición ya recortada.
        ask: OHLC ask, mismo índice que `bid`.
        params: combinación de parámetros a evaluar.
        cost_model: comisión (obligatoria); el spread se deriva de `bid`/`ask`.
        freq: frecuencia de barra para VectorBT (p.ej. "1h"), usada para anualizar.
        init_cash: capital inicial nominal de la simulación.
    """
    if not bid.index.equals(ask.index):
        raise ValueError("bid y ask deben compartir exactamente el mismo índice")

    slow_ma_func = _build_ma(params.slow_ma, params.slow_period)
    fast_ma_func = _build_ma(params.fast_ma, params.fast_period)

    signals = generate_signals(
        bid,
        slow_ma_func,
        fast_ma_func,
        params.n_bars,
        use_adx_filter=params.use_adx_filter,
        adx_threshold=params.adx_threshold,
        adx_period=params.adx_period,
    )

    price = execution_prices(
        signals.long_entries,
        signals.long_exits,
        signals.short_entries,
        signals.short_exits,
        bid["close"],
        ask["close"],
    )

    portfolio = vbt.Portfolio.from_signals(
        close=bid["close"],
        entries=signals.long_entries,
        exits=signals.long_exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        price=price,
        fees=cost_model.commission,
        freq=freq,
        init_cash=init_cash,
    )

    n_trades = int(portfolio.trades.count())
    if n_trades == 0:
        return TrialResult(
            n_trades=0,
            total_return=None,
            sharpe_annualized=None,
            max_drawdown=None,
            profit_factor=None,
            win_rate=None,
            expectancy=None,
            note="sin operaciones",
        )

    sharpe = float(portfolio.sharpe_ratio())
    return TrialResult(
        n_trades=n_trades,
        total_return=float(portfolio.total_return()),
        sharpe_annualized=sharpe if math.isfinite(sharpe) else None,
        max_drawdown=float(portfolio.max_drawdown()),
        profit_factor=float(portfolio.trades.profit_factor()),
        win_rate=float(portfolio.trades.win_rate()),
        expectancy=float(portfolio.trades.expectancy()),
        note=None,
    )


def run_sweep(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    grid: Mapping[str, list[object]],
    cost_model: CostModel,
    registry: TrialRegistry,
    *,
    experiment_id: str,
    symbol: str,
    interval: str,
    freq: str,
    partition: Partition = Partition.DEVELOPMENT,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Ejecuta toda la rejilla y registra cada combinación, una a una.

    Por defecto opera solo sobre la partición de desarrollo
    (`fxlab.split.Partition.DEVELOPMENT`); acceder al holdout exige pasar
    `partition=Partition.HOLDOUT` explícitamente (ver `fxlab.split`).

    Cada combinación se registra inmediatamente al terminar de evaluarla
    (`TrialRegistry.record_trial` hace commit por fila), así que una
    interrupción a mitad de barrido no pierde ni corrompe lo ya registrado.

    Returns:
        Número de combinaciones evaluadas y registradas.
    """
    bid_partition = filter_partition(bid, partition)
    ask_partition = filter_partition(ask, partition)
    if not bid_partition.index.equals(ask_partition.index):
        raise ValueError("bid y ask deben compartir exactamente el mismo índice tras recortar")
    if bid_partition.empty:
        raise ValueError("la partición pedida no contiene datos")

    data_hash = hash_dataset(bid_partition, ask_partition)
    start_date = bid_partition.index.min()
    end_date = bid_partition.index.max()

    combos = list(iter_grid(grid))
    total = len(combos)

    for i, params in enumerate(combos, start=1):
        result = run_trial(bid_partition, ask_partition, params, cost_model, freq=freq)
        registry.record_trial(
            experiment_id=experiment_id,
            symbol=symbol,
            interval=interval,
            partition=partition.value,
            start_date=start_date,
            end_date=end_date,
            data_hash=data_hash,
            params=params.as_dict(),
            n_trades=result.n_trades,
            total_return=result.total_return,
            sharpe_annualized=result.sharpe_annualized,
            max_drawdown=result.max_drawdown,
            profit_factor=result.profit_factor,
            win_rate=result.win_rate,
            expectancy=result.expectancy,
            note=result.note,
        )
        if progress is not None:
            progress(i, total)

    return total
