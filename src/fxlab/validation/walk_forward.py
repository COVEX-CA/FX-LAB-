"""Walk-forward optimization: re-selecciona parámetros sobre ventanas de
entrenamiento sucesivas y mide su degradación fuera de esa ventana.

## Por qué no usa los splitters de VectorBT

VectorBT trae `vbt.RollingSplitter`, `vbt.ExpandingSplitter` y
`vbt.RangeSplitter`. Se inspeccionó su código fuente
(`inspect.getsource(vbt.RollingSplitter.split)` y
`inspect.getsource(vbt.ExpandingSplitter.split)`) antes de descartarlos:

- Los tres reparten el índice en `n` ventanas equiespaciadas (calculadas con
  `np.linspace`) o, alternativamente, en ventanas de `window_len` barras, y
  dividen cada ventana en entrenamiento/prueba mediante `set_lens`
  (fracciones, no recuentos de barras). Ninguno acepta `train_size`,
  `test_size` y `step` como recuentos de barras independientes y exactos,
  que es exactamente lo que pide esta fase (y que no puede tener un valor
  por defecto: cambia qué tan sensible es la optimización a cada tramo del
  histórico). Forzar esa forma de particionar dentro de la API fraccional
  de esos splitters sería más indirecto y menos verificable que un
  generador propio de ~30 líneas.

Por eso `_generate_folds` es una implementación propia. Es deliberadamente
simple: no reescala rangos, no reparte a partes iguales, solo desliza (o
expande) una ventana de tamaño fijo, un `step` de barras a la vez.

## Restricción a desarrollo

`run_walk_forward` recorta incondicionalmente a `fxlab.split.Partition.DEVELOPMENT`
antes de generar ningún pliegue — no existe ningún parámetro que permita
pedir el holdout desde este módulo.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from fxlab.split import Partition, filter_partition
from fxlab.sweep.costs import CostModel
from fxlab.sweep.engine import SweepParams, TrialResult, iter_grid, run_trial


class WindowMode(Enum):
    """Cómo crece la ventana de entrenamiento entre pliegues sucesivos."""

    ANCHORED = "anchored"
    """El inicio del entrenamiento queda fijo en la primera barra; la
    ventana de entrenamiento crece `step` barras en cada pliegue."""

    ROLLING = "rolling"
    """Ventana de entrenamiento de tamaño fijo (`train_size`); tanto el
    inicio como el final se desplazan `step` barras en cada pliegue."""


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _generate_folds(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step: int,
    mode: WindowMode,
) -> Iterator[Fold]:
    """Genera pliegues sucesivos sobre `index` (ya ordenado, sin huecos de
    posición: se indexa por posición entera, no por fecha).

    El bloque de prueba de cada pliegue empieza exactamente donde termina
    su propio bloque de entrenamiento (sin solape): la barra en la posición
    `train_end_pos` es la última de entrenamiento, y `train_end_pos + 1` es
    la primera de prueba. El siguiente pliegue no se genera si no caben
    `test_size` barras completas de prueba dentro de `index`.
    """
    n = len(index)
    train_end_pos = train_size - 1
    train_start_pos = 0

    while True:
        if mode is WindowMode.ROLLING:
            train_start_pos = train_end_pos - train_size + 1

        test_start_pos = train_end_pos + 1
        test_end_pos = test_start_pos + test_size - 1
        if test_end_pos >= n:
            return

        yield Fold(
            train_start=index[train_start_pos],
            train_end=index[train_end_pos],
            test_start=index[test_start_pos],
            test_end=index[test_end_pos],
        )

        train_end_pos += step


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold: Fold
    best_params: SweepParams | None
    train_result: TrialResult | None
    test_result: TrialResult | None
    sharpe_degradation_annualized: float | None
    """Sharpe de entrenamiento menos Sharpe de prueba de `best_params`
    (positivo = peor fuera de la ventana de entrenamiento). `None` si el
    pliegue no tiene una combinación ganadora evaluable.

    Anualizado: es una diferencia entre dos `TrialResult.sharpe_annualized`,
    ambos con el mismo `freq`, así que la resta es consistente. NO es
    comparable con el Sharpe sin anualizar de `fxlab.validation.deflated_sharpe`."""
    note: str | None
    """Motivo por el que el pliegue no produjo un resultado evaluable, p.ej.
    "ninguna combinación tuvo operaciones en entrenamiento"."""


@dataclass(frozen=True)
class WalkForwardResult:
    mode: WindowMode
    train_size: int
    test_size: int
    step: int
    folds: list[WalkForwardFoldResult]

    @property
    def mean_sharpe_degradation_annualized(self) -> float | None:
        values = [
            f.sharpe_degradation_annualized
            for f in self.folds
            if f.sharpe_degradation_annualized is not None
        ]
        return sum(values) / len(values) if values else None


def run_walk_forward(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    grid: Mapping[str, list[object]],
    cost_model: CostModel,
    *,
    train_size: int,
    test_size: int,
    step: int,
    mode: WindowMode,
    freq: str,
    selection_metric: str = "sharpe_annualized",
) -> WalkForwardResult:
    """Walk-forward optimization sobre la rejilla `grid`, restringido a desarrollo.

    Para cada pliegue: evalúa toda la rejilla sobre la ventana de
    entrenamiento, elige la combinación con mejor `selection_metric`
    (ignorando combinaciones cuya métrica sea `None`, p.ej. sin
    operaciones), y evalúa esa misma combinación sobre la ventana de
    prueba correspondiente. Los pliegues nunca se solapan entre
    entrenamiento y prueba (ver `_generate_folds`).

    Args:
        bid: OHLC bid. Se recorta a `fxlab.split.Partition.DEVELOPMENT`
            incondicionalmente antes de generar ningún pliegue.
        ask: OHLC ask, mismo índice que `bid` antes del recorte.
        grid: rejilla de parámetros, ver `fxlab.sweep.engine.iter_grid`.
        cost_model: modelo de costes, obligatorio (ver `fxlab.sweep.costs`).
        train_size: barras en la ventana de entrenamiento. Sin valor por
            defecto.
        test_size: barras en la ventana de prueba. Sin valor por defecto.
        step: barras que avanza la ventana entre pliegues sucesivos. Sin
            valor por defecto.
        mode: `WindowMode.ANCHORED` (entrenamiento crece desde el inicio)
            o `WindowMode.ROLLING` (ventana de tamaño fijo, se desliza).
        freq: frecuencia de barra para VectorBT, ver
            `fxlab.sweep.engine.run_trial`.
        selection_metric: campo de `TrialResult` usado para elegir la
            combinación ganadora en entrenamiento. Por defecto
            "sharpe_annualized":
            es el criterio canónico de este proyecto y coincide con el que
            usa el propio PBO (`fxlab.validation.pbo`) para su ranking
            in-sample/out-of-sample.

    Raises:
        ValueError: si `train_size`, `test_size` o `step` no son >= 1, si
            `selection_metric` no es un campo numérico válido de
            `TrialResult`, o si `bid`/`ask` no comparten índice.
    """
    if train_size < 1 or test_size < 1 or step < 1:
        raise ValueError(
            f"train_size, test_size y step deben ser >= 1, se recibió "
            f"train_size={train_size}, test_size={test_size}, step={step}"
        )
    valid_metrics = {
        "total_return",
        "sharpe_annualized",
        "profit_factor",
        "win_rate",
        "expectancy",
    }
    if selection_metric not in valid_metrics:
        raise ValueError(
            f"selection_metric debe ser una de {sorted(valid_metrics)}, "
            f"se recibió {selection_metric!r}"
        )

    bid_dev = filter_partition(bid, Partition.DEVELOPMENT)
    ask_dev = filter_partition(ask, Partition.DEVELOPMENT)
    if not bid_dev.index.equals(ask_dev.index):
        raise ValueError("bid y ask deben compartir exactamente el mismo índice tras recortar")

    combos = list(iter_grid(grid))
    fold_results: list[WalkForwardFoldResult] = []

    for fold in _generate_folds(bid_dev.index, train_size, test_size, step, mode):
        train_bid = bid_dev.loc[fold.train_start : fold.train_end]
        train_ask = ask_dev.loc[fold.train_start : fold.train_end]
        test_bid = bid_dev.loc[fold.test_start : fold.test_end]
        test_ask = ask_dev.loc[fold.test_start : fold.test_end]

        best_params: SweepParams | None = None
        best_train_result: TrialResult | None = None
        best_metric: float | None = None

        for params in combos:
            result = run_trial(train_bid, train_ask, params, cost_model, freq=freq)
            metric = getattr(result, selection_metric)
            if metric is None:
                continue
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_params = params
                best_train_result = result

        if best_params is None or best_train_result is None:
            fold_results.append(
                WalkForwardFoldResult(
                    fold=fold,
                    best_params=None,
                    train_result=None,
                    test_result=None,
                    sharpe_degradation_annualized=None,
                    note="ninguna combinación tuvo una métrica evaluable en entrenamiento",
                )
            )
            continue

        test_result = run_trial(test_bid, test_ask, best_params, cost_model, freq=freq)

        degradation: float | None = None
        note: str | None = None
        if (
            best_train_result.sharpe_annualized is not None
            and test_result.sharpe_annualized is not None
        ):
            degradation = best_train_result.sharpe_annualized - test_result.sharpe_annualized
        else:
            note = "la combinación ganadora no tuvo operaciones en prueba"

        fold_results.append(
            WalkForwardFoldResult(
                fold=fold,
                best_params=best_params,
                train_result=best_train_result,
                test_result=test_result,
                sharpe_degradation_annualized=degradation,
                note=note,
            )
        )

    return WalkForwardResult(
        mode=mode, train_size=train_size, test_size=test_size, step=step, folds=fold_results
    )
