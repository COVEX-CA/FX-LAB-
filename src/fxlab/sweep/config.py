"""Carga de la configuración de un barrido desde un fichero YAML.

El YAML es la única fuente de la comisión (`fxlab.sweep.costs.CostModel`) y
de la rejilla de parámetros — no hay valor por defecto oculto en el código
para ninguno de los dos: si falta `commission` en el fichero, `load_config`
falla en vez de asumir un número.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fxlab.sweep.costs import CostModel

_REQUIRED_KEYS = ("experiment_id", "symbol", "interval", "commission", "grid")

STRATEGIES = ("pullback", "ma_cross")
"""Estrategias que el barrido sabe ejecutar. Cada una tiene su propia rejilla
(campos distintos), así que un grid que no case con la estrategia elegida falla
al expandirlo — no hay forma de correr la estrategia equivocada en silencio."""

_DEFAULT_STRATEGY = "pullback"


@dataclass(frozen=True)
class SweepConfig:
    experiment_id: str
    symbol: str
    interval: str
    strategy: str
    cost_model: CostModel
    grid: dict[str, list[Any]]


def load_config(path: Path) -> SweepConfig:
    """Lee y valida un fichero YAML de configuración de barrido.

    Args:
        path: ruta al fichero YAML (ver `configs/` para un ejemplo).

    Returns:
        `SweepConfig` con la rejilla y el modelo de costes ya construidos.

    Raises:
        ValueError: si el YAML no es un mapeo, si falta alguna clave
            obligatoria — en particular `commission`, que nunca se asume —, o
            si `strategy` no es una de las conocidas (`STRATEGIES`).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: el YAML debe ser un mapeo en el nivel superior")

    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{path}: faltan claves obligatorias: {missing}")

    grid = raw["grid"]
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: 'grid' debe ser un mapeo de campo -> lista de valores")

    # `strategy` es opcional y por defecto 'pullback' (la estrategia original del
    # laboratorio). No es un "default silencioso" peligroso: si se olvida y el
    # grid es de otra estrategia, la expansión de la rejilla falla al no
    # encontrar sus campos, así que nunca se corre la estrategia equivocada sin
    # que salte un error.
    strategy = str(raw.get("strategy", _DEFAULT_STRATEGY))
    if strategy not in STRATEGIES:
        raise ValueError(
            f"{path}: strategy={strategy!r} no es válida; debe ser una de {list(STRATEGIES)}"
        )

    return SweepConfig(
        experiment_id=str(raw["experiment_id"]),
        symbol=str(raw["symbol"]),
        interval=str(raw["interval"]),
        strategy=strategy,
        cost_model=CostModel(commission=float(raw["commission"])),
        grid=dict(grid),
    )
