"""Validación del contrato de datos OHLCV.

Contrato exigido a cualquier DataFrame que salga de `fxlab.data`:

- Índice `DatetimeIndex` en UTC, timezone-aware, ordenado y sin duplicados.
- Columnas `open`, `high`, `low`, `close`, `volume`, todas `float64`, en minúsculas.
- Sin `NaN` en las columnas OHLC.

Se llama al final de cada operación de carga (descarga, lectura de caché,
resampleo). Falla ruidosamente: un `DataContractError` es preferible a datos
silenciosamente corruptos.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
OHLC_COLUMNS = ("open", "high", "low", "close")


class DataContractError(ValueError):
    """Se lanza cuando un DataFrame no cumple el contrato de datos OHLCV."""


def validate_ohlcv_contract(df: pd.DataFrame) -> None:
    """Comprueba que `df` cumple el contrato de datos OHLCV.

    Args:
        df: DataFrame a validar.

    Raises:
        DataContractError: si el índice no es un `DatetimeIndex` UTC
            timezone-aware, ordenado y sin duplicados; si faltan columnas
            requeridas; si alguna columna requerida no es `float64`; o si hay
            `NaN` en las columnas OHLC.
    """
    violations: list[str] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        violations.append(f"el índice debe ser DatetimeIndex, es {type(df.index).__name__}")
    else:
        if df.index.tz is None:
            violations.append("el índice debe ser timezone-aware (naive encontrado)")
        elif str(df.index.tz) != "UTC":
            violations.append(f"el índice debe estar en UTC, está en {df.index.tz}")

        if not df.index.is_monotonic_increasing:
            violations.append("el índice debe estar ordenado ascendentemente")

        if df.index.has_duplicates:
            n_dup = int(df.index.duplicated().sum())
            violations.append(f"el índice tiene {n_dup} timestamps duplicados")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        violations.append(f"faltan columnas requeridas: {missing}")

    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].dtype != "float64":
            violations.append(f"la columna '{col}' debe ser float64, es {df[col].dtype}")

    for col in OHLC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            n_nan = int(df[col].isna().sum())
            violations.append(f"la columna '{col}' tiene {n_nan} valores NaN")

    if violations:
        detail = "; ".join(violations)
        raise DataContractError(f"contrato de datos OHLCV violado: {detail}")
