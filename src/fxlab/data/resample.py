"""Resampleo de velas OHLCV a timeframes mayores.

Responsabilidad única de este módulo: agregar una serie ya cargada
(descargada o leída de caché) a un timeframe mayor. No descarga ni persiste.
"""

from __future__ import annotations

import pandas as pd

from fxlab.data.validate import validate_ohlcv_contract

TIMEFRAMES = {
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _infer_base_period(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Estima la duración de una barra de origen a partir de la mediana de
    los huecos entre timestamps consecutivos. Se usa para decidir si la
    última barra resampleada llega a cerrarse o hay que descartarla.
    """
    if len(index) < 2:
        return pd.Timedelta(0)
    diffs = index.to_series().diff().dropna()
    return pd.Timedelta(diffs.median())


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reagrupa un DataFrame OHLCV de un timeframe menor a uno mayor.

    Convención de etiquetado de barras (decisión explícita del proyecto):
    las barras se etiquetan con el timestamp de APERTURA y el intervalo se
    cierra a la izquierda (`label="left"`, `closed="left"`). Es decir, la
    barra etiquetada como "10:00" en M5 agrega los datos de origen del rango
    [10:00, 10:05) — el propio timestamp de apertura incluido, el de cierre
    excluido. Esto evita cualquier lookahead: la barra "10:00" nunca contiene
    información posterior a las 10:05.

    La barra final se descarta si los datos de origen no llegan a cubrir su
    cierre (barra incompleta al final de la serie), para no reportar una
    vela cuyo high/low/close puedan cambiar si llegan más datos.

    Args:
        df: DataFrame que cumple el contrato de datos OHLCV, en un timeframe
            menor o igual al destino.
        timeframe: uno de "M5", "M15", "H1", "H4", "D1".

    Returns:
        DataFrame resampleado que cumple el contrato de datos OHLCV.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe desconocido: {timeframe!r}, debe ser uno de {list(TIMEFRAMES)}")

    validate_ohlcv_contract(df)

    if df.empty:
        return df.copy()

    index = df.index
    assert isinstance(index, pd.DatetimeIndex)

    rule = TIMEFRAMES[timeframe]
    resampled = df.resample(rule, label="left", closed="left").agg(_AGG)  # type: ignore[arg-type]
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    if not resampled.empty:
        base_period = _infer_base_period(index)
        last_bar_start = resampled.index[-1]
        last_bar_end = last_bar_start + pd.tseries.frequencies.to_offset(rule)
        if index.max() < last_bar_end - base_period:
            resampled = resampled.iloc[:-1]

    validate_ohlcv_contract(resampled)
    return resampled
