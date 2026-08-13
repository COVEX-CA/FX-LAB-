"""Caché en parquet de velas OHLCV.

Responsabilidad única de este módulo: persistir y releer en disco. No sabe
descargar de Dukascopy (`fxlab.data.download`) ni resamplear
(`fxlab.data.resample`).

Los datos se particionan en `<base_path>/<symbol>/<interval>/<offer_side>/`,
un fichero parquet por mes natural (comprimido con snappy). Particionar por
`offer_side` además de por símbolo e intervalo es necesario: bid y ask son
series distintas y mezclarlas en un mismo fichero impediría separarlas de
nuevo. El rango cubierto por cada fichero se guarda en sus metadatos parquet
para poder listar rangos cacheados sin leer los datos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fxlab.data.types import OfferSide
from fxlab.data.validate import validate_ohlcv_contract

DEFAULT_DATA_DIR = Path("data")

_META_START_KEY = b"fxlab_start"
_META_END_KEY = b"fxlab_end"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def _partition_dir(base_path: Path, symbol: str, interval: str, offer_side: OfferSide) -> Path:
    return base_path / _safe_symbol(symbol) / interval / offer_side.value


def _month_file_path(
    base_path: Path, symbol: str, interval: str, offer_side: OfferSide, year: int, month: int
) -> Path:
    return _partition_dir(base_path, symbol, interval, offer_side) / f"{year:04d}-{month:02d}.parquet"


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")
    df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return df


def save(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    base_path: Path = DEFAULT_DATA_DIR,
) -> list[Path]:
    """Guarda `df` en la caché parquet, partido en un fichero por mes natural.

    Args:
        df: DataFrame que cumple el contrato de datos OHLCV.
        symbol: instrumento, p.ej. "EUR/USD".
        interval: intervalo de las velas, p.ej. "1MIN".
        offer_side: `OfferSide.BID` o `OfferSide.ASK`.
        base_path: raíz de la caché en disco.

    Returns:
        Rutas de los ficheros parquet escritos.
    """
    validate_ohlcv_contract(df)

    directory = _partition_dir(base_path, symbol, interval, offer_side)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if df.empty:
        return written

    index = df.index
    assert isinstance(index, pd.DatetimeIndex)
    for (year, month), month_df in df.groupby([index.year, index.month]):
        path = _month_file_path(
            base_path, symbol, interval, offer_side, int(year), int(month)  # type: ignore[call-overload]
        )
        table = pa.Table.from_pandas(month_df, preserve_index=True)
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata(
            {
                **existing_meta,
                _META_START_KEY: month_df.index.min().isoformat().encode(),
                _META_END_KEY: month_df.index.max().isoformat().encode(),
            }
        )
        pq.write_table(table, path, compression="snappy")
        written.append(path)

    return written


def load(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_path: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Lee de la caché parquet el rango [start, end) ya descargado.

    Idempotente respecto a `download`: si el rango pedido ya está en disco,
    esta función no hace ninguna petición de red, solo lee ficheros locales.

    Args:
        symbol: instrumento, p.ej. "EUR/USD".
        interval: intervalo de las velas, p.ej. "1MIN".
        offer_side: `OfferSide.BID` o `OfferSide.ASK`.
        start: inicio del rango (inclusive), timezone-aware.
        end: fin del rango (exclusive), timezone-aware.
        base_path: raíz de la caché en disco.

    Returns:
        DataFrame que cumple el contrato de datos OHLCV, recortado a
        [start, end). Vacío si no hay nada cacheado en ese rango.
    """
    directory = _partition_dir(base_path, symbol, interval, offer_side)
    if not directory.exists():
        result = _empty_frame()
        validate_ohlcv_contract(result)
        return result

    frames = []
    for path in sorted(directory.glob("*.parquet")):
        year, month = _parse_month_filename(path)
        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        month_end = month_start + pd.DateOffset(months=1)
        if month_end <= start or month_start >= end:
            continue
        frames.append(pd.read_parquet(path))

    if not frames:
        result = _empty_frame()
        validate_ohlcv_contract(result)
        return result

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result[(result.index >= start) & (result.index < end)]
    validate_ohlcv_contract(result)
    return result


def cached_ranges(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    base_path: Path = DEFAULT_DATA_DIR,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Informa de qué rangos de tiempo ya están cacheados para un símbolo.

    Los rangos se derivan de los metadatos (min/max real de cada fichero
    mensual), sin leer los datos. Meses consecutivos se fusionan en un mismo
    rango si el hueco entre ellos es menor a 3 días (cubre fines de semana e
    intervalos diarios/4h).

    Returns:
        Lista de tuplas (inicio, fin) ordenadas, con timestamps tal como se
        guardaron (timezone-aware).
    """
    directory = _partition_dir(base_path, symbol, interval, offer_side)
    if not directory.exists():
        return []

    bounds = []
    for path in sorted(directory.glob("*.parquet")):
        metadata = pq.read_metadata(path).metadata or {}
        if _META_START_KEY not in metadata or _META_END_KEY not in metadata:
            continue
        chunk_start = pd.Timestamp(metadata[_META_START_KEY].decode())
        chunk_end = pd.Timestamp(metadata[_META_END_KEY].decode())
        bounds.append((chunk_start, chunk_end))

    bounds.sort(key=lambda pair: pair[0])

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    gap_tolerance = pd.Timedelta(days=3)
    for chunk_start, chunk_end in bounds:
        if merged and chunk_start - merged[-1][1] <= gap_tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], chunk_end))
        else:
            merged.append((chunk_start, chunk_end))

    return merged


def _parse_month_filename(path: Path) -> tuple[int, int]:
    year_str, month_str = path.stem.split("-")
    return int(year_str), int(month_str)
