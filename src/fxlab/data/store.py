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

Cada fichero mensual se escribe de forma atómica (fichero temporal + rename):
si el proceso se interrumpe a mitad de escritura, el fichero final nunca
queda a medias. Al leer, los tipos de las columnas OHLCV se fuerzan
explícitamente a `float64` en vez de confiar en lo que devuelva pyarrow.

Cada fichero mensual lleva además una marca explícita de completitud
(`is_month_complete`), guardada en sus metadatos parquet junto al rango que
cubre. La marca la decide quien llama a `save`, no este módulo: `store.py`
no sabe si una descarga se interrumpió o si un mes tiene pocas barras porque
es el mes en curso o porque hubo festivos — eso es responsabilidad de quien
orquesta la descarga (`fxlab.data.loader`). `cached_ranges` solo informa de
los ficheros marcados como completos.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fxlab.data.types import OfferSide
from fxlab.data.validate import REQUIRED_COLUMNS, validate_ohlcv_contract

DEFAULT_DATA_DIR = Path("data")

_META_START_KEY = b"fxlab_start"
_META_END_KEY = b"fxlab_end"
_META_COMPLETE_KEY = b"fxlab_complete"
_COMPLETE_TRUE = b"true"
_COMPLETE_FALSE = b"false"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def _partition_dir(base_path: Path, symbol: str, interval: str, offer_side: OfferSide) -> Path:
    return base_path / _safe_symbol(symbol) / interval / offer_side.value


def _month_file_path(
    base_path: Path, symbol: str, interval: str, offer_side: OfferSide, year: int, month: int
) -> Path:
    return (
        _partition_dir(base_path, symbol, interval, offer_side) / f"{year:04d}-{month:02d}.parquet"
    )


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    """Escribe `table` en `path` de forma atómica.

    Escribe primero a un fichero temporal en el mismo directorio y solo
    renombra a `path` (operación atómica en el mismo sistema de ficheros) si
    la escritura terminó con éxito. Si el proceso se interrumpe a mitad de
    escritura, el temporal se borra y `path` nunca llega a existir a medias:
    o no hay fichero (el mes se volverá a descargar en el próximo intento),
    o hay uno completo. Sin esto, un mes cuya descarga se interrumpiera
    podría quedar cacheado con datos incompletos sin que nada lo detectase.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")
    df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return df


def save(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    *,
    complete: bool,
    base_path: Path = DEFAULT_DATA_DIR,
) -> list[Path]:
    """Guarda `df` en la caché parquet, partido en un fichero por mes natural.

    Args:
        df: DataFrame que cumple el contrato de datos OHLCV.
        symbol: instrumento, p.ej. "EUR/USD".
        interval: intervalo de las velas, p.ej. "1MIN".
        offer_side: `OfferSide.BID` o `OfferSide.ASK`.
        complete: si el mes que se está guardando debe marcarse como
            completo (ver `is_month_complete`). Sin valor por defecto a
            propósito: quien llama debe decidir explícitamente si los datos
            que tiene representan el mes entero (descarga terminada sin
            error, no es el mes en curso) o son parciales.
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
            base_path,
            symbol,
            interval,
            offer_side,
            int(year),  # type: ignore[call-overload]
            int(month),  # type: ignore[call-overload]
        )
        table = pa.Table.from_pandas(month_df, preserve_index=True)
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata(
            {
                **existing_meta,
                _META_START_KEY: month_df.index.min().isoformat().encode(),
                _META_END_KEY: month_df.index.max().isoformat().encode(),
                _META_COMPLETE_KEY: _COMPLETE_TRUE if complete else _COMPLETE_FALSE,
            }
        )
        _write_table_atomic(table, path)
        written.append(path)

    return written


def is_month_complete(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    year: int,
    month: int,
    base_path: Path = DEFAULT_DATA_DIR,
) -> bool:
    """Indica si (year, month) está en caché y marcado como completo.

    Esta es la única pregunta que decide si hay que descargar un mes: no se
    infiere a partir de fechas ni de cuántas barras tiene el fichero. Un mes
    sin fichero, o cuyo fichero no fue marcado como completo por `save`
    (por ejemplo, porque es el mes en curso o porque la descarga se
    interrumpió), devuelve `False`.
    """
    path = _month_file_path(base_path, symbol, interval, offer_side, year, month)
    if not path.exists():
        return False
    metadata = pq.read_metadata(path).metadata or {}
    return metadata.get(_META_COMPLETE_KEY) == _COMPLETE_TRUE


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

    # Se fuerza el tipo explícitamente en vez de confiar en lo que devuelva
    # pyarrow: un fichero parquet ajeno o mal escrito no debe colar tipos
    # distintos de float64 silenciosamente.
    for col in REQUIRED_COLUMNS:
        if col in result.columns:
            result[col] = result[col].astype("float64")

    # Se valida el contrato de lo leído de disco ANTES de recortarlo al
    # rango pedido: así, si un fichero cacheado está corrupto o mal escrito
    # (índice naive, duplicados, NaN en OHLC...), se detecta aquí con un
    # error claro en vez de fallar más abajo al comparar timestamps
    # incompatibles al recortar el rango.
    validate_ohlcv_contract(result)

    result = result[(result.index >= start) & (result.index < end)]
    return result


def cached_ranges(
    symbol: str,
    interval: str,
    offer_side: OfferSide,
    base_path: Path = DEFAULT_DATA_DIR,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Informa de qué rangos de tiempo ya están cacheados y completos.

    Solo se consideran los ficheros mensuales marcados como completos por
    `save` (ver `is_month_complete`): un mes a medio descargar, o el mes en
    curso, tiene datos en disco pero no se reporta aquí como cacheado. Esta
    función es puramente informativa (para inspección manual); la decisión
    de si hace falta descargar un mes concreto se toma con
    `is_month_complete`, no con esto.

    Dentro de los meses completos, los rangos se derivan de los metadatos
    (min/max real de cada fichero), sin leer los datos, y los meses
    consecutivos se fusionan en un mismo rango si el hueco entre ellos es
    menor a 3 días (cubre fines de semana e intervalos diarios/4h) — esto es
    solo para que el informe salga como rangos continuos legibles, no afecta
    a qué se considera cacheado.

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
        if metadata.get(_META_COMPLETE_KEY) != _COMPLETE_TRUE:
            continue
        if _META_START_KEY not in metadata or _META_END_KEY not in metadata:
            continue
        chunk_start = pd.Timestamp(metadata[_META_START_KEY].decode())
        chunk_end = pd.Timestamp(metadata[_META_END_KEY].decode())
        bounds.append((chunk_start, chunk_end))

    bounds.sort(key=lambda pair: pair[0])

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    report_merge_gap = pd.Timedelta(days=3)
    for chunk_start, chunk_end in bounds:
        if merged and chunk_start - merged[-1][1] <= report_merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], chunk_end))
        else:
            merged.append((chunk_start, chunk_end))

    return merged


def _parse_month_filename(path: Path) -> tuple[int, int]:
    year_str, month_str = path.stem.split("-")
    return int(year_str), int(month_str)
