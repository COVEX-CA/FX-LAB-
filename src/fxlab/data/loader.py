"""Orquestación de caché y descarga: punto de entrada único para obtener datos.

Responsabilidad de este módulo: decidir, mes natural a mes natural, qué
datos ya están cacheados y completos y cuáles hay que descargar —
combinando `fxlab.data.download` y `fxlab.data.store` sin que esos dos
módulos se conozcan entre sí. Cualquier código que necesite datos OHLCV (el
barrido de parámetros en fases futuras, scripts, notebooks) debe pasar por
`load_range` en vez de reimplementar esta lógica; así solo hay una versión
del comportamiento de caché.

Un mes se descarga (o redescarga) si y solo si `fxlab.data.store` no lo
tiene marcado como completo (`is_month_complete`). No hay ningún margen ni
comparación de fechas en esa decisión: la completitud es una marca binaria
que `save` escribe únicamente cuando la descarga de ese mes ha terminado
sin error. El mes en curso nunca se marca completo (ver `_is_open_month`),
así que siempre se vuelve a pedir; eso es intencional, no una limitación.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fxlab.data.download import download
from fxlab.data.store import DEFAULT_DATA_DIR, is_month_complete, load, save
from fxlab.data.types import OfferSide
from fxlab.data.validate import validate_ohlcv_contract

logger = logging.getLogger(__name__)


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _full_calendar_months(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Meses naturales completos (sin recortar) que solapan [start, end).

    A diferencia de `fxlab.data.download.month_chunks` (que recorta el
    primer y último tramo a `start`/`end` para no descargar de más de lo
    pedido), aquí siempre se devuelve el mes natural entero: la unidad de
    caché es el mes completo, así que un mes se descarga entero (o hasta
    "ahora" si es el mes en curso) independientemente de qué sub-rango pida
    quien llama. Es lo que hace que la marca de completitud tenga el mismo
    significado sin importar qué fechas exactas se pidieron.
    """
    if end <= start:
        return

    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1)
        yield cursor, next_month
        cursor = next_month


def _is_open_month(month_start: datetime, month_end: datetime, now: datetime) -> bool:
    """Un mes está "abierto" si aún no ha terminado en el momento `now`.

    Cubre tanto el mes en curso como cualquier mes futuro (si alguien pide
    un rango que se adelanta al presente). Un mes abierto nunca puede
    marcarse como completo, porque todavía le pueden llegar más datos.
    """
    return month_end > now


def load_range(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    offer_side: OfferSide,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Obtiene velas OHLCV de un símbolo y rango, descargando solo lo que falte.

    Por cada mes natural que solapa el rango pedido (ver
    `_full_calendar_months`):

    1. si `fxlab.data.store.is_month_complete` dice que ya está completo,
       se lee de disco (`fxlab.data.store.load`) sin tocar la red;
    2. si no (no hay fichero, no está marcado completo, o es el mes en
       curso), se descarga el mes entero —o hasta `now` si está en
       curso— (`fxlab.data.download.download`) y, si la descarga termina
       sin error, se guarda marcado como completo solo cuando el mes ya ha
       terminado (`fxlab.data.store.save`).

    El resultado combinado se recorta al rango exacto pedido, se valida
    contra el contrato de datos y se devuelve.

    Args:
        symbol: instrumento, p.ej. "EUR/USD".
        start: inicio del rango (inclusive). Si es naive se asume UTC.
        end: fin del rango (exclusive). Si es naive se asume UTC.
        interval: intervalo nativo de Dukascopy (constantes `INTERVAL_*` de
            `dukascopy_python`).
        offer_side: `OfferSide.BID` o `OfferSide.ASK`.
        data_dir: raíz de la caché en disco.
        now: instante considerado "el presente", usado únicamente para
            decidir si un mes está en curso (ver `_is_open_month`).
            Inyectable para tests; por defecto la hora actual en UTC.

    Returns:
        DataFrame que cumple el contrato de datos OHLCV.
    """
    start = _ensure_utc(start)
    end = _ensure_utc(end)
    now = _ensure_utc(now) if now is not None else datetime.now(UTC)

    frames = []
    for month_start, month_end in _full_calendar_months(start, end):
        year, month = month_start.year, month_start.month
        open_month = _is_open_month(month_start, month_end, now)

        if not open_month and is_month_complete(
            symbol, interval, offer_side, year, month, data_dir
        ):
            logger.info("mes %04d-%02d completo en caché, leyendo de disco", year, month)
            month_df = load(
                symbol,
                interval,
                offer_side,
                pd.Timestamp(month_start),
                pd.Timestamp(month_end),
                data_dir,
            )
        else:
            fetch_end = min(month_end, now) if open_month else month_end
            logger.info(
                "mes %04d-%02d no completo en caché, descargando %s a %s",
                year,
                month,
                month_start,
                fetch_end,
            )
            month_df = download(symbol, month_start, fetch_end, interval, offer_side)
            if not month_df.empty:
                save(
                    month_df,
                    symbol,
                    interval,
                    offer_side,
                    complete=not open_month,
                    base_path=data_dir,
                )

        if not month_df.empty:
            frames.append(month_df)

    if not frames:
        result = load(
            symbol, interval, offer_side, pd.Timestamp(start), pd.Timestamp(end), data_dir
        )
    else:
        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        result = result[(result.index >= start) & (result.index < end)]

    validate_ohlcv_contract(result)
    return result
