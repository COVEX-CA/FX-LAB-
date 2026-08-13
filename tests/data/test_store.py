from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fxlab.data import store as store_module
from fxlab.data.store import cached_ranges, load, save
from fxlab.data.types import OfferSide
from fxlab.data.validate import DataContractError


def _daily_df(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    n = len(index)
    return pd.DataFrame(
        {
            "open": [float(i) for i in range(n)],
            "high": [float(i) + 0.5 for i in range(n)],
            "low": [float(i) - 0.5 for i in range(n)],
            "close": [float(i) + 0.2 for i in range(n)],
            "volume": [10.0 for _ in range(n)],
        },
        index=index,
    )


def test_save_creates_one_file_per_month(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)  # cruza de enero a febrero 2024
    written = save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    assert len(written) == 2
    directory = tmp_path / "EUR_USD" / "1DAY" / "bid"
    assert (directory / "2024-01.parquet").exists()
    assert (directory / "2024-02.parquet").exists()


def test_save_load_roundtrip_is_lossless(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    loaded = load(
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(days=1),
        base_path=tmp_path,
    )

    # El roundtrip por parquet no preserva el atributo `freq` del índice,
    # solo los valores y su orden: eso es lo que importa para "sin pérdida".
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_load_filters_to_requested_range(tmp_path: Path) -> None:
    df = _daily_df("2024-01-01", periods=20)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    start = pd.Timestamp("2024-01-10", tz="UTC")
    end = pd.Timestamp("2024-01-15", tz="UTC")
    loaded = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)

    assert loaded.index.min() == start
    assert loaded.index.max() == pd.Timestamp("2024-01-14", tz="UTC")
    assert (loaded.index >= start).all()
    assert (loaded.index < end).all()


def test_load_missing_symbol_returns_empty_frame(tmp_path: Path) -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-02-01", tz="UTC")
    loaded = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)

    assert loaded.empty
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_bid_and_ask_are_stored_separately(tmp_path: Path) -> None:
    bid_df = _daily_df("2024-01-01", periods=5)
    ask_df = bid_df.copy()
    ask_df["open"] += 0.001

    save(bid_df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)
    save(ask_df, "EUR/USD", "1DAY", OfferSide.ASK, base_path=tmp_path)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-06", tz="UTC")
    loaded_bid = load("EUR/USD", "1DAY", OfferSide.BID, start=start, end=end, base_path=tmp_path)
    loaded_ask = load("EUR/USD", "1DAY", OfferSide.ASK, start=start, end=end, base_path=tmp_path)

    assert not loaded_bid["open"].equals(loaded_ask["open"])


def test_cached_ranges_reports_saved_data(tmp_path: Path) -> None:
    df = _daily_df("2024-01-25", periods=10)
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    ranges = cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    assert len(ranges) == 1
    start, end = ranges[0]
    assert start == df.index.min()
    assert end == df.index.max()


def test_cached_ranges_empty_when_nothing_saved(tmp_path: Path) -> None:
    assert cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path) == []


def _write_raw_parquet(
    df: pd.DataFrame, directory: Path, filename: str, *, with_range_metadata: bool = True
) -> None:
    """Escribe un parquet directamente con pyarrow, sin pasar por `save()`.

    Sirve para simular ficheros de caché ajenos o mal escritos (contrato
    violado, tipos incorrectos) que `save()` nunca dejaría pasar porque
    valida antes de escribir.
    """
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    if with_range_metadata:
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata(
            {
                **existing_meta,
                b"fxlab_start": df.index.min().isoformat().encode(),
                b"fxlab_end": df.index.max().isoformat().encode(),
            }
        )
    pq.write_table(table, directory / filename, compression="snappy")


# --- 2.1: el contrato se valida también al leer, no solo al descargar -----


def test_load_raises_on_corrupted_cached_file(tmp_path: Path) -> None:
    directory = tmp_path / "EUR_USD" / "1DAY" / "bid"
    naive_index = pd.date_range("2024-01-01", periods=3, freq="1D")  # sin tz: viola el contrato
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=naive_index,
    )
    _write_raw_parquet(df, directory, "2024-01.parquet", with_range_metadata=False)

    with pytest.raises(DataContractError):
        load(
            "EUR/USD",
            "1DAY",
            OfferSide.BID,
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-05", tz="UTC"),
            base_path=tmp_path,
        )


# --- 2.2: los tipos float64 se fuerzan al leer, no solo se comprueban -----


def test_load_forces_float64_even_if_file_has_other_dtype(tmp_path: Path) -> None:
    directory = tmp_path / "EUR_USD" / "1DAY" / "bid"
    index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {
            "open": pd.array([1, 2, 3], dtype="int64"),
            "high": pd.array([1, 2, 3], dtype="int64"),
            "low": pd.array([1, 2, 3], dtype="int64"),
            "close": pd.array([1, 2, 3], dtype="int64"),
            "volume": pd.array([10, 20, 30], dtype="int64"),
        },
        index=index,
    )
    assert df["open"].dtype == "int64"  # de partida NO es float64
    _write_raw_parquet(df, directory, "2024-01.parquet")

    loaded = load(
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-05", tz="UTC"),
        base_path=tmp_path,
    )

    for col in ("open", "high", "low", "close", "volume"):
        assert loaded[col].dtype == "float64"


# --- 2.3: escritura atómica: un mes a medias nunca se reporta como cacheado


def test_interrupted_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _daily_df("2024-01-01", periods=10)

    def flaky_write_table(table: pa.Table, where: object, **kwargs: object) -> None:
        # Simula un proceso que muere a mitad de la escritura: deja basura
        # en el fichero (el temporal, no el final) y nunca termina.
        Path(str(where)).write_bytes(b"esto no es un parquet valido")
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(store_module.pq, "write_table", flaky_write_table)

    with pytest.raises(OSError):
        save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    directory = tmp_path / "EUR_USD" / "1DAY" / "bid"
    # ni el fichero final ni restos temporales deben quedar en disco
    assert list(directory.glob("*.parquet")) == []
    assert list(directory.glob("*.tmp")) == []

    # y por tanto el mes no se reporta como cacheado
    assert cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path) == []
    loaded = load(
        "EUR/USD",
        "1DAY",
        OfferSide.BID,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        base_path=tmp_path,
    )
    assert loaded.empty


def test_successful_write_is_visible_after_interrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tras un fallo simulado, un reintento normal sí deja el mes cacheado."""
    df = _daily_df("2024-01-01", periods=10)

    real_write_table = pq.write_table
    call_count = {"n": 0}

    def flaky_then_ok(table: pa.Table, where: object, **kwargs: object) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            Path(str(where)).write_bytes(b"esto no es un parquet valido")
            raise OSError("simulated crash mid-write")
        real_write_table(table, where, **kwargs)

    monkeypatch.setattr(store_module.pq, "write_table", flaky_then_ok)

    with pytest.raises(OSError):
        save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    # reintento: esta vez la escritura se completa
    save(df, "EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)

    ranges = cached_ranges("EUR/USD", "1DAY", OfferSide.BID, base_path=tmp_path)
    assert len(ranges) == 1
