from __future__ import annotations

import pandas as pd
import pytest

from fxlab.data.validate import DataContractError, validate_ohlcv_contract


def _valid_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [10.0, 20.0, 30.0],
        },
        index=index,
    )


def test_valid_contract_does_not_raise() -> None:
    validate_ohlcv_contract(_valid_df())


def test_unsorted_index_raises() -> None:
    df = _valid_df().iloc[::-1]
    with pytest.raises(DataContractError, match="ordenado"):
        validate_ohlcv_contract(df)


def test_naive_timestamps_raise() -> None:
    df = _valid_df()
    df.index = df.index.tz_localize(None)
    with pytest.raises(DataContractError, match="timezone-aware"):
        validate_ohlcv_contract(df)


def test_non_utc_timezone_raises() -> None:
    df = _valid_df()
    df.index = df.index.tz_convert("America/New_York")
    with pytest.raises(DataContractError, match="UTC"):
        validate_ohlcv_contract(df)


def test_duplicated_index_raises() -> None:
    df = _valid_df()
    df.index = pd.DatetimeIndex([df.index[0], df.index[0], df.index[1]], tz="UTC")
    with pytest.raises(DataContractError, match="duplicados"):
        validate_ohlcv_contract(df)


def test_nan_in_ohlc_raises() -> None:
    df = _valid_df()
    df.loc[df.index[1], "close"] = float("nan")
    with pytest.raises(DataContractError, match="NaN"):
        validate_ohlcv_contract(df)


def test_missing_column_raises() -> None:
    df = _valid_df().drop(columns=["volume"])
    with pytest.raises(DataContractError, match="faltan columnas"):
        validate_ohlcv_contract(df)


def test_wrong_dtype_raises() -> None:
    df = _valid_df()
    df["open"] = df["open"].astype("float32")
    with pytest.raises(DataContractError, match="float64"):
        validate_ohlcv_contract(df)
