from __future__ import annotations

import pandas as pd
import pytest

from fxlab.data.resample import resample


def _m1_fixture() -> pd.DataFrame:
    """12 velas M1 desde las 10:00 hasta las 10:11 (UTC).

    Cubre dos ventanas M5 completas ([10:00,10:05) y [10:05,10:10)) y una
    tercera incompleta (solo 10:10 y 10:11, faltan datos hasta las 10:15).
    """
    index = pd.date_range("2024-01-01 10:00", periods=12, freq="1min", tz="UTC")
    n = len(index)
    return pd.DataFrame(
        {
            "open": [float(i) for i in range(n)],
            "high": [float(i) + 0.5 for i in range(n)],
            "low": [float(i) - 0.5 for i in range(n)],
            "close": [float(i) + 0.2 for i in range(n)],
            "volume": [100.0 for _ in range(n)],
        },
        index=index,
    )


def test_resample_ohlc_is_correct() -> None:
    df = _m1_fixture()
    result = resample(df, "M5")

    # Primera ventana [10:00, 10:05): minutos 0..4
    first = result.iloc[0]
    assert first["open"] == 0.0
    assert first["high"] == 4.5
    assert first["low"] == -0.5
    assert first["close"] == pytest.approx(4.2)
    assert first["volume"] == 500.0

    # Segunda ventana [10:05, 10:10): minutos 5..9
    second = result.iloc[1]
    assert second["open"] == 5.0
    assert second["high"] == 9.5
    assert second["low"] == 4.5
    assert second["close"] == pytest.approx(9.2)
    assert second["volume"] == 500.0


def test_bars_are_labeled_with_open_timestamp() -> None:
    df = _m1_fixture()
    result = resample(df, "M5")

    assert result.index[0] == pd.Timestamp("2024-01-01 10:00", tz="UTC")
    assert result.index[1] == pd.Timestamp("2024-01-01 10:05", tz="UTC")


def test_incomplete_trailing_bar_is_dropped() -> None:
    df = _m1_fixture()
    result = resample(df, "M5")

    # Solo llegan datos hasta las 10:11, la ventana [10:10, 10:15) está
    # incompleta y no debe aparecer en el resultado.
    assert len(result) == 2
    assert result.index[-1] == pd.Timestamp("2024-01-01 10:05", tz="UTC")


def test_complete_trailing_bar_is_kept() -> None:
    index = pd.date_range("2024-01-01 10:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    result = resample(df, "M5")

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-01-01 10:00", tz="UTC")
    assert result.iloc[0]["close"] == 5.0


def test_unknown_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="timeframe desconocido"):
        resample(_m1_fixture(), "M3")


def test_resample_output_satisfies_contract() -> None:
    from fxlab.data.validate import validate_ohlcv_contract

    result = resample(_m1_fixture(), "M5")
    validate_ohlcv_contract(result)
