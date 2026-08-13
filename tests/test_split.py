from __future__ import annotations

import logging

import pandas as pd
import pytest

from fxlab.split import HOLDOUT_START, Partition, filter_partition


def _straddling_df() -> pd.DataFrame:
    index = pd.date_range("2019-12-01", "2020-02-01", freq="1D", tz="UTC")
    return pd.DataFrame({"close": range(len(index))}, index=index)


def test_default_partition_is_development() -> None:
    df = _straddling_df()
    result = filter_partition(df)

    assert (result.index < HOLDOUT_START).all()
    assert not result.empty


def test_development_includes_last_bar_of_2019() -> None:
    df = _straddling_df()
    result = filter_partition(df, partition=Partition.DEVELOPMENT)

    assert pd.Timestamp("2019-12-31", tz="UTC") in result.index
    assert pd.Timestamp("2020-01-01", tz="UTC") not in result.index


def test_holdout_requires_explicit_argument_and_only_returns_2020_onward() -> None:
    df = _straddling_df()
    result = filter_partition(df, partition=Partition.HOLDOUT)

    assert (result.index >= HOLDOUT_START).all()
    assert not result.empty
    assert pd.Timestamp("2019-12-31", tz="UTC") not in result.index


def test_holdout_access_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    df = _straddling_df()
    with caplog.at_level(logging.WARNING, logger="fxlab.split"):
        filter_partition(df, partition=Partition.HOLDOUT)

    assert any("holdout" in record.message.lower() for record in caplog.records)


def test_development_access_does_not_log_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    df = _straddling_df()
    with caplog.at_level(logging.WARNING, logger="fxlab.split"):
        filter_partition(df)

    assert caplog.records == []
