"""Tests for the flight-delay EDA functions."""

from pathlib import Path

import pandas as pd

from eda import (
    airport_summary,
    load_data,
    missing_summary,
    quantitative_summary,
)


def test_load_data() -> None:
    """The loader should preserve rows and create analysis columns."""
    data = load_data()
    assert len(data) == 77
    assert "DELAY_OVER_30" in data.columns
    assert "TIME_OF_DAY" in data.columns
    assert data["DELAY_OVER_30"].dtype == bool


def test_missing_summary() -> None:
    """Missing counts should match direct pandas calculations."""
    data = load_data()
    summary = missing_summary(data)
    assert summary.loc["ARRIVAL_DELAY", "missing_count"] == 2
    assert summary["missing_count"].sum() == data.isna().sum().sum()


def test_quantitative_summary() -> None:
    """Summary values should agree with independently computed values."""
    data = load_data()
    summary = quantitative_summary(data)
    assert (
        summary.loc["DEPARTURE_DELAY", "min"]
        == data["DEPARTURE_DELAY"].min()
    )
    assert summary.loc["DISTANCE", "max"] == 3711


def test_airport_summary() -> None:
    """Airport grouping should account for every record exactly once."""
    data = load_data()
    summary = airport_summary(data)
    assert summary["flights"].sum() == len(data)
    assert set(summary.index) == {"SEA", "DFW", "ORD"}


def test_small_file() -> None:
    """The loader should also work on a controlled miniature CSV."""
    path = Path(__file__).parent / "test_small.csv"
    tiny = pd.DataFrame(
        {
            "YEAR": [2015, 2015],
            "MONTH": [1, 1],
            "DAY": [1, 1],
            "DAY_OF_WEEK": [4, 4],
            "AIRLINE": ["AA", "AA"],
            "ORIGIN_AIRPORT": ["SEA", "SEA"],
            "DESTINATION_AIRPORT": ["DFW", "DFW"],
            "SCHEDULED_DEPARTURE": [900, 1800],
            "DEPARTURE_DELAY": [10, 45],
            "DISTANCE": [1660, 1660],
            "ARRIVAL_DELAY": [5, 40],
            "CANCELLED": [0, 0],
            "CANCELLATION_REASON": ["Carrier Delay", "Weather Delay"],
            "IATA_Code": ["DFW", "DFW"],
            "Airport_Name": ["Dallas", "Dallas"],
            "City": ["Dallas", "Dallas"],
            "State": ["Texas", "Texas"],
        }
    )
    tiny.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded["DELAY_OVER_30"].tolist() == [False, True]
    path.unlink()
