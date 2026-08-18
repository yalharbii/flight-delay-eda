"""
Yousef Alharbi
CSE 163 - Summer 2026
Final Project

Tests the flight-delay project functions using the full dataset and a small,
controlled dataset with known results.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analysis import day_of_week_summary, route_summary
from eda import destination_summary, distance_summary, load_data
from eda import missing_summary, origin_summary, time_of_day_summary


def make_tiny_data(path: Path) -> pd.DataFrame:
    """Create and load a small controlled flight dataset."""
    tiny = pd.DataFrame({
        "date": [
            "2001-01-01T09:00:00",
            "2001-01-02T18:00:00",
            "2001-01-03T23:00:00"
        ],
        "delay": [10, 45, 80],
        "distance": [300, 900, 2000],
        "origin": ["SEA", "SEA", "LAX"],
        "destination": ["SFO", "SFO", "SEA"]
    })
    tiny.to_csv(path, index=False)
    return load_data(path)


def test_load_data() -> None:
    """Test the full dataset and engineered columns."""
    data = load_data()
    assert len(data) == 10000
    assert "delay_over_30" in data.columns
    assert "time_of_day" in data.columns
    assert "day_of_week" in data.columns
    assert data["delay_over_30"].dtype == bool


def test_missing_summary() -> None:
    """Test that missing counts match pandas calculations."""
    data = load_data()
    summary = missing_summary(data)
    assert summary["missing_count"].sum() == data.isna().sum().sum()


def test_feature_engineering() -> None:
    """Test engineered columns with known values."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.csv"
        data = make_tiny_data(path)
        assert data["delay_over_30"].tolist() == [False, True, True]
        assert data["hour"].tolist() == [9, 18, 23]
        assert data["time_of_day"].astype(str).tolist() == [
            "Morning", "Evening", "Late night"
        ]


def test_airport_summaries() -> None:
    """Test airport grouping with known values."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.csv"
        data = make_tiny_data(path)
        destinations = destination_summary(data, min_flights=1)
        origins = origin_summary(data, min_flights=1)
        assert destinations["flights"].sum() == 3
        assert origins["flights"].sum() == 3
        assert destinations.loc["SFO", "flights"] == 2


def test_time_and_distance_summaries() -> None:
    """Test time and distance summaries with known values."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.csv"
        data = make_tiny_data(path)
        assert time_of_day_summary(data)["flights"].sum() == 3
        assert distance_summary(data)["flights"].sum() == 3


def test_final_analysis_summaries() -> None:
    """Test day and route summaries with known values."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.csv"
        data = make_tiny_data(path)
        assert day_of_week_summary(data)["flights"].sum() == 3
        assert route_summary(data, min_flights=1)["flights"].sum() == 3


def main() -> None:
    """Run all project tests."""
    test_load_data()
    test_missing_summary()
    test_feature_engineering()
    test_airport_summaries()
    test_time_and_distance_summaries()
    test_final_analysis_summaries()
    print("All tests passed!")


if __name__ == "__main__":
    main()
