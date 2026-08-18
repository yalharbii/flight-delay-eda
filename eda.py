"""
Yousef Alharbi
CSE 163 - Summer 2026
Final Project

Loads the 10,000-row U.S. flight-delay dataset, creates useful analysis
columns, summarizes the data, and generates exploratory figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).parent / "flight_sample.csv"
FIGURE_DIR = Path(__file__).parent / "figures"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the flight data and create columns used in the analysis."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    data["delay_over_30"] = data["delay"] > 30
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.day_name()
    data["month"] = data["date"].dt.month
    data["time_of_day"] = pd.cut(
        data["hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["Overnight", "Morning", "Afternoon", "Evening", "Late night"]
    )
    return data


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for each column."""
    counts = data.isna().sum()
    result = pd.DataFrame({"missing_count": counts})
    result["missing_percent"] = counts / len(data) * 100
    return result


def quantitative_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for delay and distance."""
    summary = data[["delay", "distance"]].describe().T
    columns = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    return summary[columns]


def destination_summary(data: pd.DataFrame,
                        min_flights: int = 20) -> pd.DataFrame:
    """Summarize delay outcomes by destination airport."""
    grouped = data.groupby("destination").agg(
        flights=("destination", "size"),
        mean_delay=("delay", "mean"),
        median_delay=("delay", "median"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )
    grouped = grouped[grouped["flights"] >= min_flights]
    return grouped.sort_values("mean_delay", ascending=False)


def origin_summary(data: pd.DataFrame,
                   min_flights: int = 20) -> pd.DataFrame:
    """Summarize delay outcomes by origin airport."""
    grouped = data.groupby("origin").agg(
        flights=("origin", "size"),
        mean_delay=("delay", "mean"),
        median_delay=("delay", "median"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )
    grouped = grouped[grouped["flights"] >= min_flights]
    return grouped.sort_values("mean_delay", ascending=False)


def time_of_day_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize delays by time of day."""
    return data.groupby("time_of_day", observed=False).agg(
        flights=("time_of_day", "size"),
        mean_delay=("delay", "mean"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )


def distance_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize delays across flight-distance categories."""
    result = data.copy()
    result["distance_group"] = pd.cut(
        result["distance"],
        bins=[0, 500, 1000, 1500, 2500],
        labels=["0-500", "501-1000", "1001-1500", "1501-2500"]
    )
    return result.groupby("distance_group", observed=False).agg(
        flights=("distance_group", "size"),
        mean_delay=("delay", "mean"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )


def create_figures(data: pd.DataFrame,
                   output_dir: Path = FIGURE_DIR) -> None:
    """Create the exploratory figures used in the report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    destinations = destination_summary(data).head(10).sort_values("mean_delay")
    ax = destinations["mean_delay"].plot(kind="barh")
    ax.set_title("Top Destination Airports by Mean Flight Delay")
    ax.set_xlabel("Mean delay (minutes)")
    ax.set_ylabel("Destination airport")
    plt.tight_layout()
    plt.savefig(output_dir / "destination_delay.png", dpi=200)
    plt.close()

    time_data = time_of_day_summary(data)
    ax = (time_data["delayed_over_30_rate"] * 100).plot(kind="bar")
    ax.set_title("Flights Delayed More Than 30 Minutes by Time of Day")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Flights delayed over 30 minutes (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "time_of_day_delay.png", dpi=200)
    plt.close()

    distances = distance_summary(data)
    ax = distances["mean_delay"].plot(kind="bar")
    ax.set_title("Mean Flight Delay by Distance Group")
    ax.set_xlabel("Flight distance (miles)")
    ax.set_ylabel("Mean delay (minutes)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "distance_delay.png", dpi=200)
    plt.close()


def write_tables(data: pd.DataFrame,
                 output_dir: Path = FIGURE_DIR) -> None:
    """Save summary tables used in the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_summary(data).to_csv(output_dir / "missing_summary.csv")
    quantitative_summary(data).to_csv(output_dir / "quantitative_summary.csv")
    destination_summary(data).to_csv(output_dir / "destination_summary.csv")
    origin_summary(data).to_csv(output_dir / "origin_summary.csv")
    time_of_day_summary(data).to_csv(output_dir / "time_of_day_summary.csv")
    distance_summary(data).to_csv(output_dir / "distance_summary.csv")


def main() -> None:
    """Run the exploratory analysis."""
    data = load_data()
    create_figures(data)
    write_tables(data)
    print(f"Rows: {len(data)}, columns after engineering: {data.shape[1]}")
    print("\nMissing values:\n", missing_summary(data))
    print("\nQuantitative summary:\n", quantitative_summary(data))
    print("\nTop destinations:\n", destination_summary(data).head(10))
    print("\nTop origins:\n", origin_summary(data).head(10))
    print("\nTime of day:\n", time_of_day_summary(data))
    print("\nDistance groups:\n", distance_summary(data))


if __name__ == "__main__":
    main()
