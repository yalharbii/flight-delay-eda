"""Exploratory data analysis for a 2015 U.S. flight-delay extract."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).parent / "flight_sample.csv"
FIGURE_DIR = Path(__file__).parent / "figures"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the flight-delay CSV and add analysis variables."""
    data = pd.read_csv(path)
    data["DELAY_OVER_30"] = data["DEPARTURE_DELAY"] > 30
    data["DEPARTURE_HOUR"] = data["SCHEDULED_DEPARTURE"] // 100
    data["TIME_OF_DAY"] = pd.cut(
        data["DEPARTURE_HOUR"],
        bins=[-1, 5, 11, 16, 20, 24],
        labels=["Overnight", "Morning", "Afternoon", "Evening", "Late night"],
    )
    return data


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every column."""
    counts = data.isna().sum()
    result = pd.DataFrame({"missing_count": counts})
    result["missing_percent"] = counts / len(data) * 100
    return result


def quantitative_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return the required seven-number summaries."""
    columns = ["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE"]
    summary = data[columns].describe().T
    return summary[["mean", "std", "min", "25%", "50%", "75%", "max"]]


def airport_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize delay outcomes by destination airport."""
    return (
        data.groupby("IATA_Code")
        .agg(
            flights=("IATA_Code", "size"),
            mean_departure_delay=("DEPARTURE_DELAY", "mean"),
            mean_arrival_delay=("ARRIVAL_DELAY", "mean"),
            delayed_over_30_rate=("DELAY_OVER_30", "mean"),
        )
        .sort_values("mean_departure_delay", ascending=False)
    )


def cause_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count delay-cause categories and calculate their shares."""
    counts = data["CANCELLATION_REASON"].value_counts().rename("count")
    result = counts.to_frame()
    result["percent"] = result["count"] / len(data) * 100
    return result


def create_figures(data: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> None:
    """Create and save all EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    airport = airport_summary(data)
    ax = airport["mean_departure_delay"].plot(kind="bar")
    ax.set_title("Mean Departure Delay by Destination Airport")
    ax.set_xlabel("Destination airport")
    ax.set_ylabel("Mean departure delay (minutes)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "airport_departure_delay.png", dpi=200)
    plt.close()

    causes = cause_summary(data)
    ax = causes["count"].plot(kind="bar")
    ax.set_title("Flight Records by Reported Delay Cause")
    ax.set_xlabel("Reported cause")
    ax.set_ylabel("Number of records")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "delay_causes.png", dpi=200)
    plt.close()

    time_rates = data.groupby("TIME_OF_DAY", observed=False)["DELAY_OVER_30"].mean()
    ax = (time_rates * 100).plot(kind="bar")
    ax.set_title("Share of Flights Delayed More Than 30 Minutes")
    ax.set_xlabel("Scheduled time of day")
    ax.set_ylabel("Flights delayed over 30 minutes (%)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "time_of_day_delay_rate.png", dpi=200)
    plt.close()


def write_tables(data: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> None:
    """Save numeric and categorical summaries used in the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_summary(data).to_csv(output_dir / "missing_summary.csv")
    quantitative_summary(data).to_csv(output_dir / "quantitative_summary.csv")
    airport_summary(data).to_csv(output_dir / "airport_summary.csv")
    cause_summary(data).to_csv(output_dir / "cause_summary.csv")


def main() -> None:
    """Run the EDA and print key results."""
    data = load_data()
    create_figures(data)
    write_tables(data)
    print(f"Rows: {data.shape[0]}, columns after feature engineering: {data.shape[1]}")
    print("\nMissing values:\n", missing_summary(data))
    print("\nQuantitative summaries:\n", quantitative_summary(data))
    print("\nAirport summaries:\n", airport_summary(data))
    print("\nDelay-cause summaries:\n", cause_summary(data))


if __name__ == "__main__":
    main()
