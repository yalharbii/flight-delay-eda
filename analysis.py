"""
Yousef Alharbi
CSE 163 - Summer 2026
Final Project

Performs the final flight-delay analysis, compares routes and days of the
week, and trains two models to predict delays longer than 30 minutes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from eda import FIGURE_DIR, load_data


def day_of_week_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize delays by day of week."""
    summary = data.groupby("day_of_week").agg(
        flights=("day_of_week", "size"),
        mean_delay=("delay", "mean"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )
    return summary.sort_values("mean_delay", ascending=False)


def route_summary(data: pd.DataFrame,
                  min_flights: int = 20) -> pd.DataFrame:
    """Summarize delays for origin-destination routes."""
    result = data.copy()
    result["route"] = result["origin"] + "-" + result["destination"]
    grouped = result.groupby("route").agg(
        flights=("route", "size"),
        mean_delay=("delay", "mean"),
        delayed_over_30_rate=("delay_over_30", "mean")
    )
    grouped = grouped[grouped["flights"] >= min_flights]
    return grouped.sort_values("mean_delay", ascending=False)


def model_results(data: pd.DataFrame) -> pd.DataFrame:
    """Train and compare two models for predicting long flight delays."""
    features = [
        "origin", "destination", "time_of_day", "day_of_week",
        "month", "hour", "distance"
    ]
    clean = data.dropna(subset=features).copy()

    x = pd.get_dummies(clean[features])
    y = clean["delay_over_30"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )

    models = {
        "Random forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=1
        )
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions,
                                         zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0)
        })

    return pd.DataFrame(rows).set_index("model")


def create_analysis_figures(data: pd.DataFrame,
                            output_dir: Path = FIGURE_DIR) -> None:
    """Create figures for the final analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    days = day_of_week_summary(data).sort_values("mean_delay")
    ax = days["mean_delay"].plot(kind="barh")
    ax.set_title("Mean Flight Delay by Day of Week")
    ax.set_xlabel("Mean delay (minutes)")
    ax.set_ylabel("Day of week")
    plt.tight_layout()
    plt.savefig(output_dir / "day_of_week_delay.png", dpi=200)
    plt.close()

    routes = route_summary(data).head(10).sort_values("mean_delay")
    ax = routes["mean_delay"].plot(kind="barh")
    ax.set_title("Top Routes by Mean Flight Delay")
    ax.set_xlabel("Mean delay (minutes)")
    ax.set_ylabel("Route")
    plt.tight_layout()
    plt.savefig(output_dir / "route_delay.png", dpi=200)
    plt.close()

    results = model_results(data)
    ax = results[["accuracy", "f1"]].plot(kind="bar")
    ax.set_title("Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance.png", dpi=200)
    plt.close()


def write_analysis_tables(data: pd.DataFrame,
                          output_dir: Path = FIGURE_DIR) -> None:
    """Save final-analysis tables used in the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    day_of_week_summary(data).to_csv(output_dir / "day_of_week_summary.csv")
    route_summary(data).to_csv(output_dir / "route_summary.csv")
    model_results(data).to_csv(output_dir / "model_results.csv")


def main() -> None:
    """Run the final analysis."""
    data = load_data()
    create_analysis_figures(data)
    write_analysis_tables(data)
    print("\nDay-of-week summary:\n", day_of_week_summary(data))
    print("\nTop routes:\n", route_summary(data).head(10))
    print("\nModel results:\n", model_results(data))


if __name__ == "__main__":
    main()
