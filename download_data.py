"""
Yousef Alharbi
CSE 163 - Summer 2026
Final Project

Downloads the public 10,000-row flight dataset and saves it as a CSV file
used by the project.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

JSON_PATH = Path(__file__).parent / "flights-10k.json"
CSV_PATH = Path(__file__).parent / "flight_sample.csv"
DATA_URL = (
    "https://unpkg.com/vega-datasets@1.25.0/data/"
    "flights-10k.json"
)


def main() -> None:
    """Download the public flight dataset and save it as a CSV file."""
    urlretrieve(DATA_URL, JSON_PATH)
    data = pd.read_json(JSON_PATH)
    data.to_csv(CSV_PATH, index=False)
    JSON_PATH.unlink()
    print(f"Saved {len(data)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
