"""
Data-cleaning module for the EU life expectancy dataset.

This module follows the Single Responsibility Principle (SRP):
- `load_data` reads the raw input file.
- `clean_data` transforms the raw dataframe into tidy format.
- `save_data` writes the cleaned output to disk.
- `main` coordinates the full pipeline and command-line usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_FILE = DATA_DIR / "eu_life_expectancy_raw.tsv"
OUT_FILE = DATA_DIR / "pt_life_expectancy.csv"


def load_data() -> pd.DataFrame:
    """Load the raw TSV dataset."""
    return pd.read_csv(RAW_FILE, sep="\t")


def clean_data(df: pd.DataFrame, country_code: str = "PT") -> pd.DataFrame:
    """Transform raw data into a tidy dataframe filtered by country."""
    id_col = df.columns[0]

    long_df = df.melt(id_vars=[id_col], var_name="year", value_name="value")

    # Split composite field: "unit,sex,age,region"
    split_cols = long_df[id_col].str.split(",", expand=True)
    split_cols.columns = ["unit", "sex", "age", "region"]
    split_cols = split_cols.apply(lambda s: s.astype(str).str.strip())

    long_df = pd.concat([split_cols, long_df[["year", "value"]]], axis=1)

    # Extract YYYY
    long_df["year"] = long_df["year"].astype(str).str.extract(r"(\d{4})").astype(int)

    # Extract numeric value from strings like ":" or "12.3 p"
    value_num = long_df["value"].astype(str).str.extract(
        r"([-+]?\d+(?:\.\d+)?)", expand=False
    )
    long_df["value"] = pd.to_numeric(value_num, errors="coerce")

    long_df = long_df.dropna(subset=["value"])

    filtered = long_df[long_df["region"] == country_code]

    return filtered[["unit", "sex", "age", "region", "year", "value"]]


def save_data(df: pd.DataFrame, output_path: Path = OUT_FILE) -> None:
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean EU life expectancy dataset and filter by country."
    )
    parser.add_argument(
        "--country", "-c", default="PT",
        help="Country code to filter (default: PT).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full cleaning pipeline: load, clean, save."""
    args = _parse_args()
    raw = load_data()
    cleaned = clean_data(raw, country_code=args.country)
    save_data(cleaned)


if __name__ == "__main__":  # pragma: no cover
    main()
