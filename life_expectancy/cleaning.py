from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def clean_data(country_code: str = "PT") -> pd.DataFrame:
    """
    Clean the raw EU life expectancy dataset and produce a cleaned CSV filtered by country.

    Steps performed:
    - Load the wide-format TSV file.
    - Unpivot (melt) the year columns into long format.
    - Split the compound first column into: unit, sex, age, region.
    - Convert `year` to int and `value` to float, removing invalid entries.
    - Filter the dataset by the given country code (default: PT).
    - Save the result as `pt_life_expectancy.csv` (no index).
    """
    data_dir = Path(__file__).resolve().parent / "data"
    in_path = data_dir / "eu_life_expectancy_raw.tsv"
    out_path = data_dir / "pt_life_expectancy.csv"

    df = pd.read_csv(in_path, sep="\t")

    # Unpivot year columns to long format
    id_col = df.columns[0]
    long_df = df.melt(id_vars=[id_col], var_name="year", value_name="value")

    # Split composite column into separate fields
    split_cols = long_df[id_col].str.split(",", expand=True)
    split_cols.columns = ["unit", "sex", "age", "region"]
    split_cols = split_cols.apply(lambda s: s.astype(str).str.strip())

    long_df = pd.concat([split_cols, long_df[["year", "value"]]], axis=1)

    # Extract 4-digit year
    long_df["year"] = (
        long_df["year"]
        .astype(str)
        .str.extract(r"(\d{4})")
        .astype(int)
    )

    # Extract numeric value (regex handles values like ":" or "12.3 p")
    numeric = long_df["value"].astype(str).str.extract(
        r"([-+]?\d+(?:\.\d+)?)", expand=False
    )
    long_df["value"] = pd.to_numeric(numeric, errors="coerce")

    # Remove rows without numeric values
    long_df = long_df.dropna(subset=["value"]).copy()

    # Country filter
    long_df = long_df[long_df["region"] == country_code].copy()

    long_df = long_df[["unit", "sex", "age", "region", "year", "value"]]
    long_df.to_csv(out_path, index=False)

    return long_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean EU life expectancy dataset and filter by country."
    )
    parser.add_argument(
        "--country", "-c", default="PT",
        help="Country code to filter (default: PT).",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    clean_data(country_code=args.country)