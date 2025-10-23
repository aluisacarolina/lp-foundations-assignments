from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def clean_data(country_code: str = "PT") -> pd.DataFrame:
    """
    Limpa o ficheiro wide da UE e produz um CSV filtrado por país.
    - Lê data/eu_life_expectancy_raw.tsv
    - Faz unpivot para colunas: unit, sex, age, region, year, value
    - year -> int; value -> float (coerce), remove NaN
    - Filtra region == country_code
    - Guarda em data/pt_life_expectancy.csv (nome fixo)
    """
    data_dir = Path(__file__).resolve().parent / "data"
    in_path = data_dir / "eu_life_expectancy_raw.tsv"
    out_path = data_dir / "pt_life_expectancy.csv"

    # 1) Ler TSV wide
    df = pd.read_csv(in_path, sep="\t")

    # 2) Unpivot -> long
    id_col = df.columns[0]
    long_df = df.melt(id_vars=[id_col], var_name="year", value_name="value")

    # 3) Separar a coluna composta e LIMPAR espaços
    split_cols = long_df[id_col].str.split(",", expand=True)
    split_cols.columns = ["unit", "sex", "age", "region"]
    split_cols = split_cols.apply(lambda s: s.astype(str).str.strip())

    # 4) Substituir no long_df e remover a coluna original
    long_df = pd.concat([split_cols, long_df[["year", "value"]]], axis=1)

    # 5) Limpar year -> int (extrair YYYY)
    long_df["year"] = long_df["year"].astype(str).str.extract(r"(\d{4})").astype(int)

    # 6) Extrair parte numérica de value e converter -> float
    val_str = long_df["value"].astype(str)
    num = val_str.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    long_df["value"] = pd.to_numeric(num, errors="coerce")

    # 7) Remover NaN
    long_df = long_df.dropna(subset=["value"]).copy()

    # 8) Filtrar país
    long_df = long_df[long_df["region"].astype(str) == country_code].copy()

    # 9) Ordenar colunas e guardar
    long_df = long_df[["unit", "sex", "age", "region", "year", "value"]]
    long_df.to_csv(out_path, index=False)

    return long_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpa dados de esperança de vida da UE e guarda CSV filtrado por país."
    )
    parser.add_argument(
        "--country", "-c", default="PT", help="Código do país a filtrar (default: PT)."
    )
    return parser.parse_args()


if __name__ == "__main__": 
    args = _parse_args()
    clean_data(country_code=args.country)
