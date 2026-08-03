from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_XLSX = Path("data/catalog/base.xlsx")
DEFAULT_CSV = Path(os.getenv("FUTPYTHON_CATALOG_PATH", "data/catalog/ligas_catalog.csv"))

COLUMN_MAP = {
    "País": "country",
    "Liga": "liga",
    "Temporada": "season",
    "ID (Copie este valor)": "liga_id",
}


def build_catalog(input_xlsx: Path = DEFAULT_XLSX, output_csv: Path = DEFAULT_CSV) -> pd.DataFrame:
    if not input_xlsx.exists():
        raise FileNotFoundError(f"Planilha não encontrada: {input_xlsx}")

    df = pd.read_excel(input_xlsx).rename(columns=COLUMN_MAP)
    required = ["country", "liga", "season", "liga_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na planilha base: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=["country", "liga", "season"])
    df["country"] = df["country"].astype(str).str.strip()
    df["liga"] = df["liga"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip()
    df["liga_id"] = df["liga_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["country", "liga", "season"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera catálogo local de ligas a partir da base.xlsx.")
    parser.add_argument("--input", default=str(DEFAULT_XLSX))
    parser.add_argument("--output", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    df = build_catalog(Path(args.input), Path(args.output))
    print(f"Catálogo salvo em: {args.output}")
    print(f"Total de ligas/temporadas: {len(df)}")
    print(f"Países: {df['country'].nunique()}")


if __name__ == "__main__":
    main()
