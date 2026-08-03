"""Normaliza JSONs brutos da FutPython para o layout usado pelo projeto.

A saída padrão é data/base_unificada.csv, mantendo as principais colunas que o
pipeline histórico/ML já conhece. Campos ausentes na API entram vazios/0 para
não quebrar backtest e treino.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

RAW_DIR = Path(os.getenv("FUTPYTHON_RAW_DIR", "data/raw/futpython"))
OUTPUT_PATH = Path(os.getenv("FUTPYTHON_PROCESSED_PATH", "data/base_unificada.csv"))

BASE_COLUMNS = [
    "Nº", "Id_Jogo", "League", "Season", "Date", "Rodada", "Home", "Away",
    "Goals_H_HT", "Goals_A_HT", "TotalGoals_HT", "Goals_H_FT", "Goals_A_FT", "TotalGoals_FT",
    "Goals_H_Minutes", "Goals_A_Minutes", "Odd_H_HT", "Odd_D_HT", "Odd_A_HT",
    "Odd_Over05_HT", "Odd_Under05_HT", "Odd_Over15_HT", "Odd_Under15_HT",
    "Odd_Over25_HT", "Odd_Under25_HT", "Odd_H_FT", "Odd_D_FT", "Odd_A_FT",
    "Odd_Over05_FT", "Odd_Under05_FT", "Odd_Over15_FT", "Odd_Under15_FT",
    "Odd_Over25_FT", "Odd_Under25_FT", "Odd_BTTS_Yes", "Odd_BTTS_No",
    "Odd_DC_1X", "Odd_DC_12", "Odd_DC_X2", "PPG_Home_Pre", "PPG_Away_Pre",
    "PPG_Home", "PPG_Away", "XG_Home_Pre", "XG_Away_Pre", "XG_Total_Pre",
    "ShotsOnTarget_H", "ShotsOnTarget_A", "ShotsOffTarget_H", "ShotsOffTarget_A",
    "Shots_H", "Shots_A", "Corners_H_FT", "Corners_A_FT", "TotalCorners_FT",
    "Odd_Corners_H", "Odd_Corners_D", "Odd_Corners_A", "Odd_Corners_Over75",
    "Odd_Corners_Under75", "Odd_Corners_Over85", "Odd_Corners_Under85",
    "Odd_Corners_Over95", "Odd_Corners_Under95", "Odd_Corners_Over105",
    "Odd_Corners_Under105", "Odd_Corners_Over115", "Odd_Corners_Under115",
    "arquivo_origem", "liga_arquivo",
]


def first_value(data: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def deep_first(data: Any, paths: List[List[str]], default: Any = "") -> Any:
    for path in paths:
        cur = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return default


def as_list(payload: Any) -> List[dict]:
    """Extrai lista de partidas de formatos comuns de resposta JSON."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("matches", "games", "fixtures", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = as_list(value)
            if nested:
                return nested
    return []


def normalize_match(match: Dict[str, Any], catalog: Dict[str, Any], arquivo_origem: str) -> Dict[str, Any]:
    home = deep_first(match, [["home", "name"], ["home_team", "name"], ["teams", "home", "name"]])
    away = deep_first(match, [["away", "name"], ["away_team", "name"], ["teams", "away", "name"]])

    goals_h_ft = first_value(match, "Goals_H_FT", "home_goals", "goals_home", "home_score", default="")
    goals_a_ft = first_value(match, "Goals_A_FT", "away_goals", "goals_away", "away_score", default="")
    if goals_h_ft == "":
        goals_h_ft = deep_first(match, [["score", "fulltime", "home"], ["scores", "home"], ["goals", "home"]], default="")
    if goals_a_ft == "":
        goals_a_ft = deep_first(match, [["score", "fulltime", "away"], ["scores", "away"], ["goals", "away"]], default="")

    goals_h_ht = deep_first(match, [["score", "halftime", "home"], ["halftime", "home"]], default=first_value(match, "Goals_H_HT", default=""))
    goals_a_ht = deep_first(match, [["score", "halftime", "away"], ["halftime", "away"]], default=first_value(match, "Goals_A_HT", default=""))

    odds = match.get("odds", {}) if isinstance(match.get("odds"), dict) else {}
    ft = odds.get("ft") or odds.get("fulltime") or odds.get("1x2") or {}
    ht = odds.get("ht") or odds.get("halftime") or {}
    ou = odds.get("over_under") or odds.get("totals") or {}
    btts = odds.get("btts") or {}

    row = {col: "" for col in BASE_COLUMNS}
    row.update({
        "Id_Jogo": first_value(match, "Id_Jogo", "id", "match_id", "fixture_id", default=""),
        "League": first_value(match, "League", "liga", "liga_name", default=catalog.get("liga", "")),
        "Season": first_value(match, "Season", "season", default=catalog.get("season", "")),
        "Date": first_value(match, "Date", "date", "match_date", "datetime", "start_time", default=""),
        "Rodada": first_value(match, "Rodada", "round", "round_name", "stage", default=""),
        "Home": first_value(match, "Home", "home_name", "homeTeam", default=home),
        "Away": first_value(match, "Away", "away_name", "awayTeam", default=away),
        "Goals_H_HT": goals_h_ht,
        "Goals_A_HT": goals_a_ht,
        "Goals_H_FT": goals_h_ft,
        "Goals_A_FT": goals_a_ft,
        "Goals_H_Minutes": first_value(match, "Goals_H_Minutes", "home_goal_minutes", default="[]"),
        "Goals_A_Minutes": first_value(match, "Goals_A_Minutes", "away_goal_minutes", default="[]"),
        "Odd_H_HT": first_value(ht, "home", "h", "1", default=first_value(match, "Odd_H_HT", default="")),
        "Odd_D_HT": first_value(ht, "draw", "d", "x", default=first_value(match, "Odd_D_HT", default="")),
        "Odd_A_HT": first_value(ht, "away", "a", "2", default=first_value(match, "Odd_A_HT", default="")),
        "Odd_H_FT": first_value(ft, "home", "h", "1", default=first_value(match, "Odd_H_FT", default="")),
        "Odd_D_FT": first_value(ft, "draw", "d", "x", default=first_value(match, "Odd_D_FT", default="")),
        "Odd_A_FT": first_value(ft, "away", "a", "2", default=first_value(match, "Odd_A_FT", default="")),
        "Odd_Over25_FT": first_value(ou, "over_25", "over25", "o25", default=first_value(match, "Odd_Over25_FT", default="")),
        "Odd_Under25_FT": first_value(ou, "under_25", "under25", "u25", default=first_value(match, "Odd_Under25_FT", default="")),
        "Odd_BTTS_Yes": first_value(btts, "yes", "sim", default=first_value(match, "Odd_BTTS_Yes", default="")),
        "Odd_BTTS_No": first_value(btts, "no", "nao", "não", default=first_value(match, "Odd_BTTS_No", default="")),
        "arquivo_origem": arquivo_origem,
        "liga_arquivo": f"{catalog.get('country', '')} {catalog.get('liga', '')} {catalog.get('season', '')}".strip(),
    })

    for total_col, a, b in [
        ("TotalGoals_HT", "Goals_H_HT", "Goals_A_HT"),
        ("TotalGoals_FT", "Goals_H_FT", "Goals_A_FT"),
        ("TotalCorners_FT", "Corners_H_FT", "Corners_A_FT"),
    ]:
        try:
            row[total_col] = float(row[a]) + float(row[b])
        except Exception:
            row[total_col] = ""

    return row


def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    return sorted(raw_dir.glob("*.json"))


def normalize(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH, append: bool = True) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_raw_files(raw_dir):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        catalog = envelope.get("catalog", {}) if isinstance(envelope, dict) else {}
        payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else envelope
        for match in as_list(payload):
            rows.append(normalize_match(match, catalog, path.name))

    df = pd.DataFrame(rows, columns=BASE_COLUMNS)
    if df.empty:
        print("Nenhuma partida encontrada nos JSONs brutos.")
        return df

    df.insert(0, "_tmp_order", range(1, len(df) + 1))
    df["Nº"] = df["_tmp_order"]
    df = df.drop(columns=["_tmp_order"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and output_path.exists():
        old = pd.read_csv(output_path, dtype=str).fillna("")
        combined = pd.concat([old, df.astype(str)], ignore_index=True)
        if "Id_Jogo" in combined.columns:
            combined = combined.drop_duplicates(subset=["Id_Jogo"], keep="last")
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")
        return combined

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalizar JSON FutPython para base_unificada.csv")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--replace", action="store_true", help="Substitui a base em vez de anexar")
    args = parser.parse_args()

    df = normalize(Path(args.raw_dir), Path(args.output), append=not args.replace)
    print(f"Arquivo gerado/atualizado: {args.output}")
    print(f"Linhas totais: {len(df)}")


if __name__ == "__main__":
    main()
