#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIST = PROJECT_ROOT / "04_ml" / "banca" / "historico_apostas.csv"
OUT_DIR = PROJECT_ROOT / "04_ml" / "reports"


def atomic_csv(df, path):
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json(path, payload):
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def norm(x):
    return str(x).strip().upper()


def key(*parts):
    return "||".join(norm(p) for p in parts)


def profit_factor(lucros):
    wins = lucros[lucros > 0].sum()
    losses = abs(lucros[lucros < 0].sum())
    if losses <= 0:
        return float(wins) if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown(lucros):
    curve = pd.to_numeric(lucros, errors="coerce").fillna(0).cumsum()
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    dd = curve - peak
    return float(dd.min())


def context_score(row):
    roi = float(row.get("roi", 0) or 0)
    wr = float(row.get("winrate", 0) or 0)
    pf = float(row.get("profit_factor", 0) or 0)
    apostas = float(row.get("apostas", 0) or 0)
    # 0..100, 50 neutro. ROI manda, PF/WR e amostra refinam.
    sample_bonus = min(10.0, np.log1p(apostas) * 2.0)
    score = 50.0 + (roi * 180.0) + ((pf - 1.0) * 12.0) + ((wr - 0.5) * 18.0) + sample_bonus
    return round(max(0.0, min(100.0, score)), 4)


def calc_resumo(df, group_cols):
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            apostas=("mercado", "count"),
            ganhos=("resultado", lambda x: (x.astype(str).str.lower() == "ganhou").sum()),
            lucro=("lucro", "sum"),
            stake=("valor_apostado", "sum"),
            odd_media=("odd", "mean"),
            primeiro_dia=("data", "min"),
            ultimo_dia=("data", "max"),
        )
        .reset_index()
    )
    g["winrate"] = np.where(g["apostas"] > 0, g["ganhos"] / g["apostas"], 0.0)
    g["roi"] = np.where(g["stake"] > 0, g["lucro"] / g["stake"], 0.0)
    # profit factor e drawdown por grupo
    pf_map = {}
    dd_map = {}
    for name, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(name, tuple):
            name = (name,)
        k = tuple(norm(x) for x in name)
        lucros = pd.to_numeric(grp["lucro"], errors="coerce").fillna(0)
        pf_map[k] = profit_factor(lucros)
        dd_map[k] = max_drawdown(lucros)
    def get_metric(row, source, default=0.0):
        k = tuple(norm(row[c]) for c in group_cols)
        return source.get(k, default)
    g["profit_factor"] = g.apply(lambda r: get_metric(r, pf_map), axis=1)
    g["drawdown_max"] = g.apply(lambda r: get_metric(r, dd_map), axis=1)
    g["context_score"] = g.apply(context_score, axis=1)
    return g


def to_dict(df, cols):
    out = {}
    for _, r in df.iterrows():
        k = key(*(r[c] for c in cols))
        out[k] = {
            "apostas": int(r.get("apostas", 0)),
            "ganhos": int(r.get("ganhos", 0)),
            "lucro": round(float(r.get("lucro", 0) or 0), 4),
            "stake": round(float(r.get("stake", 0) or 0), 4),
            "odd_media": round(float(r.get("odd_media", 0) or 0), 4),
            "winrate": round(float(r.get("winrate", 0) or 0), 6),
            "roi": round(float(r.get("roi", 0) or 0), 6),
            "profit_factor": round(float(r.get("profit_factor", 0) or 0), 6),
            "drawdown_max": round(float(r.get("drawdown_max", 0) or 0), 4),
            "score": round(float(r.get("context_score", 50) or 50), 4),
        }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HIST.exists():
        raise FileNotFoundError(f"Histórico não encontrado: {HIST}")
    hist = pd.read_csv(HIST, low_memory=False)
    hist = hist[hist["resultado"].astype(str).str.lower().isin(["ganhou", "perdeu"])].copy()
    if hist.empty:
        raise RuntimeError("Histórico liquidado vazio; rode settlement antes dos scores contextuais.")
    for col in ["valor_apostado", "lucro", "odd"]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0)
    for col in ["mercado", "liga", "jogo"]:
        if col not in hist.columns:
            hist[col] = ""
    hist["mercado"] = hist["mercado"].map(norm)
    hist["liga"] = hist["liga"].map(norm)
    partes = hist["jogo"].astype(str).str.split(" x ", n=1, expand=True)
    hist["home"] = partes[0].fillna("").map(norm)
    hist["away"] = partes[1].fillna("").map(norm)

    market = calc_resumo(hist, ["mercado"])
    league = calc_resumo(hist, ["mercado", "liga"])
    team_base = pd.concat([
        hist.assign(time=hist["home"], lado="HOME"),
        hist.assign(time=hist["away"], lado="AWAY"),
    ], ignore_index=True)
    team = calc_resumo(team_base, ["mercado", "time"])
    league_team = calc_resumo(team_base, ["mercado", "liga", "time"])
    matchup = calc_resumo(hist, ["mercado", "liga", "home", "away"])

    atomic_csv(market, OUT_DIR / "market_scores.csv")
    atomic_csv(league, OUT_DIR / "league_market_scores.csv")
    atomic_csv(team, OUT_DIR / "team_market_scores.csv")
    atomic_csv(league_team, OUT_DIR / "league_team_market_scores.csv")
    atomic_csv(matchup, OUT_DIR / "matchup_market_scores.csv")

    payload = {
        "generated_from": str(HIST),
        "market": to_dict(market, ["mercado"]),
        "league": to_dict(league, ["mercado", "liga"]),
        "team": to_dict(team, ["mercado", "time"]),
        "league_team": to_dict(league_team, ["mercado", "liga", "time"]),
        "matchup": to_dict(matchup, ["mercado", "liga", "home", "away"]),
    }
    atomic_json(OUT_DIR / "context_operational_scores.json", payload)

    # Mantém compatibilidade com versões anteriores.
    blacklist = {
        "ligas": league[(league["apostas"] >= 10) & (league["roi"] <= -0.05)][["mercado", "liga", "apostas", "roi", "lucro"]].to_dict("records"),
        "times": team[(team["apostas"] >= 5) & (team["roi"] <= -0.08)][["mercado", "time", "apostas", "roi", "lucro"]].to_dict("records"),
        "liga_time": league_team[(league_team["apostas"] >= 3) & (league_team["roi"] <= -0.10)][["mercado", "liga", "time", "apostas", "roi", "lucro"]].to_dict("records"),
    }
    atomic_json(OUT_DIR / "context_blacklist.json", blacklist)

    print("=" * 60)
    print("SCORES CONTEXTUAIS OPERACIONAIS GERADOS")
    print("=" * 60)
    print(f"Market          : {len(market)}")
    print(f"League+Market   : {len(league)}")
    print(f"Team+Market     : {len(team)}")
    print(f"League+Team     : {len(league_team)}")
    print(f"Confronto       : {len(matchup)}")
    print(f"JSON operacional: {OUT_DIR / 'context_operational_scores.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
