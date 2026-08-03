#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor de Elegibilidade por Mercado x Liga.

Versão corrigida:
- ROI por mercado x liga é ponderado por stake quando a coluna existe.
- Se não houver stake, usa ROI ponderado por quantidade de apostas.
- Winrate também é ponderado por apostas.
- Mantém histerese/strikes para evitar bloqueio instável.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = ML_DIR.parent
for _p in (ROOT_DIR, ML_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.project_paths import REPORTS_DIR, now_iso, read_json, write_json  # noqa: E402

CONTEXTO_DIR = ROOT_DIR / "03_backtest" / "results" / "contexto"
ELEGIBILIDADE_CSV_PATH = REPORTS_DIR / "elegibilidade_mercado_liga.csv"
ELEGIBILIDADE_HISTORICO_PATH = REPORTS_DIR / "historico_elegibilidade_mercado_liga.json"

MIN_APOSTAS_LIGA = 30
MIN_ROI_LIGA = 0.03
MIN_STRIKES_BLOQUEIO = 2
MAX_HISTORICO_EXECUCOES = 50

STATUS_VALIDOS = {"ATIVA", "OBSERVACAO", "BLOQUEADA"}


def _carregar_contexto_liga() -> pd.DataFrame:
    if not CONTEXTO_DIR.exists():
        return pd.DataFrame(columns=["mercado", "League_std", "apostas", "lucro", "stake", "roi", "winrate"])

    frames: List[pd.DataFrame] = []
    for path in sorted(CONTEXTO_DIR.glob("*_liga.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[AVISO] Falha ao ler {path.name}: {exc}", flush=True)
            continue

        if df.empty or "League_std" not in df.columns:
            continue

        if "mercado" not in df.columns:
            df["mercado"] = path.name.replace("_liga.csv", "")

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["mercado", "League_std", "apostas", "lucro", "stake", "roi", "winrate"])

    todos = pd.concat(frames, ignore_index=True, sort=False)

    for col in ("apostas", "lucro", "stake", "roi", "winrate"):
        if col not in todos.columns:
            todos[col] = 0.0
        todos[col] = pd.to_numeric(todos[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    todos["mercado"] = todos["mercado"].astype(str).str.strip()
    todos["League_std"] = todos["League_std"].astype(str).str.strip().str.upper()
    return todos


def _carregar_snapshot_anterior() -> Dict[str, Dict[str, Any]]:
    if not ELEGIBILIDADE_CSV_PATH.exists():
        return {}
    try:
        anterior = pd.read_csv(ELEGIBILIDADE_CSV_PATH, encoding="utf-8-sig")
    except Exception:
        return {}

    resultado: Dict[str, Dict[str, Any]] = {}
    for _, row in anterior.iterrows():
        chave = f"{row.get('mercado')}||{row.get('liga')}"
        resultado[chave] = row.to_dict()
    return resultado


def _classificar(apostas: float, roi: float, strikes_anteriores: int, status_anterior: str) -> tuple[str, int, str]:
    if apostas < MIN_APOSTAS_LIGA:
        if status_anterior == "ATIVA":
            return "ATIVA", 0, "amostra_pequena_mas_status_anterior_ativo_preservado"
        return "OBSERVACAO", 0, "amostra_insuficiente"

    if roi > MIN_ROI_LIGA:
        return "ATIVA", 0, "roi_acima_do_minimo"

    novos_strikes = strikes_anteriores + 1
    if novos_strikes >= MIN_STRIKES_BLOQUEIO:
        return "BLOQUEADA", novos_strikes, f"roi_abaixo_do_minimo_{novos_strikes}x_seguidas"
    return "OBSERVACAO", novos_strikes, f"roi_abaixo_do_minimo_strike_{novos_strikes}_de_{MIN_STRIKES_BLOQUEIO}"


def _agrupar_ponderado(contexto: pd.DataFrame) -> pd.DataFrame:
    base = contexto.copy()

    # Quando o runner exporta stake, essa é a forma correta: lucro / stake.
    # Quando não há stake, cai para ROI ponderado por apostas.
    base["roi_x_apostas"] = base["roi"] * base["apostas"]
    base["winrate_x_apostas"] = base["winrate"] * base["apostas"]

    g = (
        base.groupby(["mercado", "League_std"], as_index=False)
        .agg(
            apostas=("apostas", "sum"),
            lucro=("lucro", "sum"),
            stake=("stake", "sum"),
            roi_x_apostas=("roi_x_apostas", "sum"),
            winrate_x_apostas=("winrate_x_apostas", "sum"),
        )
    )

    apostas_safe = g["apostas"].replace(0, np.nan)
    stake_safe = g["stake"].replace(0, np.nan)

    roi_por_stake = g["lucro"] / stake_safe
    roi_por_apostas = g["roi_x_apostas"] / apostas_safe
    g["roi"] = roi_por_stake.where(g["stake"] > 0, roi_por_apostas)
    g["winrate"] = g["winrate_x_apostas"] / apostas_safe

    return g.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def calcular_elegibilidade() -> pd.DataFrame:
    contexto = _carregar_contexto_liga()
    anterior = _carregar_snapshot_anterior()

    if contexto.empty:
        print(
            "[AVISO] Nenhum arquivo de contexto por liga encontrado em "
            f"{CONTEXTO_DIR}. Rode o backtest antes deste motor.",
            flush=True,
        )
        return pd.DataFrame(columns=[
            "mercado", "liga", "apostas", "roi", "winrate", "status",
            "strikes_consecutivos", "motivo", "atualizado_em",
        ])

    agrupado = _agrupar_ponderado(contexto)

    linhas: List[Dict[str, Any]] = []
    for _, row in agrupado.iterrows():
        mercado = str(row["mercado"])
        liga = str(row["League_std"])
        apostas = float(row["apostas"])
        roi = float(row["roi"])
        winrate = float(row["winrate"])

        chave = f"{mercado}||{liga}"
        info_anterior = anterior.get(chave, {})
        strikes_anteriores = int(info_anterior.get("strikes_consecutivos", 0) or 0)
        status_anterior = str(info_anterior.get("status", "DESCONHECIDA") or "DESCONHECIDA")

        status, strikes, motivo = _classificar(apostas, roi, strikes_anteriores, status_anterior)

        linhas.append({
            "mercado": mercado,
            "liga": liga,
            "apostas": int(apostas),
            "lucro": round(float(row.get("lucro", 0.0)), 4),
            "stake": round(float(row.get("stake", 0.0)), 4),
            "roi": round(roi, 4),
            "winrate": round(winrate, 4),
            "status": status,
            "strikes_consecutivos": strikes,
            "status_anterior": status_anterior,
            "motivo": motivo,
            "atualizado_em": now_iso(),
        })

    return pd.DataFrame(linhas).sort_values(["mercado", "status", "liga"]).reset_index(drop=True)


def _registrar_historico(df: pd.DataFrame) -> None:
    historico = read_json(ELEGIBILIDADE_HISTORICO_PATH, {"execucoes": []})
    if not isinstance(historico, dict) or "execucoes" not in historico:
        historico = {"execucoes": []}

    resumo_status = df["status"].value_counts().to_dict() if not df.empty else {}
    historico["execucoes"].append({
        "quando": now_iso(),
        "total_combinacoes": int(len(df)),
        "por_status": {str(k): int(v) for k, v in resumo_status.items()},
    })
    historico["execucoes"] = historico["execucoes"][-MAX_HISTORICO_EXECUCOES:]
    write_json(ELEGIBILIDADE_HISTORICO_PATH, historico)


def main() -> None:
    print("=" * 60, flush=True)
    print("Motor de Elegibilidade por Mercado x Liga", flush=True)
    print("=" * 60, flush=True)

    df = calcular_elegibilidade()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ELEGIBILIDADE_CSV_PATH, index=False, encoding="utf-8-sig")
    _registrar_historico(df)

    if df.empty:
        print("Nenhuma combinação mercado x liga avaliada.", flush=True)
        return

    resumo = df["status"].value_counts()
    print(f"\nCombinações avaliadas : {len(df)}", flush=True)
    for status in ("ATIVA", "OBSERVACAO", "BLOQUEADA"):
        print(f"  {status:<12}: {int(resumo.get(status, 0))}", flush=True)
    print(f"\nSalvo em: {ELEGIBILIDADE_CSV_PATH}", flush=True)
    print(f"Histórico em: {ELEGIBILIDADE_HISTORICO_PATH}", flush=True)


if __name__ == "__main__":
    main()
