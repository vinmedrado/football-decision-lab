#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "04_ml"
MODELS_SUMMARY = ML_DIR / "models" / "resumo_modelos.pkl"
HISTORICO = ML_DIR / "banca" / "historico_apostas.csv"
OUT = ML_DIR / "reports" / "perfil_operacional_mercados.json"
ELEGIBILIDADE = ML_DIR / "reports" / "elegibilidade_mercado_liga.csv"

# roi_bt vem de resumo_modelos.pkl e é o ROI agregado de TODAS as ligas do
# mercado, calculado antes da elegibilidade por liga entrar em ação
# (01_dataset_builder.py já filtra as ligas ruins na hora de montar o dataset
# de treino -- roi_bt aqui é só um número histórico de referência, não reflete
# os dados que o modelo realmente usou). Um mercado resgatado por liga (ex:
# R_FT_H) pode ter roi_bt global negativo mesmo assim. Por isso o corte aqui é
# permissivo por padrão: quem já decide qualidade real é o AUC, calculado no
# teste do próprio modelo treinado nos dados filtrados. Ajuste via env var
# PERFIL_MIN_ROI_BT se quiser reativar um corte extra por cima do AUC.
MIN_ROI_BT = float(os.getenv("PERFIL_MIN_ROI_BT", "-1"))
MIN_AUC = 0.58
ODD_MIN_DEFAULT = 1.20
ODD_MAX_DEFAULT = 3.50
MIN_EV_DEFAULT = 0.05

ODD_MIN_POR_TIPO = {
    "TG_HT_U05": 1.45,
    "TG_HT_U15": 1.35,
    "TG_HT_U25": 1.15,
    "DC_12": 1.20,
}


def get_mercado(meta):
    return str(meta.get("mercado") or meta.get("market") or meta.get("source") or "").strip()


def odd_min_sugerida(mercado):
    mercado = mercado.upper()
    return ODD_MIN_POR_TIPO.get(mercado, ODD_MIN_DEFAULT)


def carregar_resumo():
    with open(MODELS_SUMMARY, "rb") as f:
        return pickle.load(f)


def carregar_banca():
    if not HISTORICO.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(HISTORICO, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if df.empty or "resultado" not in df.columns:
        return pd.DataFrame()

    return df[df["resultado"].isin(["ganhou", "perdeu"])].copy()


def resumo_banca(df, mercado):
    if df.empty:
        return None

    g = df[df["mercado"].astype(str).str.upper() == mercado.upper()].copy()
    if g.empty:
        return None

    g["odd"] = pd.to_numeric(g["odd"], errors="coerce")
    g["lucro"] = pd.to_numeric(g["lucro"], errors="coerce")
    g["valor_apostado"] = pd.to_numeric(g["valor_apostado"], errors="coerce")

    stake = g["valor_apostado"].sum()
    lucro = g["lucro"].sum()
    apostas = len(g)
    ganhos = int((g["resultado"] == "ganhou").sum())

    return {
        "banca_apostas": int(apostas),
        "banca_winrate": round(ganhos / apostas, 4) if apostas else 0,
        "banca_roi": round(lucro / stake, 4) if stake else 0,
        "banca_odd_media": round(g["odd"].mean(), 4),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    resumo = carregar_resumo()
    banca = carregar_banca()

    perfil = {}

    ligas_ativas = {}

    if ELEGIBILIDADE.exists():
        eleg = pd.read_csv(ELEGIBILIDADE)

        ligas_ativas = (
            eleg.groupby("mercado")["status"]
            .apply(lambda s: (s == "ATIVA").any())
            .to_dict()
        )



    for meta in resumo:
        mercado = get_mercado(meta)
        if not mercado:
            continue

        roi_bt = float(meta.get("roi_bt", 0) or 0)
        auc = float(meta.get("auc", 0) or 0)
        winrate_bt = float(meta.get("winrate_bt", 0) or 0)
        status_modelo = str(meta.get("status", "")).lower()

        aprovado = (
            ligas_ativas.get(mercado, False)
            and auc >= MIN_AUC
            and status_modelo in {"approved", "aprovado", ""}
        )

        item = {
            "ativo": bool(aprovado),
            "fonte": "resumo_modelos_backtest",
            "odd_min": odd_min_sugerida(mercado),
            "odd_max": ODD_MAX_DEFAULT,
            "min_ev": MIN_EV_DEFAULT,
            "roi_bt": round(roi_bt, 6),
            "auc": round(auc, 6),
            "winrate_bt": round(winrate_bt, 6),
            "model_name": meta.get("model_name", ""),
            "status_modelo": meta.get("status", ""),
            "status": "ATIVO_POR_BACKTEST" if aprovado else "BLOQUEADO_POR_CRITERIO_BT",
        }

        apoio = resumo_banca(banca, mercado)
        if apoio:
            item.update(apoio)

        if not aprovado:
            motivos = []
            if not ligas_ativas.get(mercado, False):
                motivos.append("LIGA_NAO_ATIVA")
            if auc < MIN_AUC:
                motivos.append("AUC_ABAIXO")
            if status_modelo not in {"approved", "aprovado", ""}:
                motivos.append("STATUS_MODELO_NAO_APROVADO")
            item["motivo"] = ";".join(motivos) or "NAO_APROVADO"

        perfil[mercado] = item

    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp_out = OUT.with_name(f".{OUT.name}.{os.getpid()}.tmp")
    try:
        temp_out.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_out, OUT)
    finally:
        if temp_out.exists():
            temp_out.unlink()

    print("=" * 60)
    print("PERFIL OPERACIONAL GERADO")
    print("=" * 60)
    print(f"Mercados no perfil : {len(perfil)}")
    print(f"Ativos             : {sum(1 for x in perfil.values() if x.get('ativo'))}")
    print(f"Arquivo salvo      : {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
