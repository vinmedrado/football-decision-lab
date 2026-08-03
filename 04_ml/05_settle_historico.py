#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.settlement_utils import resolver_mercado

HISTORICO = os.path.join(BASE_DIR, "banca", "historico_apostas.csv")
BASE = os.path.join(ROOT_DIR, "data", "base_oficial.csv")
COMANDO_ATUALIZAR_BASE = (
    "python 01_scripts/run_pipeline.py "
    "--fetch --country <pais> --season <temporada> --incremental"
)


def norm(x):
    x = unicodedata.normalize("NFKD", str(x))
    x = x.encode("ascii", "ignore").decode("ascii")
    return x.strip().lower()


def split_jogo(jogo):
    jogo = str(jogo).strip()
    if " x " not in jogo:
        return "", ""
    home, away = jogo.split(" x ", 1)
    return home.strip(), away.strip()


def _datas_pendentes(hist):
    if "resultado" not in hist.columns or "data" not in hist.columns:
        return []

    resultado = hist["resultado"].fillna("").astype(str).str.strip().str.lower()
    datas = hist.loc[resultado.eq("pendente"), "data"].fillna("").astype(str).str.strip()
    return sorted(data for data in datas.unique() if data)


def _datas_disponiveis_base(base_path, datas_necessarias):
    if not datas_necessarias:
        return set()

    datas_necessarias = set(datas_necessarias)
    try:
        datas_base = set()
        for chunk in pd.read_csv(
            base_path,
            usecols=["Date"],
            chunksize=200_000,
            low_memory=False,
        ):
            normalizadas = chunk["Date"].fillna("").astype(str).str.strip()
            datas_base.update(data for data in normalizadas.unique() if data in datas_necessarias)
            if datas_base == datas_necessarias:
                break
        return datas_base
    except ValueError:
        print("ERRO: data/base_oficial.csv nao possui a coluna obrigatoria Date.")
        return set()


def validar_cobertura_datas_pendentes(hist, base_path):
    datas_pendentes = _datas_pendentes(hist)
    if not datas_pendentes:
        return []

    datas_base = _datas_disponiveis_base(base_path, datas_pendentes)
    return [data for data in datas_pendentes if data not in datas_base]


def cobertura_datas_pendentes(hist, base_path):
    datas_pendentes = _datas_pendentes(hist)
    if not datas_pendentes:
        return set(), []

    datas_base = _datas_disponiveis_base(base_path, datas_pendentes)
    datas_sem_base = [data for data in datas_pendentes if data not in datas_base]
    return datas_base, datas_sem_base


def carregar_base_para_datas(base_path, datas_necessarias):
    """Carrega somente as datas necessárias para o settlement.

    A base oficial é grande e antes era lida duas vezes: uma para cobertura e
    outra integralmente. O filtro em chunks preserva os mesmos registros usados
    na liquidação e reduz o pico de memória.
    """
    datas_necessarias = set(datas_necessarias)
    cabecalho = pd.read_csv(base_path, nrows=0)
    colunas = set(cabecalho.columns)
    home_col = "Home_std" if "Home_std" in colunas else "Home"
    away_col = "Away_std" if "Away_std" in colunas else "Away"
    liga_col = "League_std" if "League_std" in colunas else ("League" if "League" in colunas else None)

    obrigatorias = {"Date", home_col, away_col, "G_H_FT", "G_A_FT", "G_H_HT", "G_A_HT"}
    if liga_col:
        obrigatorias.update({liga_col, "Round"})
    faltantes = sorted(obrigatorias - colunas)
    if faltantes:
        raise ValueError("base_oficial sem colunas obrigatorias: " + ", ".join(faltantes))

    partes = []
    for chunk in pd.read_csv(
        base_path,
        usecols=sorted(obrigatorias),
        chunksize=200_000,
        low_memory=False,
    ):
        datas = chunk["Date"].fillna("").astype(str).str.strip()
        filtrado = chunk.loc[datas.isin(datas_necessarias)].copy()
        if not filtrado.empty:
            partes.append(filtrado)

    if not partes:
        return pd.DataFrame(columns=sorted(obrigatorias))
    return pd.concat(partes, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Liquida apostas pendentes contra data/base_oficial.csv.")
    parser.add_argument("--date", help="Liquida apenas apostas pendentes da data YYYY-MM-DD informada.")
    parser.add_argument(
        "--skip-post-update",
        action="store_true",
        help="Não executa rebuild/perfil/contexto; usado por orquestradores que executam essas etapas explicitamente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(HISTORICO):
        print(f"❌ Histórico não encontrado: {HISTORICO}")
        return 1

    if not os.path.exists(BASE):
        print(f"❌ Base oficial não encontrada: {BASE}")
        print("   Gere data/base_oficial.csv antes de executar o settlement.")
        return 1

    print("Carregando histórico...")
    hist = pd.read_csv(HISTORICO, low_memory=False)
    hist_para_settlement = hist
    if args.date and "data" in hist.columns:
        hist_para_settlement = hist[hist["data"].astype(str).str.strip() == str(args.date)]

    datas_pendentes = _datas_pendentes(hist_para_settlement)
    if not datas_pendentes:
        print("Nenhuma aposta pendente encontrada para o recorte solicitado.")
        return 0

    print("Carregando recorte necessário da base oficial...")
    try:
        base = carregar_base_para_datas(BASE, datas_pendentes)
    except (ValueError, pd.errors.EmptyDataError) as exc:
        print(f"ERRO: falha ao carregar base oficial: {exc}")
        return 1

    base["Date"] = base["Date"].fillna("").astype(str).str.strip()
    datas_com_base = set(base["Date"].unique()) & set(datas_pendentes)
    datas_sem_base = [data for data in datas_pendentes if data not in datas_com_base]
    if datas_sem_base:
        print("=" * 60)
        print("Settlement parcial: base oficial sem algumas datas pendentes")
        print("=" * 60)
        print("Datas pendentes ausentes em data/base_oficial.csv:")
        for data in datas_sem_base:
            print(f"  - {data}")
        print("")
        print("Apostas dessas datas serao preservadas como pendentes.")
        print("Datas cobertas serao liquidadas normalmente.")
        print("Atualize a base oficial para liquidar as datas restantes.")
        print("Comando minimo recomendado por pais/temporada pendente:")
        print(f"  {COMANDO_ATUALIZAR_BASE}")
        print("=" * 60)

    hist["data"] = hist["data"].astype(str).str.strip()

    print("Criando índice da base...")

    home_col = "Home_std" if "Home_std" in base.columns else "Home"
    away_col = "Away_std" if "Away_std" in base.columns else "Away"
    liga_col = "League_std" if "League_std" in base.columns else ("League" if "League" in base.columns else None)

    if liga_col:
        base["_key"] = (
            base["Date"].astype(str).str.strip()
            + "|"
            + base["Round"].astype(str).str.strip()
            + "|"
            + base[liga_col].astype(str).str.strip().str.upper()
            + "|"
            + base[home_col].astype(str).map(norm)
            + "|"
            + base[away_col].astype(str).map(norm)
        )
    else:
        base["_key"] = (
            base["Date"].astype(str).str.strip()
            + "|"
            + base[home_col].astype(str).map(norm)
            + "|"
            + base[away_col].astype(str).map(norm)
        )

    base = base.drop_duplicates(subset=["_key"], keep="first")
    base_map = base.set_index("_key")[["G_H_FT", "G_A_FT", "G_H_HT", "G_A_HT"]].to_dict("index")

    resolvidas = 0
    nao_achadas = 0
    mercados_nao_suportados = 0
    pendentes = 0
    pendentes_sem_cobertura = 0

    print("Liquidando apostas...")

    for idx, row in hist.iterrows():
        if str(row.get("resultado", "")).strip().lower() != "pendente":
            continue
        if args.date and str(row.get("data", "")).strip() != str(args.date):
            continue

        pendentes += 1

        data = str(row.get("data", "")).strip()
        if datas_sem_base and data not in datas_com_base:
            pendentes_sem_cobertura += 1
            continue

        jogo = str(row.get("jogo", "")).strip()
        mercado = str(row.get("mercado", "")).strip()

        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()

        if not home or not away:
            home, away = split_jogo(jogo)
        if not home or not away:
            nao_achadas += 1
            continue

        liga = str(row.get("liga", "")).strip().upper()
        round_ = str(row.get("Round", "")).strip()

        if liga_col:
            key = f"{data}|{round_}|{liga}|{norm(home)}|{norm(away)}"
        else:
            key = f"{data}|{norm(home)}|{norm(away)}"

        jb = base_map.get(key)

        if jb is None:
            print(f"SEM MATCH -> {key}")
            nao_achadas += 1
            continue

        ganhou = resolver_mercado(
            mercado,
            jb.get("G_H_FT", 0),
            jb.get("G_A_FT", 0),
            jb.get("G_H_HT", 0),
            jb.get("G_A_HT", 0),
        )

        if ganhou is None:
            mercados_nao_suportados += 1
            continue

        valor = float(row.get("valor_apostado", 0) or 0)
        odd = float(row.get("odd", 0) or 0)

        lucro = round(valor * (odd - 1), 2) if ganhou else round(-valor, 2)

        hist.at[idx, "resultado"] = "ganhou" if ganhou else "perdeu"
        hist.at[idx, "lucro"] = lucro
        resolvidas += 1

        if resolvidas % 1000 == 0:
            print(f"  Resolvidas: {resolvidas}")

    temp_historico = f"{HISTORICO}.{os.getpid()}.tmp"
    try:
        hist.to_csv(temp_historico, index=False, encoding="utf-8-sig")
        os.replace(temp_historico, HISTORICO)
    finally:
        if os.path.exists(temp_historico):
            os.remove(temp_historico)

    print("=" * 60)
    print("Settlement finalizado")
    print("=" * 60)
    print(f"Pendentes analisadas        : {pendentes}")
    print(f"Resolvidas                 : {resolvidas}")
    print(f"Pendentes sem cobertura     : {pendentes_sem_cobertura}")
    if datas_sem_base:
        print("Datas pendentes sem cobertura:")
        for data in datas_sem_base:
            print(f"  - {data}")
    print(f"Não achadas na base         : {nao_achadas}")
    print(f"Mercados não suportados     : {mercados_nao_suportados}")
    print("=" * 60)

    if resolvidas > 0 and not args.skip_post_update:
        print("Reconstruindo banca após settlement...")
        cmd = [sys.executable, os.path.join(BASE_DIR, "04_banca.py"), "--rebuild-bank"]
        result = subprocess.run(cmd, cwd=ROOT_DIR)
        if result.returncode != 0:
            return result.returncode

        # Aprendizado contínuo pós-settlement: sempre que novas apostas forem
        # liquidadas, atualiza os scores e relatórios contextuais usados pela banca
        # e pelo predict no próximo ciclo. Falhas aqui não invalidam o settlement.
        falhas_pos_settlement = []
        for script, titulo in [
            ("09_gerar_perfil_operacional_mercados.py", "perfil operacional de mercados"),
            ("10_gerar_scores_contexto.py", "scores contextuais"),
            ("11_relatorio_contexto_operacional.py", "relatório contextual operacional"),
        ]:
            script_path = os.path.join(BASE_DIR, script)
            if os.path.exists(script_path):
                print(f"Atualizando {titulo}...")
                post_result = subprocess.run([sys.executable, script_path], cwd=ROOT_DIR)
                if post_result.returncode != 0:
                    falhas_pos_settlement.append(f"{titulo}: exit {post_result.returncode}")
        if falhas_pos_settlement:
            print("AVISO: settlement concluído, mas houve falhas nas atualizações posteriores:")
            for falha in falhas_pos_settlement:
                print(f"  - {falha}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
