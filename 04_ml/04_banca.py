import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
import argparse
import subprocess
import webbrowser
from datetime import date, datetime
from urllib.parse import quote_plus

from utils.settlement_utils import resolver_resultado_mercado
from utils.prediction_paths import (
    HISTORICAL_PREDICTIONS_DIR,
    PREDICTIONS_DIR,
    historical_prediction_files,
    normal_prediction_files,
    normal_prediction_path,
    prediction_date_from_path,
)

import importlib.util

def _load_responsible_guard():
    guard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_responsible_mode.py")
    spec = importlib.util.spec_from_file_location("responsible_mode_guard", guard_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

responsible_guard = _load_responsible_guard()

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREVISOES_DIR = str(PREDICTIONS_DIR)
PREVISOES_HISTORICAS_DIR = str(HISTORICAL_PREDICTIONS_DIR)


ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from controles.mercados.status import is_mercado_allowed_by_lifecycle
from core.perfil_operacional import config_mercado, mercados_ativos
BANCA_DIR = os.path.join(BASE_DIR, "banca")
BANCA_FILE = os.path.join(BANCA_DIR, "banca_estado.json")
HISTORICO_FILE = os.path.join(BANCA_DIR, "historico_apostas.csv")
CALIBRATION_GUARD_SCRIPT = os.path.join(BASE_DIR, "09_calibration_guard.py")
CALIBRATION_GUARD_STATE = os.path.join(BASE_DIR, "reports", "estado_guard_calibracao.json")

os.makedirs(BANCA_DIR, exist_ok=True)

# Kelly fracionado. Mantém a fórmula de Kelly, mas com travas conservadoras.
KELLY_FRACAO = 0.25
ODD_PADRAO = 1.50

# Risco por aposta e por dia. Para banca de teste de R$ 100:
# - mínimo operacional: R$ 1,00 quando houver valor
# - máximo por aposta: 2% da banca
# - máximo do dia: 8% da banca
APOSTA_MIN_PCT = 0.005
APOSTA_MAX_PCT = 0.02
EXPOSICAO_MAX_DIA_PCT = 0.10

# Camada profissional de diversificação/classificacao.
# Evita concentrar a banca em muitas apostas parecidas no mesmo dia.
MAX_APOSTAS_DIA = 20
MAX_APOSTAS_POR_LIGA = 5
MAX_APOSTAS_POR_MERCADO = 20

# Pesos do score. O EV manda, mas confiança, ROI histórico e AUC ajudam no desempate.
PESO_EV = 1.00
PESO_CONFIANCA = 0.35
PESO_ROI_BT = 0.20
PESO_AUC = 0.10

# O predict já filtra odds fora da faixa. Aqui a banca só reforça segurança.
# As faixas oficiais vêm de 04_ml/reports/perfil_operacional_mercados.json.
ODD_MIN_OPERACIONAL = 1.35
ODD_MAX_OPERACIONAL = 3.50
ALERTA_DRAWDOWN = 0.20

# Trava de produção: mercado que perdeu dinheiro no histórico real deixa de entrar
# mesmo que o backtest/meta do modelo ainda esteja positivo.
MIN_ROI_PRODUCAO = -0.05
MIN_APOSTAS_ROI_PRODUCAO = 100

BASE_OFICIAL = os.path.join(ROOT_DIR, "data", "base_oficial.csv")

# ==============================
# MAPA DE MERCADOS → LÓGICA DE RESULTADO
# ==============================
# A lógica de settlement foi centralizada em utils/settlement_utils.py.
# Mantemos o nome resolver_resultado_mercado por compatibilidade com o menu.

# ==============================
# FUNÇÕES UTILITÁRIAS
# ==============================
def valor_ou_padrao(valor, padrao=0.0):
    if valor is None:
        return padrao
    if isinstance(valor, str):
        valor = valor.strip()
        if valor == "":
            return padrao
    try:
        if pd.isna(valor):
            return padrao
    except Exception:
        pass
    try:
        return float(valor)
    except Exception:
        return padrao


def texto_ou_padrao(valor, padrao=""):
    if valor is None:
        return padrao
    try:
        if pd.isna(valor):
            return padrao
    except Exception:
        pass
    valor = str(valor).strip()
    return valor if valor else padrao


def caminho_previsoes_do_dia(data_ref=None):
    return str(normal_prediction_path(data_ref or date.today()))


def extrair_data_previsao(path, fallback=None) -> str:
    """Extrai YYYY-MM-DD de previsoes_YYYY-MM-DD.csv."""
    return prediction_date_from_path(path, fallback=fallback or date.today())


def bool_col(valor) -> bool:
    """Converte flags vindas do CSV de previsão para booleano real."""
    if isinstance(valor, bool):
        return valor
    if valor is None:
        return False
    try:
        if pd.isna(valor):
            return False
    except Exception:
        pass
    return str(valor).strip().lower() in {"true", "1", "sim", "s", "yes", "y"}


def primeira_coluna_valida(row, colunas, padrao=0.0):
    """Retorna o primeiro valor numérico válido encontrado na linha."""
    for col in colunas:
        if col in row.index:
            valor = valor_ou_padrao(row.get(col), None)
            if valor is not None:
                return valor
    return padrao


def formatar_data_pesquisa(data_valor):
    texto = str(data_valor or "").strip()
    try:
        return datetime.strptime(texto, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        pass
    try:
        return datetime.strptime(texto, "%d/%m/%Y").strftime("%d/%m/%Y")
    except Exception:
        return texto


def url_pesquisa_resultado(aposta):
    home = str(aposta.get("home", "") or "").strip()
    away = str(aposta.get("away", "") or "").strip()
    data = formatar_data_pesquisa(aposta.get("data", ""))
    if not home or not away or not data:
        return ""
    query = quote_plus(f"{home} {away} {data}")
    return f"https://www.google.com/search?q={query}"


def abrir_pesquisa_resultado(aposta):
    url = url_pesquisa_resultado(aposta)
    if not url:
        print("  ⚠️  Pesquisa indisponível: home, away ou data ausente.")
        return False
    print(f"  Abrindo pesquisa: {url}")
    webbrowser.open_new_tab(url)
    return True


def montar_nome_jogo(row) -> str:
    """Monta nome do jogo a partir das colunas atuais do predict."""
    jogo = texto_ou_padrao(row.get("jogo"), "")
    if jogo:
        return jogo
    home = texto_ou_padrao(row.get("home"), "") or texto_ou_padrao(row.get("Home"), "")
    away = texto_ou_padrao(row.get("away"), "") or texto_ou_padrao(row.get("Away"), "")
    if home or away:
        return f"{home} x {away}".strip()
    return "Jogo não informado"


def data_da_previsao(row, fallback=None) -> str:
    fallback = fallback or str(date.today())
    return texto_ou_padrao(row.get("data"), fallback)


def calcular_score_banca(row) -> float:
    """Score profissional para priorizar apostas quando há limite diário.

    O objetivo não é prever resultado individual, mas escolher quais sinais ocuparão
    a exposição limitada do dia. EV é o fator principal; confiança, ROI de backtest
    e AUC entram como desempate/qualidade do sinal.
    """
    ev = max(0.0, valor_ou_padrao(row.get("ev"), 0.0))
    confianca = max(0.0, valor_ou_padrao(row.get("prob_banca"), valor_ou_padrao(row.get("confianca"), 0.0)))
    roi_bt = max(0.0, valor_ou_padrao(row.get("roi_bt"), 0.0))
    auc = max(0.0, valor_ou_padrao(row.get("auc"), 0.0))

    # score_operacional do predict continua sendo aproveitado quando existir,
    # mas não pode dominar sozinho se vier vazio/baixo.
    score_predict = max(0.0, valor_ou_padrao(row.get("score_operacional"), 0.0))

    score = (
        (ev * PESO_EV)
        + (confianca * PESO_CONFIANCA)
        + (roi_bt * PESO_ROI_BT)
        + (auc * PESO_AUC)
    )
    return round(max(score, score_predict), 8)


def aplicar_politica_profissional(df: pd.DataFrame, banca_atual: float) -> pd.DataFrame:
    """Aplica classificacao + limite diário + diversificação.

    Regras:
    - só considera status=disponivel;
    - ordena por score_banca desc;
    - respeita exposição máxima diária;
    - respeita máximo de apostas no dia;
    - respeita máximo por liga;
    - respeita máximo por mercado.

    As linhas bloqueadas continuam no CSV/tela com motivo claro para auditoria.
    """
    if df.empty or banca_atual <= 0:
        return df

    df = df.copy()
    limite_dia = round(float(banca_atual) * EXPOSICAO_MAX_DIA_PCT, 2)
    if limite_dia <= 0:
        return df

    df["score_banca"] = df.apply(calcular_score_banca, axis=1)

    candidatas = df[df["status"].eq("disponivel")].sort_values(
        ["score_banca", "ev", "prob_banca", "odd_usada"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )

    acumulado = 0.0
    total_apostas = 0
    cont_liga = {}
    cont_mercado = {}
    manter = set()

    for idx, row in candidatas.iterrows():
        valor = valor_ou_padrao(row.get("valor_aposta"), 0.0)
        liga = texto_ou_padrao(row.get("liga"), "")
        mercado = texto_ou_padrao(row.get("mercado"), "")

        if valor <= 0:
            df.at[idx, "status"] = "sem_valor"
            continue

        if total_apostas >= MAX_APOSTAS_DIA:
            df.at[idx, "status"] = "limite_qtd_apostas_dia"
            df.at[idx, "valor_aposta"] = 0.0
            df.at[idx, "kelly_pct"] = 0.0
            continue

        if acumulado + valor > limite_dia + 1e-9:
            df.at[idx, "status"] = "limite_exposicao_dia"
            df.at[idx, "valor_aposta"] = 0.0
            df.at[idx, "kelly_pct"] = 0.0
            continue

        if cont_liga.get(liga, 0) >= MAX_APOSTAS_POR_LIGA:
            df.at[idx, "status"] = "limite_liga_dia"
            df.at[idx, "valor_aposta"] = 0.0
            df.at[idx, "kelly_pct"] = 0.0
            continue

        if cont_mercado.get(mercado, 0) >= MAX_APOSTAS_POR_MERCADO:
            df.at[idx, "status"] = "limite_mercado_dia"
            df.at[idx, "valor_aposta"] = 0.0
            df.at[idx, "kelly_pct"] = 0.0
            continue

        manter.add(idx)
        acumulado += valor
        total_apostas += 1
        cont_liga[liga] = cont_liga.get(liga, 0) + 1
        cont_mercado[mercado] = cont_mercado.get(mercado, 0) + 1

    df["selecionada_banca"] = df.index.isin(manter)
    return df

def verificar_guard_calibracao(bloquear=True):
    """Executa o guard de calibração antes de registrar apostas automáticas."""
    if not os.path.exists(CALIBRATION_GUARD_SCRIPT):
        return True, "guard_nao_encontrado"

    try:
        result = subprocess.run(
            [sys.executable, CALIBRATION_GUARD_SCRIPT],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        if bloquear:
            return False, f"erro_ao_executar_guard:{exc}"
        return True, f"erro_ao_executar_guard_ignorado:{exc}"

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())

    if result.returncode == 2:
        return False, "calibracao_bloqueada"
    if result.returncode != 0 and bloquear:
        return False, f"guard_exit_code_{result.returncode}"
    return True, "calibracao_ok"


def calcular_roi_real_por_mercado(historico: pd.DataFrame | None = None) -> dict:
    """Calcula ROI real por mercado usando apenas apostas liquidadas."""
    if historico is None:
        historico = carregar_historico()
    if historico.empty:
        return {}

    df = historico.copy()
    df["resultado"] = df["resultado"].astype(str).str.strip().str.lower()
    df = df[df["resultado"].isin(["ganhou", "perdeu"])]
    if df.empty:
        return {}

    df["valor_apostado"] = pd.to_numeric(df["valor_apostado"], errors="coerce").fillna(0)
    df["lucro"] = pd.to_numeric(df["lucro"], errors="coerce").fillna(0)

    resumo = {}
    for mercado, grp in df.groupby(df["mercado"].astype(str).str.strip()):
        stake = float(grp["valor_apostado"].sum())
        lucro = float(grp["lucro"].sum())
        bets = int(len(grp))
        roi = (lucro / stake) if stake > 0 else 0.0
        resumo[mercado] = {"bets": bets, "stake": stake, "lucro": lucro, "roi": roi}
    return resumo


def mercado_aprovado_por_roi_real(mercado, resumo_roi=None):
    """Bloqueia mercados com amostra real suficiente e ROI abaixo do mínimo."""
    resumo_roi = resumo_roi or calcular_roi_real_por_mercado()
    info = resumo_roi.get(str(mercado).strip())
    if not info:
        return True, "sem_historico_producao"
    if info["bets"] < MIN_APOSTAS_ROI_PRODUCAO:
        return True, "amostra_producao_insuficiente"
    if info["roi"] < MIN_ROI_PRODUCAO:
        return False, f"roi_real_mercado_baixo_{info['roi']:+.1%}_bets_{info['bets']}"
    return True, "roi_real_ok"


def aposta_ja_registrada(jogo, mercado, event, data_ref=None):
    historico = carregar_historico()
    if historico.empty:
        return False
    data_ref = data_ref or str(date.today())
    filtro = (
        historico["data"].astype(str).eq(str(data_ref))
        & historico["jogo"].astype(str).eq(str(jogo))
        & historico["mercado"].astype(str).eq(str(mercado))
        & historico["event"].astype(str).eq(str(event))
    )
    return filtro.any()


# ==============================
# FUNÇÕES DE BANCA
# ==============================
def carregar_estado():
    if os.path.exists(BANCA_FILE):
        with open(BANCA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def salvar_estado(estado):
    os.makedirs(os.path.dirname(BANCA_FILE), exist_ok=True)
    temp_path = f"{BANCA_FILE}.{os.getpid()}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, BANCA_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def inicializar_banca(banca_inicial):
    estado = {
        "banca_inicial": banca_inicial,
        "banca_atual": banca_inicial,
        "banca_pico": banca_inicial,
        "total_apostas": 0,
        "total_ganhos": 0,
        "total_perdas": 0,
        "lucro_total": 0.0,
        "roi_total": 0.0,
        "data_inicio": str(date.today()),
        "ultima_atualizacao": str(date.today()),
    }
    salvar_estado(estado)
    return estado


def calcular_kelly(prob, odd, fracao=KELLY_FRACAO):
    if odd <= 1:
        return 0.0
    kelly = (prob * odd - 1) / (odd - 1)
    if kelly <= 0:
        return 0.0
    return kelly * fracao


def calcular_aposta(banca_atual, prob, odd, fracao=KELLY_FRACAO):
    kelly_pct = calcular_kelly(prob, odd, fracao)
    if kelly_pct <= 0:
        return 0.0, 0.0
    kelly_pct = max(APOSTA_MIN_PCT, min(APOSTA_MAX_PCT, kelly_pct))
    valor = banca_atual * kelly_pct
    return round(valor, 2), round(kelly_pct * 100, 2)


def carregar_historico():
    """Carrega o histórico de apostas com tratamento de CSV corrompido."""
    colunas_base = [
        "data", "jogo", "liga", "mercado", "event",
        "prob_modelo", "confianca", "odd", "valor_apostado",
        "kelly_pct", "roi_bt", "resultado", "lucro", "banca_apos"
    ]

    if not os.path.exists(HISTORICO_FILE) or os.path.getsize(HISTORICO_FILE) == 0:
        return pd.DataFrame(columns=colunas_base)

    # Detecta separador automaticamente e tolera BOM.
    with open(HISTORICO_FILE, "r", encoding="utf-8-sig", errors="replace") as f:
        primeira_linha = f.readline()
    sep = ";" if ";" in primeira_linha and primeira_linha.count(";") > primeira_linha.count(",") else ","

    try:
        df = pd.read_csv(HISTORICO_FILE, sep=sep, encoding="utf-8-sig", low_memory=False)
    except Exception:
        df = pd.read_csv(HISTORICO_FILE, sep=sep, encoding="latin-1", low_memory=False)

    if df.empty:
        return pd.DataFrame(columns=colunas_base)

    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [str(c).strip() for c in df.columns]

    if "data" in df.columns:
        df = df[df["data"].astype(str).str.strip().str.lower() != "data"].reset_index(drop=True)

    df = df.dropna(how="all").reset_index(drop=True)

    for col in colunas_base:
        if col not in df.columns:
            df[col] = np.nan

    return df

def salvar_historico(df):
    """Salva o histórico sempre limpo, sem duplicatas de header."""
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how='all').reset_index(drop=True)
    os.makedirs(os.path.dirname(HISTORICO_FILE), exist_ok=True)
    temp_path = f"{HISTORICO_FILE}.{os.getpid()}.tmp"
    try:
        df.to_csv(temp_path, index=False, sep=',', encoding='utf-8-sig')
        os.replace(temp_path, HISTORICO_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def registrar_aposta(jogo, liga, mercado, event, prob, confianca, odd, valor, kelly_pct, roi_bt, data_ref=None):
    historico = carregar_historico()
    estado = carregar_estado()

    lifecycle_allowed, status_ciclo_vida, lifecycle_motivo = is_mercado_allowed_by_lifecycle(str(mercado))
    if not lifecycle_allowed:
        print(f"  🛑 Mercado bloqueado por lifecycle: {mercado} | {status_ciclo_vida} | {lifecycle_motivo}")
        return False

    cfg_mercado = config_mercado(mercado)
    if not cfg_mercado["ativo"]:
        print(f"  🛑 Mercado inativo no perfil operacional: {mercado} | {cfg_mercado.get('status')} | {cfg_mercado.get('motivo')}")
        return False

    if odd < cfg_mercado["odd_min"] or odd > cfg_mercado["odd_max"]:
        print(f"  🛑 Odd fora da faixa operacional: {mercado} | odd={odd:.2f} | faixa={cfg_mercado['odd_min']:.2f}-{cfg_mercado['odd_max']:.2f}")
        return False

    data_ref = str(data_ref or date.today())

    if aposta_ja_registrada(jogo, mercado, event, data_ref=data_ref):
        print(f"  ⚠️  Aposta já registrada em {data_ref}: {jogo} | {mercado} | {event}")
        return False

    nova = {
        "data": data_ref,
        "jogo": jogo,
        "liga": liga,
        "mercado": mercado,
        "event": event,
        "prob_modelo": prob,
        "confianca": confianca,
        "odd": odd,
        "valor_apostado": valor,
        "kelly_pct": kelly_pct,
        "roi_bt": roi_bt,
        "resultado": "pendente",
        "lucro": 0.0,
        "banca_apos": estado["banca_atual"],
    }

    nova_df = pd.DataFrame([nova])
    if historico.empty:
        historico = nova_df
    else:
        historico = pd.concat([historico.dropna(how="all"), nova_df], ignore_index=True)
    salvar_historico(historico)
    print(f"  ✅ Aposta registrada: {jogo} | {mercado} | R$ {valor:.2f}")
    return True


def atualizar_resultado(idx, ganhou):
    historico = carregar_historico()
    estado = carregar_estado()

    if idx >= len(historico):
        print(f"❌ Índice {idx} não encontrado no histórico.")
        return

    aposta = historico.iloc[idx]

    if aposta["resultado"] != "pendente":
        print(f"⚠️  Aposta {idx} já foi atualizada: {aposta['resultado']}")
        return

    try:
        valor = float(aposta["valor_apostado"])
        odd = float(aposta["odd"])
    except Exception:
        print(f"❌ Erro nos dados da aposta {idx}")
        return

    if ganhou:
        lucro = round(valor * (odd - 1), 2)
        resultado = "ganhou"
        estado["total_ganhos"] += 1
    else:
        lucro = round(-valor, 2)
        resultado = "perdeu"
        estado["total_perdas"] += 1

    estado["banca_atual"] = round(estado["banca_atual"] + lucro, 2)
    estado["lucro_total"] = round(estado["lucro_total"] + lucro, 2)
    estado["total_apostas"] += 1
    estado["ultima_atualizacao"] = str(date.today())

    if estado["banca_atual"] > estado["banca_pico"]:
        estado["banca_pico"] = estado["banca_atual"]

    historico.at[idx, "resultado"] = resultado
    historico.at[idx, "lucro"] = lucro
    historico.at[idx, "banca_apos"] = estado["banca_atual"]

    salvar_historico(historico)

    historico_finalizado = historico[historico["resultado"] != "pendente"]
    total_stake = historico_finalizado["valor_apostado"].sum()
    estado["roi_total"] = round(
        (estado["lucro_total"] / total_stake) * 100, 2
    ) if total_stake > 0 else 0

    salvar_estado(estado)

    print(
        f"  {'✅' if ganhou else '❌'} "
        f"[{aposta['mercado']}] {aposta['jogo']} → {resultado.upper()} | "
        f"Lucro: R$ {lucro:+.2f} | Banca: R$ {estado['banca_atual']:.2f}"
    )

    drawdown = (
        (estado["banca_pico"] - estado["banca_atual"]) / estado["banca_pico"]
    ) if estado["banca_pico"] > 0 else 0

    if drawdown >= ALERTA_DRAWDOWN:
        print(f"\n  🚨 ALERTA: Drawdown de {drawdown:.1%} — considere reduzir o risco!")


# ==============================
# ATUALIZAÇÃO MANUAL DE RESULTADO
# ==============================

def menu_auto_resultado():
    """
    Mantém compatibilidade com chamadas antigas do menu, mas não executa
    scraping externo. O legado de busca automática foi removido; resultados
    devem ser atualizados manualmente até existir uma fonte oficial FutPython
    para placares encerrados.
    """
    print("\n  ⚠️  Busca automática de resultados removida nesta versão.")
    print("  Atualize os resultados manualmente pela opção de atualização de aposta.")


# ==============================
# DASHBOARD
# ==============================
def mostrar_dashboard():
    estado = carregar_estado()
    historico = carregar_historico()

    print("\n" + "=" * 55)
    print("💰 PAINEL DA BANCA")
    print("=" * 55)

    print(f"\n  📅 Início          : {estado['data_inicio']}")
    print(f"  📅 Última atualiz. : {estado['ultima_atualizacao']}")
    print(f"\n  💵 Banca inicial   : R$ {estado['banca_inicial']:.2f}")
    print(f"  💵 Banca atual     : R$ {estado['banca_atual']:.2f}")
    print(f"  📈 Banca no pico   : R$ {estado['banca_pico']:.2f}")

    variacao = ((estado["banca_atual"] / estado["banca_inicial"]) - 1) * 100
    print(f"  📊 Variação total  : {variacao:+.2f}%")

    drawdown = (estado["banca_pico"] - estado["banca_atual"]) / estado["banca_pico"] * 100
    print(f"  📉 Drawdown atual  : {drawdown:.2f}%")

    print(f"\n  🎯 Total apostas   : {estado['total_apostas']}")
    print(f"  ✅ Ganhos          : {estado['total_ganhos']}")
    print(f"  ❌ Perdas          : {estado['total_perdas']}")

    if estado["total_apostas"] > 0:
        winrate = estado["total_ganhos"] / estado["total_apostas"] * 100
        print(f"  🎯 Winrate         : {winrate:.1f}%")
        print(f"  📊 ROI geral       : {estado['roi_total']:+.2f}%")
        print(f"  💰 Lucro total     : R$ {estado['lucro_total']:+.2f}")

    if len(historico) > 0:
        pendentes = historico[historico["resultado"] == "pendente"]
        if len(pendentes) > 0:
            print(f"\n  ⏳ Apostas pendentes ({len(pendentes)}):")
            for idx, row in pendentes.iterrows():
                print(
                    f"     [{idx}] {row['jogo']} | {row['mercado']} | "
                    f"R$ {float(row['valor_apostado']):.2f} | {row['data']}"
                )

    if len(historico) > 0:
        finalizadas = historico[historico["resultado"] != "pendente"]
        if len(finalizadas) > 0:
            print(f"\n  📊 Performance por mercado:")
            for mercado, grp in finalizadas.groupby("mercado"):
                ganhos = (grp["resultado"] == "ganhou").sum()
                lucro = grp["lucro"].sum()
                wr = ganhos / len(grp) * 100
                print(f"     {mercado:<12} | {len(grp):>3} apostas | WR={wr:.0f}% | Lucro=R$ {lucro:+.2f}")

    print("=" * 55)


# ==============================
# PREVISÕES DO DIA
# ==============================
def preparar_previsoes(previsoes_path, data_ref=None, usar_roi_real=True):
    estado = carregar_estado()
    data_ref = str(data_ref or extrair_data_previsao(previsoes_path))
    if estado is None:
        print("❌ Banca ainda não inicializada.")
        return None

    if not os.path.exists(previsoes_path):
        print(f"❌ Arquivo de previsões não encontrado: {previsoes_path}")
        return None

    df = pd.read_csv(previsoes_path, encoding="utf-8-sig").copy()

    if df.empty:
        print("⚠️  Arquivo de previsões está vazio.")
        return None

    # Compatibilidade com o CSV novo do 03_predict.py:
    # home/away, prob_evento, prob_modelo, odd, apostar, motivo_nao_apostar.
    if "apostar" in df.columns:
        df = df[df["apostar"].apply(bool_col)].copy()
    
    ativos = set(mercados_ativos().keys())
    df = df[df["mercado"].astype(str).str.upper().isin(ativos)].copy()

    # A banca não revalida jogos contra base_oficial.
    # O predict já validou o arquivo daily e a revalidação removia apostas válidas por divergência de nomes/datas.
    if df.empty:
        print("⚠️  Nenhuma linha com apostar=True no arquivo de previsões.")
        return None

    df["jogo"] = df.apply(montar_nome_jogo, axis=1)

    for col in ["liga", "mercado", "event"]:
        if col not in df.columns:
            df[col] = ""

    # Probabilidade correta para banca: prob_evento/confianca.
    # prob_modelo pode ser a classe bruta e, em alguns mercados, ficar abaixo de 0.5.
    df["prob_banca"] = df.apply(
        lambda row: primeira_coluna_valida(row, ["prob_evento", "confianca", "prob_modelo", "prob", "prob_sim"], 0.0),
        axis=1,
    )
    df["prob_modelo_banca"] = df.apply(
        lambda row: primeira_coluna_valida(row, ["prob_modelo", "prob", "prob_evento", "confianca", "prob_sim"], 0.0),
        axis=1,
    )
    df["roi_bt"] = df.apply(lambda row: primeira_coluna_valida(row, ["roi_bt"], 0.0), axis=1)
    df["ev"] = df.apply(lambda row: primeira_coluna_valida(row, ["ev"], 0.0), axis=1)
    df["odd_usada"] = df.apply(lambda row: primeira_coluna_valida(row, ["odd", "odd_real"], ODD_PADRAO), axis=1)
    df.loc[df["odd_usada"] <= 1.0, "odd_usada"] = ODD_PADRAO

    df["valor_aposta"] = 0.0
    df["kelly_pct"] = 0.0
    df["status"] = "sem_valor"

    valores, percentuais, status = [], [], []
    resumo_roi_real = calcular_roi_real_por_mercado() if usar_roi_real else {}

    for _, row in df.iterrows():
        jogo = texto_ou_padrao(row.get("jogo"))
        mercado = texto_ou_padrao(row.get("mercado"))
        event = texto_ou_padrao(row.get("event"))
        prob = valor_ou_padrao(row.get("prob_banca"), 0.0)
        odd = valor_ou_padrao(row.get("odd_usada"), ODD_PADRAO)
        ev = valor_ou_padrao(row.get("ev"), 0.0)
        context_ok_pred = bool_col(row.get("context_operacional_ok")) if "context_operacional_ok" in row.index else True
        context_motivo_pred = texto_ou_padrao(row.get("context_block_motivo"), "")

        valor, kelly_pct = calcular_aposta(estado["banca_atual"], prob, odd)

        cfg_mercado = config_mercado(mercado)
        if not cfg_mercado["ativo"]:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append("mercado_inativo_perfil")
            continue

        odd_min_mercado = float(cfg_mercado["odd_min"])
        odd_max_mercado = float(cfg_mercado["odd_max"])
        min_ev_mercado = float(cfg_mercado["min_ev"])

        if odd < odd_min_mercado or odd > odd_max_mercado:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append(f"odd_fora_da_faixa_{odd_min_mercado:.2f}_{odd_max_mercado:.2f}")
            continue

        if ev <= min_ev_mercado:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append("ev_abaixo_do_minimo")
            continue

        if not context_ok_pred:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append(context_motivo_pred or "contexto_operacional_reprovado")
            continue

        mercado_ok, motivo_roi_real = mercado_aprovado_por_roi_real(mercado, resumo_roi_real) if usar_roi_real else (True, "roi_real_ignorado")
        if not mercado_ok:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append(motivo_roi_real)
            continue

        if valor <= 0:
            valores.append(0.0)
            percentuais.append(0.0)
            status.append("sem_valor")
            continue

        if aposta_ja_registrada(jogo, mercado, event, data_ref=data_ref):
            valores.append(0.0)
            percentuais.append(0.0)
            status.append("ja_registrada")
            continue

        valores.append(valor)
        percentuais.append(kelly_pct)
        status.append("disponivel")

    df["valor_aposta"] = valores
    df["kelly_pct"] = percentuais
    df["status"] = status
    df = aplicar_politica_profissional(df, estado["banca_atual"])

    # Ordena a tela e os índices por prioridade operacional.
    # Primeiro aparecem as apostas realmente disponíveis, priorizadas pelo maior score/EV;
    # depois vêm as linhas bloqueadas, apenas para auditoria.
    if not df.empty:
        if "score_banca" not in df.columns:
            df["score_banca"] = pd.to_numeric(df.get("score_operacional", 0), errors="coerce").fillna(0)
            if "ev" in df.columns:
                ev_num = pd.to_numeric(df["ev"], errors="coerce").fillna(0)
                df["score_banca"] = df["score_banca"].where(df["score_banca"] != 0, ev_num)

        df["prioridade_status"] = np.where(df["status"].eq("disponivel"), 0, 1)
        df = (
            df.sort_values(
                ["prioridade_status", "score_banca", "ev"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .drop(columns=["prioridade_status"], errors="ignore")
            .reset_index(drop=True)
        )

    return df.reset_index(drop=True)


def calcular_apostas_do_dia(previsoes_path, mostrar_tela=True, data_ref=None, usar_roi_real=True):
    estado = carregar_estado()
    data_ref = str(data_ref or extrair_data_previsao(previsoes_path))
    df = preparar_previsoes(previsoes_path, data_ref=data_ref, usar_roi_real=usar_roi_real)

    if df is None:
        return None

    if mostrar_tela:
        print("\n" + "=" * 55)
        print(f"💡 APOSTAS RECOMENDADAS — {data_ref}")
        print(
            f"   Banca atual: R$ {estado['banca_atual']:.2f} | "
            f"Kelly: {int(KELLY_FRACAO*100)}% | "
            f"Máx/aposta: {APOSTA_MAX_PCT:.1%} | Máx/dia: {EXPOSICAO_MAX_DIA_PCT:.1%} | Máx apostas: {MAX_APOSTAS_DIA}"
        )
        print("=" * 55)

        total_valor = 0.0

        for idx, row in df.iterrows():
            prob = valor_ou_padrao(row.get("prob_modelo_banca"), 0.0)
            confianca = valor_ou_padrao(row.get("prob_banca"), 0.0)
            mercado = texto_ou_padrao(row.get("mercado"))
            event = texto_ou_padrao(row.get("event"))
            jogo = texto_ou_padrao(row.get("jogo"))
            liga = texto_ou_padrao(row.get("liga"))
            roi_bt = valor_ou_padrao(row.get("roi_bt"), 0.0)
            odd = valor_ou_padrao(row.get("odd_usada"), ODD_PADRAO)
            ev = valor_ou_padrao(row.get("ev"), 0.0)
            valor = valor_ou_padrao(row.get("valor_aposta"), 0.0)
            kelly_pct = valor_ou_padrao(row.get("kelly_pct"), 0.0)
            status = texto_ou_padrao(row.get("status"))
            motivo_predict = texto_ou_padrao(row.get("motivo_nao_apostar"), "ok")

            print(f"\n  [{idx}] ⚽ {jogo}")
            print(f"      Liga      : {liga}")
            print(f"      Mercado   : {mercado} ({event})")
            print(f"      Confiança : {confianca:.1%} | Prob modelo: {prob:.1%} | EV: {ev:+.3f}")
            print(f"      ROI BT    : {roi_bt:+.1%} | Odd real: {odd:.2f}")
            score_banca = valor_ou_padrao(row.get("score_banca"), 0.0)
            context_score = valor_ou_padrao(row.get("context_score_final"), 0.0)
            print(f"      Score     : {score_banca:.4f} | Contexto: {context_score:.1f}")

            if status != "disponivel":
                print(f"      Status    : {status}")
                if motivo_predict and motivo_predict != "ok":
                    print(f"      Predict   : {motivo_predict}")
                continue

            total_valor += valor
            print(f"      Kelly     : {kelly_pct:.2f}% da banca")
            print(f"      💰 APOSTAR: R$ {valor:.2f}")

        print(f"\n  {'=' * 50}")
        print(f"  💰 Total a apostar hoje : R$ {total_valor:.2f}")
        print(f"  📊 % da banca           : {(total_valor / estado['banca_atual'] * 100) if estado['banca_atual'] > 0 else 0:.1f}%")
        print(f"  🧯 Limite diário        : {EXPOSICAO_MAX_DIA_PCT:.1%} da banca")
        print(f"  🎯 Máx apostas/dia     : {MAX_APOSTAS_DIA}")
        print(f"  🏟️ Máx por liga        : {MAX_APOSTAS_POR_LIGA}")
        print(f"  📌 Máx por mercado     : {MAX_APOSTAS_POR_MERCADO}")

        if estado["banca_atual"] > 0 and total_valor / estado["banca_atual"] > EXPOSICAO_MAX_DIA_PCT + 1e-9:
            print("  🚨 ATENÇÃO: exposição diária acima do limite configurado!")

        print("=" * 55)

    return df

def registrar_apostas_automaticas(previsoes_path, selecao="todas", data_ref=None):
    responsible_guard.assert_auto_fill_allowed("auto_fill_today")
    data_ref = str(data_ref or extrair_data_previsao(previsoes_path))
    df = calcular_apostas_do_dia(previsoes_path, mostrar_tela=False, data_ref=data_ref)

    if df is None or df.empty:
        return

    registradas = 0
    ignoradas = 0
    indices = df.index.tolist() if selecao == "todas" else selecao

    print("\n" + "=" * 55)
    print("🤖 PREENCHIMENTO AUTOMÁTICO DAS APOSTAS")
    print("=" * 55)

    for idx in indices:
        if idx not in df.index:
            print(f"  ⚠️  Índice inválido: {idx}")
            ignoradas += 1
            continue

        row = df.loc[idx]

        if texto_ou_padrao(row["status"]) != "disponivel":
            print(f"  ⏭️  [{idx}] Ignorada — status: {row['status']}")
            ignoradas += 1
            continue

        sucesso = registrar_aposta(
            jogo=texto_ou_padrao(row["jogo"]),
            liga=texto_ou_padrao(row["liga"]),
            mercado=texto_ou_padrao(row["mercado"]),
            event=texto_ou_padrao(row["event"]),
            prob=valor_ou_padrao(row["prob_modelo_banca"], 0.0),
            confianca=valor_ou_padrao(row["prob_banca"], 0.0),
            odd=valor_ou_padrao(row["odd_usada"], ODD_PADRAO),
            valor=valor_ou_padrao(row["valor_aposta"], 0.0),
            kelly_pct=valor_ou_padrao(row["kelly_pct"], 0.0),
            roi_bt=valor_ou_padrao(row["roi_bt"], 0.0),
            data_ref=data_ref,
        )

        if sucesso:
            registradas += 1
        else:
            ignoradas += 1

    print("\n" + "=" * 55)
    print(f"✅ Registradas : {registradas}")
    print(f"⏭️  Ignoradas   : {ignoradas}")
    print("=" * 55)


def menu_apostas_do_dia(estado):
    previsoes_path = caminho_previsoes_do_dia()
    df = calcular_apostas_do_dia(previsoes_path, mostrar_tela=True)

    if df is None or df.empty:
        return

    print("\n  O que deseja fazer?")
    print("  1. Só visualizar")
    print("  2. Preencher automaticamente todas as disponíveis")
    print("  3. Preencher automaticamente escolhendo os índices")
    print("  4. Voltar")

    escolha = input("  Escolha: ").strip()

    if escolha == "1":
        return

    if escolha == "2":
        confirmar = input("  Confirmar registro automático de todas as disponíveis? (s/n): ").strip().lower()
        if confirmar == "s":
            registrar_apostas_automaticas(previsoes_path, selecao="todas")
        return

    if escolha == "3":
        entrada = input("  Digite os índices separados por vírgula (ex: 0,2,5): ").strip()
        if not entrada:
            print("  ⚠️  Nenhum índice informado.")
            return
        try:
            indices = [int(x.strip()) for x in entrada.split(",") if x.strip() != ""]
        except ValueError:
            print("  ❌ Índices inválidos.")
            return
        confirmar = input("  Confirmar registro automático dos índices informados? (s/n): ").strip().lower()
        if confirmar == "s":
            registrar_apostas_automaticas(previsoes_path, selecao=indices)
        return

    if escolha == "4":
        return

    print("  ⚠️  Opção inválida.")


# ==============================
# PREVISÕES ANTERIORES
# ==============================
def listar_previsoes_anteriores():
    """Lista previsões anteriores em 04_ml/previsoes e 04_ml/previsoes_historicas."""
    encontrados = {}

    for path in historical_prediction_files():
        data_str = prediction_date_from_path(path)
        if data_str != str(date.today()):
            encontrados[data_str] = str(path)

    for path in normal_prediction_files(include_legacy=True):
        data_str = prediction_date_from_path(path)
        if data_str != str(date.today()):
            encontrados[data_str] = str(path)

    return sorted(encontrados.items(), reverse=True)

def registrar_apostas_dia_anterior(path, data_str):
    responsible_guard.assert_auto_fill_allowed("auto_fill_previous")
    """
    Registra apostas de um arquivo de previsões anterior no histórico,
    usando a data do arquivo (não a data de hoje) e respeitando o CSV novo do predict.
    """
    estado = carregar_estado()

    if not os.path.exists(path):
        print(f"❌ Arquivo não encontrado: {path}")
        return 0

    df = preparar_previsoes(path, data_ref=data_str, usar_roi_real=False)
    if df is None or df.empty:
        print("⚠️  Nenhuma aposta disponível para registrar neste arquivo.")
        return 0

    data_registro = data_str
    historico = carregar_historico()
    registradas = 0
    ignoradas = 0

    print(f"\n  Registrando apostas de {data_str}...\n")

    for _, row in df.iterrows():
        if texto_ou_padrao(row.get("status")) != "disponivel":
            ignoradas += 1
            continue

        jogo = texto_ou_padrao(row.get("jogo"))
        liga = texto_ou_padrao(row.get("liga"))
        mercado = texto_ou_padrao(row.get("mercado"))
        event = texto_ou_padrao(row.get("event"))
        prob = valor_ou_padrao(row.get("prob_modelo_banca"), 0.0)
        confianca = valor_ou_padrao(row.get("prob_banca"), 0.0)
        odd = valor_ou_padrao(row.get("odd_usada"), ODD_PADRAO)
        roi_bt = valor_ou_padrao(row.get("roi_bt"), 0.0)
        valor = valor_ou_padrao(row.get("valor_aposta"), 0.0)
        kelly_pct = valor_ou_padrao(row.get("kelly_pct"), 0.0)

        filtro = (
            historico["data"].astype(str).eq(data_registro)
            & historico["jogo"].astype(str).eq(jogo)
            & historico["mercado"].astype(str).eq(mercado)
            & historico["event"].astype(str).eq(event)
        )
        if filtro.any():
            print(f"  ⚠️  Já registrada: {jogo} | {mercado}")
            ignoradas += 1
            continue

        nova = {
            "data": data_registro,
            "jogo": jogo,
            "liga": liga,
            "mercado": mercado,
            "event": event,
            "prob_modelo": prob,
            "confianca": confianca,
            "odd": odd,
            "valor_apostado": valor,
            "kelly_pct": kelly_pct,
            "roi_bt": roi_bt,
            "resultado": "pendente",
            "lucro": 0.0,
            "banca_apos": estado["banca_atual"],
        }

        historico = pd.concat([historico, pd.DataFrame([nova])], ignore_index=True)
        print(f"  ✅ {jogo} | {mercado} | Odd {odd:.2f} | R$ {valor:.2f}")
        registradas += 1

    salvar_historico(historico)

    print(f"\n  ✅ Registradas : {registradas}")
    print(f"  ⏭️  Ignoradas   : {ignoradas}")
    return registradas

def menu_apostas_anteriores(estado):
    arquivos = listar_previsoes_anteriores()

    if not arquivos:
        print("\n  ⚠️  Nenhum arquivo de previsões anteriores encontrado.")
        return

    print("\n" + "=" * 55)
    print("📂 PREVISÕES ANTERIORES DISPONÍVEIS")
    print("=" * 55)
    for i, (data_str, _) in enumerate(arquivos):
        print(f"  [{i}] {data_str}")
    print("=" * 55)

    try:
        escolha = int(input("  Escolha o índice do dia: ").strip())
        if escolha < 0 or escolha >= len(arquivos):
            print("  ❌ Índice inválido.")
            return
    except ValueError:
        print("  ❌ Entrada inválida.")
        return

    data_str, path = arquivos[escolha]
    df = calcular_apostas_do_dia(path, mostrar_tela=True, data_ref=data_str, usar_roi_real=False)

    if df is None or df.empty:
        return

    # Verifica quantas já estão registradas
    historico = carregar_historico()
    # O histórico usa ISO (YYYY-MM-DD), igual ao nome do arquivo previsoes_YYYY-MM-DD.csv.
    data_registro = data_str

    ja_registradas = historico[historico["data"].astype(str).eq(data_registro)]

    print("\n  O que deseja fazer?")
    print("  1. Só visualizar")
    print("  2. Registrar apostas deste dia no histórico")
    print("  3. Atualizar resultado de uma aposta deste dia")
    print("  4. Voltar")

    acao = input("  Escolha: ").strip()

    if acao == "1":
        return

    if acao == "2":
        if len(ja_registradas) > 0:
            print(f"\n  ⚠️  Já existem {len(ja_registradas)} apostas registradas para {data_str}.")
            confirmar = input("  Continuar mesmo assim (registra apenas as que faltam)? (s/n): ").strip().lower()
            if confirmar != "s":
                return

        registradas = registrar_apostas_dia_anterior(path, data_str)

        if registradas > 0:
            print(f"\n  ✅ {registradas} apostas registradas!")
            atualizar = input("  Deseja atualizar os resultados agora? (s/n): ").strip().lower()
            if atualizar == "s":
                _menu_atualizar_pendentes_do_dia(data_str, data_registro)
        return

    if acao == "3":
        _menu_atualizar_pendentes_do_dia(data_str, data_registro)
        return

    if acao == "4":
        return

    print("  ⚠️  Opção inválida.")


def _menu_atualizar_pendentes_do_dia(data_str, data_registro):
    """Submenu para atualizar resultados pendentes de um dia específico."""
    historico = carregar_historico()
    pendentes = historico[
        (historico["resultado"] == "pendente") &
        (historico["data"].astype(str).eq(data_registro))
    ]

    if len(pendentes) == 0:
        print(f"\n  ⚠️  Nenhuma aposta pendente para {data_str}.")
        todas = historico[historico["data"].astype(str).eq(data_registro)]
        if len(todas) > 0:
            print(f"\n  📊 Apostas de {data_str}:")
            for idx, row in todas.iterrows():
                print(
                    f"     [{idx}] {row['jogo']} | {row['mercado']} | "
                    f"R$ {float(row['valor_apostado']):.2f} | {row['resultado']}"
                )
        return

    print(f"\n  ⏳ Apostas pendentes de {data_str}:")
    for idx, row in pendentes.iterrows():
        pesquisa = " | Ver resultado disponível" if url_pesquisa_resultado(row) else ""
        print(
            f"     [{idx}] {row['jogo']} | {row['mercado']} | "
            f"R$ {float(row['valor_apostado']):.2f}{pesquisa}"
        )

    print("\n  Como deseja atualizar?")
    print("  1. Um a um manualmente")
    print("  2. Ver resultado")
    print("  3. Voltar")

    modo = input("  Escolha: ").strip()

    if modo == "1":
        try:
            idx = int(input("\n  Índice da aposta: "))
            resultado = input("  Resultado (g=ganhou / p=perdeu): ").strip().lower()
            atualizar_resultado(idx, ganhou=(resultado == "g"))
        except ValueError:
            print("  ❌ Entrada inválida.")

    elif modo == "2":
        try:
            idx = int(input("\n  Índice da aposta: "))
        except ValueError:
            print("  ❌ Entrada inválida.")
            return
        if idx not in pendentes.index:
            print("  ⚠️  Índice não encontrado entre as apostas pendentes.")
            return
        abrir_pesquisa_resultado(pendentes.loc[idx])

    elif modo == "3":
        return


# ==============================
# UTILITÁRIO
# ==============================
def limpar_historico_corrompido():
    """Executa uma vez para corrigir histórico com headers duplicados."""
    df = carregar_historico()
    salvar_historico(df)
    print(f"✅ Histórico limpo: {len(df)} apostas mantidas.")

def reconstruir_banca():
    estado = carregar_estado()
    if estado is None:
        # --rebuild-bank pode ser chamado antes de qualquer inicialização manual
        # (ex: fluxo automático de backfill). Em vez de quebrar, inicializa com
        # o mesmo valor padrão já usado em 06_importar_backfill_banca.py (R$ 250),
        # assim o rebuild funciona de ponta a ponta sem passo manual extra.
        print("  ⚠️  Banca ainda não inicializada — inicializando automaticamente com R$ 250,00.")
        estado = inicializar_banca(250.00)
    historico = carregar_historico()

    banca_inicial = float(estado.get("banca_inicial", 100.0))
    banca = banca_inicial
    pico = banca_inicial
    ganhos = 0
    perdas = 0
    lucro_total = 0.0
    total_apostas = 0

    for idx, row in historico.iterrows():
        resultado = str(row.get("resultado", "")).strip().lower()

        if resultado == "pendente":
            historico.at[idx, "banca_apos"] = banca
            continue

        valor = float(row.get("valor_apostado", 0))
        odd = float(row.get("odd", 0))

        if resultado == "ganhou":
            lucro = round(valor * (odd - 1), 2)
            ganhos += 1
        elif resultado == "perdeu":
            lucro = round(-valor, 2)
            perdas += 1
        else:
            continue

        banca = round(banca + lucro, 2)
        pico = max(pico, banca)
        lucro_total = round(lucro_total + lucro, 2)
        total_apostas += 1

        historico.at[idx, "lucro"] = lucro
        historico.at[idx, "banca_apos"] = banca

    finalizadas = historico[historico["resultado"].astype(str).isin(["ganhou", "perdeu"])]
    total_stake = pd.to_numeric(finalizadas["valor_apostado"], errors="coerce").fillna(0).sum()
    roi_total = round((lucro_total / total_stake) * 100, 2) if total_stake > 0 else 0.0

    estado["banca_atual"] = banca
    estado["banca_pico"] = pico
    estado["total_apostas"] = total_apostas
    estado["total_ganhos"] = ganhos
    estado["total_perdas"] = perdas
    estado["lucro_total"] = lucro_total
    estado["roi_total"] = roi_total
    estado["ultima_atualizacao"] = str(date.today())

    salvar_historico(historico)
    salvar_estado(estado)

    print("✅ Banca reconstruída com sucesso.")
    print(f"Banca atual: R$ {banca:.2f}")
    print(f"Lucro total: R$ {lucro_total:+.2f}")
    print(f"ROI total: {roi_total:+.2f}%")

# ==============================
# MENU INTERATIVO
# ==============================
def menu():
    estado = carregar_estado()

    if estado is None:
        print("=" * 55)
        print("🏦 INICIALIZAR BANCA")
        print("=" * 55)
        banca_inicial = float(input("  💵 Qual o valor inicial da sua banca? R$ "))
        estado = inicializar_banca(banca_inicial)
        print(f"  ✅ Banca inicializada com R$ {banca_inicial:.2f}")

    while True:
        estado = carregar_estado()

        print("\n" + "=" * 55)
        print(f"💰 GESTÃO DE BANCA | Banca: R$ {estado['banca_atual']:.2f}")
        print("=" * 55)
        print("  1. Ver previsões do dia / preencher automaticamente")
        print("  2. Registrar aposta manualmente")
        print("  3. Atualizar resultado de aposta")
        print("  4. 📂 Ver previsões anteriores")
        print("  5. Ver painel completo")
        print("  6. Sair")
        print("=" * 55)

        opcao = input("  Escolha: ").strip()

        if opcao == "1":
            menu_apostas_do_dia(estado)

        elif opcao == "2":
            print("\n  📝 Registrar aposta manual:")
            jogo = input("  Jogo (ex: Flamengo vs Palmeiras): ").strip()
            liga = input("  Liga: ").strip()
            mercado = input("  Mercado (ex: G_H_HT): ").strip()
            event = input("  Evento (ex: Gols_H_HT): ").strip()
            prob = float(input("  Probabilidade do modelo (0-1, ex: 0.72): ") or 0)
            confianca = float(input("  Confiança do modelo (0-1, ex: 0.75): ") or prob)
            odd = float(input(f"  Odd da casa (enter para {ODD_PADRAO}): ") or ODD_PADRAO)
            roi_bt = float(input("  ROI do backtest (ex: 0.035): ") or 0)

            valor, kelly_pct = calcular_aposta(estado["banca_atual"], confianca, odd)

            if valor <= 0:
                print("\n  ⚠️  Essa aposta ficou sem valor esperado positivo pelo Kelly.")
                continue

            print(f"\n  💰 Kelly recomenda: R$ {valor:.2f} ({kelly_pct:.2f}% da banca)")
            confirmar = input("  Confirmar aposta? (s/n): ").strip().lower()

            if confirmar == "s":
                registrar_aposta(jogo, liga, mercado, event, prob, confianca, odd, valor, kelly_pct, roi_bt)

        elif opcao == "3":
            historico = carregar_historico()
            pendentes = historico[historico["resultado"] == "pendente"]

            if len(pendentes) == 0:
                print("  ⚠️  Nenhuma aposta pendente.")
                continue

            print("\n  ⏳ Apostas pendentes:")
            for idx, row in pendentes.iterrows():
                pesquisa = " | Ver resultado disponível" if url_pesquisa_resultado(row) else ""
                print(
                    f"     [{idx}] {row['jogo']} | {row['mercado']} | "
                    f"R$ {float(row['valor_apostado']):.2f} | {row['data']}{pesquisa}"
                )

            print("\n  Ação:")
            print("  1. Atualizar resultado")
            print("  2. Ver resultado")
            print("  3. Voltar")
            acao = input("  Escolha: ").strip()

            if acao == "1":
                idx = int(input("\n  Índice da aposta: "))
                resultado = input("  Resultado (g=ganhou / p=perdeu): ").strip().lower()
                atualizar_resultado(idx, ganhou=(resultado == "g"))
            elif acao == "2":
                idx = int(input("\n  Índice da aposta: "))
                if idx not in pendentes.index:
                    print("  ⚠️  Índice não encontrado entre as apostas pendentes.")
                    continue
                abrir_pesquisa_resultado(pendentes.loc[idx])
            elif acao == "3":
                continue
            else:
                print("  ⚠️  Opção inválida.")

        elif opcao == "4":
            menu_apostas_anteriores(estado)

        elif opcao == "5":
            mostrar_dashboard()

        elif opcao == "6":
            print("\n  👋 Até logo!")
            break

        else:
            print("  ⚠️  Opção inválida.")



# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-fill", action="store_true")
    parser.add_argument("--auto-fill-previous", action="store_true")
    parser.add_argument("--rebuild-bank", action="store_true")
    args = parser.parse_args()

    if args.auto_fill:
        estado = carregar_estado()
        if estado is None:
            print("❌ Banca ainda não inicializada.")
            sys.exit(1)

        previsoes_path = caminho_previsoes_do_dia()

        if not os.path.exists(previsoes_path):
            print(f"❌ Previsões não encontradas: {previsoes_path}")
            sys.exit(1)

        registrar_apostas_automaticas(previsoes_path, selecao="todas")
        sys.exit(0)

    if args.auto_fill_previous:
        estado = carregar_estado()
        if estado is None:
            print("❌ Banca ainda não inicializada.")
            sys.exit(1)

        arquivos = listar_previsoes_anteriores()

        total = 0
        print("\n🤖 Registrando previsões anteriores automaticamente...\n")

        for data_str, path in sorted(arquivos):
            registradas = registrar_apostas_dia_anterior(path, data_str)
            total += registradas

        print(f"\n✅ Total registrado automaticamente: {total}")
        sys.exit(0)

    if args.rebuild_bank:
        reconstruir_banca()
        sys.exit(0)

    menu()

