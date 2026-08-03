#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / '04_ml'
if str(ML_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(ML_DIR.resolve()))

from utils.prediction_paths import (  # noqa: E402
    HISTORICAL_PREDICTIONS_DIR,
    LEGACY_PREDICTIONS_DIR,
    PREDICTIONS_DIR,
    historical_prediction_files,
    normal_prediction_files,
)

HISTORICO = ML_DIR / 'banca' / 'historico_apostas.csv'
BASE_OFICIAL = PROJECT_ROOT / 'data' / 'base_oficial.csv'
PERFIL = ML_DIR / 'reports' / 'perfil_operacional_mercados.json'
DIAGNOSTICO = ML_DIR / 'banca' / 'diagnostico_importacao_backfill.csv'

BANCA_INICIAL = 250.00
RISCO_MAX_POR_APOSTA = 0.02
STAKE_FIXA = 5.00
MAX_APOSTAS_DIA = 20
MAX_APOSTAS_POR_LIGA = 5
MAX_APOSTAS_POR_MERCADO = 20
EXPOSICAO_MAX_DIA_PCT = 0.20

# Em backfill histórico, diferenças de padronização entre previsão e base oficial
# não devem apagar todas as entradas. O match fica como diagnóstico por padrão.
EXIGIR_MATCH_BASE_OFICIAL = False

HIST_COLUMNS = [
    'data', 'liga','Round', 'jogo', 'home', 'away', 'mercado', 'event',
    'prob_modelo', 'confianca', 'odd', 'valor_apostado', 'kelly_pct',
    'roi_bt', 'resultado', 'lucro', 'banca_apos', 'base_match', 'origem',
]


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm(x: Any) -> str:
    return str(x).strip().lower()


def bool_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'sim', 's', 'yes', 'y'])


def carregar_perfil() -> dict[str, dict]:
    if not PERFIL.exists():
        print(f'[AVISO] Perfil não encontrado: {PERFIL}')
        return {}
    try:
        data = json.loads(PERFIL.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'[AVISO] Falha ao ler perfil: {exc}')
        return {}
    return {
        str(m).upper(): cfg
        for m, cfg in data.items()
        if isinstance(cfg, dict) and cfg.get('ativo') is True
    }


def carregar_jogos_validos(datas_necessarias: set[str] | None = None) -> tuple[set[str], bool]:
    if not BASE_OFICIAL.exists():
        print(f'[AVISO] Base oficial não encontrada: {BASE_OFICIAL}')
        return set(), False
    try:
        cabecalho = pd.read_csv(BASE_OFICIAL, nrows=0)
    except pd.errors.EmptyDataError:
        print(f'[AVISO] Base oficial vazia: {BASE_OFICIAL}')
        return set(), False
    except Exception as exc:
        print(f'[AVISO] Falha ao ler base oficial: {exc}')
        return set(), False

    if 'Date' not in cabecalho.columns:
        return set(), False

    home_col = 'Home_std' if 'Home_std' in cabecalho.columns else 'Home'
    away_col = 'Away_std' if 'Away_std' in cabecalho.columns else 'Away'
    liga_col = 'League_std' if 'League_std' in cabecalho.columns else ('League' if 'League' in cabecalho.columns else None)

    if home_col not in cabecalho.columns or away_col not in cabecalho.columns:
        print('[AVISO] Base oficial sem colunas Home/Away.')
        return set(), False

    usecols = ['Date', home_col, away_col] + ([liga_col] if liga_col else [])
    jogos: set[str] = set()
    for base in pd.read_csv(BASE_OFICIAL, usecols=usecols, chunksize=200_000, low_memory=False):
        base['Date_key'] = base['Date'].fillna('').astype(str).str.strip()
        if datas_necessarias:
            base = base[base['Date_key'].isin(datas_necessarias)].copy()
        if base.empty:
            continue
        base['Home_key'] = base[home_col].astype(str).map(norm)
        base['Away_key'] = base[away_col].astype(str).map(norm)
        if liga_col:
            base['Liga_key'] = base[liga_col].astype(str).str.strip().str.upper()
            jogos.update(base['Date_key'] + '|' + base['Liga_key'] + '|' + base['Home_key'] + '|' + base['Away_key'])
        else:
            jogos.update(base['Date_key'] + '|' + base['Home_key'] + '|' + base['Away_key'])
    return jogos, bool(liga_col)


def salvar_historico(hist: pd.DataFrame) -> None:
    HISTORICO.parent.mkdir(parents=True, exist_ok=True)
    if hist.empty:
        hist = pd.DataFrame(columns=HIST_COLUMNS)
    else:
        for col in HIST_COLUMNS:
            if col not in hist.columns:
                hist[col] = ''
        hist = hist[HIST_COLUMNS]
    temp_path = HISTORICO.with_name(f'.{HISTORICO.name}.{os.getpid()}.tmp')
    try:
        hist.to_csv(temp_path, index=False, encoding='utf-8-sig')
        os.replace(temp_path, HISTORICO)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def salvar_diagnostico(registros: list[dict[str, Any]]) -> None:
    DIAGNOSTICO.parent.mkdir(parents=True, exist_ok=True)
    temp_path = DIAGNOSTICO.with_name(f'.{DIAGNOSTICO.name}.{os.getpid()}.tmp')
    try:
        pd.DataFrame(registros).to_csv(temp_path, index=False, encoding='utf-8-sig')
        os.replace(temp_path, DIAGNOSTICO)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def diag(registros: list[dict[str, Any]], arquivo: str, etapa: str, linhas: int) -> None:
    registros.append({'arquivo': arquivo, 'etapa': etapa, 'linhas': int(linhas)})



def localizar_arquivos_previsao(modo: str = 'auto') -> tuple[list[Path], str]:
    """Localiza previsões sem misturar modo normal e histórico.

    Prioridades:
    - modo=historicas: usa apenas 04_ml/previsoes_historicas
    - modo=normais: usa 04_ml/previsoes e, como fallback, arquivos na raiz 04_ml
    - modo=auto: escolhe o conjunto cuja modificação mais recente é a mais nova
    """
    historicas = historical_prediction_files('previsoes_2026-*.csv')
    normais_completas = normal_prediction_files('previsoes_2026-*.csv', include_legacy=True)

    # Compatibilidade com versões antigas que salvavam direto em 04_ml/.

    if modo == 'historicas':
        return historicas, 'históricas'

    if modo == 'normais':
        return normais_completas, 'normais'

    if historicas and normais_completas:
        ultima_historica = max(p.stat().st_mtime for p in historicas)
        ultima_normal = max(p.stat().st_mtime for p in normais_completas)
        if ultima_historica >= ultima_normal:
            return historicas, 'históricas (auto)'
        return normais_completas, 'normais (auto)'

    if historicas:
        return historicas, 'históricas (auto)'

    if normais_completas:
        return normais_completas, 'normais (auto)'

    return [], 'nenhuma'


def filtrar_arquivos_por_data(arquivos: list[Path], data: str | None) -> list[Path]:
    if not data:
        return arquivos
    nome_esperado = f'previsoes_{data}.csv'
    return [arq for arq in arquivos if arq.name == nome_esperado]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Importa previsões normais ou históricas para a banca simulada.'
    )
    parser.add_argument(
        '--source',
        choices=['auto', 'historicas', 'normais'],
        default='auto',
        help='Origem das previsões. Padrão: auto.',
    )
    parser.add_argument(
        '--date',
        help='Importa apenas o arquivo previsoes_YYYY-MM-DD.csv informado, preservando outras datas do historico.',
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    mercados_ativos = carregar_perfil()

    arquivos, origem = localizar_arquivos_previsao(args.source)
    arquivos = filtrar_arquivos_por_data(arquivos, args.date)
    linhas: list[dict[str, Any]] = []
    diagnosticos: list[dict[str, Any]] = []
    sem_match_base = 0
    importadas_sem_match = 0

    if not arquivos:
        print('[AVISO] Nenhum arquivo previsoes_2026-*.csv encontrado.')
        print(f'[AVISO] Pastas verificadas: {PREDICTIONS_DIR}, {HISTORICAL_PREDICTIONS_DIR} e {LEGACY_PREDICTIONS_DIR}')
        print('[AVISO] Histórico existente foi preservado; nenhuma escrita foi realizada.')
        return 1

    datas_necessarias = {str(args.date)} if args.date else {
        match.group(1)
        for arq in arquivos
        if (match := re.search(r'(\d{4}-\d{2}-\d{2})', arq.name))
    }
    jogos_validos, base_tem_liga = carregar_jogos_validos(datas_necessarias)

    print('=' * 60)
    print('ARQUIVOS DE PREVISÃO LOCALIZADOS')
    print('=' * 60)
    print(f'Origem escolhida        : {origem}')
    print(f'Total de arquivos       : {len(arquivos)}')
    print(f'Primeiro arquivo        : {arquivos[0]}')
    print(f'Último arquivo          : {arquivos[-1]}')
    print('=' * 60)

    erros_leitura = 0
    for arq in arquivos:
        try:
            df = pd.read_csv(arq, low_memory=False)
        except pd.errors.EmptyDataError:
            diag(diagnosticos, arq.name, 'arquivo_vazio', 0)
            erros_leitura += 1
            continue
        except Exception as exc:
            print(f'[AVISO] Falha ao ler {arq.name}: {exc}')
            diag(diagnosticos, arq.name, 'erro_leitura', 0)
            erros_leitura += 1
            continue

        diag(diagnosticos, arq.name, '01_carregado', len(df))
        if df.empty or 'mercado' not in df.columns:
            continue

        df['mercado_key'] = df['mercado'].astype(str).str.upper().str.strip()

        modo_auditoria = (
            "modo_auditoria" in df.columns
            and bool_mask(df["modo_auditoria"]).any()
        )

        if modo_auditoria:
            diag(diagnosticos, arq.name, "02_modo_auditoria", len(df))
        else:
            if "entrada_simulada" in df.columns:
                df = df[bool_mask(df["entrada_simulada"])]
                diag(diagnosticos, arq.name, "02_entrada_simulada", len(df))
            else:
                diag(diagnosticos, arq.name, "02_sem_entrada_simulada_nao_bloqueado", len(df))

            if df.empty:
                continue

            if "apostar" in df.columns:
                df = df[bool_mask(df["apostar"])]
                diag(diagnosticos, arq.name, "03_apostar", len(df))
            elif "recomendacao_operacional" in df.columns:
                df = df[bool_mask(df["recomendacao_operacional"])]
                diag(diagnosticos, arq.name, "03_recomendacao_operacional", len(df))
            elif "operacao_real" in df.columns:
                df = df[bool_mask(df["operacao_real"])]
                diag(diagnosticos, arq.name, "03_operacao_real", len(df))
            else:
                diag(diagnosticos, arq.name, "03_sem_coluna_sinal", 0)
                continue

            if df.empty:
                continue

        if mercados_ativos:
            df = df[df['mercado_key'].isin(mercados_ativos.keys())]
            diag(diagnosticos, arq.name, '04_mercado_ativo', len(df))
        else:
            diag(diagnosticos, arq.name, '04_sem_perfil_nao_bloqueado', len(df))

        if df.empty:
            continue

        if 'odd_ok' in df.columns:
            df = df[bool_mask(df['odd_ok'])]
            diag(diagnosticos, arq.name, '05_odd_ok', len(df))
        else:
            diag(diagnosticos, arq.name, '05_sem_odd_ok_nao_bloqueado', len(df))

        if df.empty or 'odd' not in df.columns:
            continue

        df['odd_num'] = pd.to_numeric(df['odd'], errors='coerce')
        df = df[df['odd_num'].notna()]
        diag(diagnosticos, arq.name, '06_odd_numerica', len(df))
        if df.empty:
            continue

        if mercados_ativos:
            df['odd_min_mercado'] = df['mercado_key'].map(lambda m: mercados_ativos[m].get('odd_min', 1.20))
            df['odd_max_mercado'] = df['mercado_key'].map(lambda m: mercados_ativos[m].get('odd_max', 3.50))
        else:
            df['odd_min_mercado'] = 1.20
            df['odd_max_mercado'] = 1000.0

        df = df[(df['odd_num'] >= df['odd_min_mercado']) & (df['odd_num'] <= df['odd_max_mercado'])]
        diag(diagnosticos, arq.name, '07_faixa_odd', len(df))
        if df.empty:
            continue

        if 'ev' in df.columns:
            df['ev_sort'] = pd.to_numeric(df['ev'], errors='coerce').fillna(0)
        elif 'valor_esperado' in df.columns:
            df['ev_sort'] = pd.to_numeric(df['valor_esperado'], errors='coerce').fillna(0)
        else:
            df['ev_sort'] = 0.0

        if 'data' not in df.columns:
            continue
        if 'liga' not in df.columns:
            df['liga'] = ''
        for col in ['home', 'away']:
            if col not in df.columns:
                df[col] = ''

        limite_dia = BANCA_INICIAL * EXPOSICAO_MAX_DIA_PCT
        max_apostas_dia = min(MAX_APOSTAS_DIA, int(limite_dia // STAKE_FIXA))

        df = df.sort_values('ev_sort', ascending=False)
        if max_apostas_dia > 0:
            df = df.groupby('data', group_keys=False).head(max_apostas_dia)
        diag(diagnosticos, arq.name, '08_limite_dia', len(df))

        df = df.sort_values('ev_sort', ascending=False).groupby(['data', 'liga'], group_keys=False).head(MAX_APOSTAS_POR_LIGA)
        diag(diagnosticos, arq.name, '09_limite_liga', len(df))

        df = df.sort_values('ev_sort', ascending=False).groupby(['data', 'mercado_key'], group_keys=False).head(MAX_APOSTAS_POR_MERCADO)
        diag(diagnosticos, arq.name, '10_limite_mercado', len(df))

        for _, row in df.iterrows():
            data = str(row.get('data', '')).strip()
            liga = str(row.get('liga', '')).strip().upper()
            home = str(row.get('home', '')).strip()
            away = str(row.get('away', '')).strip()

            key = (
                f'{data}|{liga}|{norm(home)}|{norm(away)}'
                if base_tem_liga
                else f'{data}|{norm(home)}|{norm(away)}'
            )
            base_match = bool(jogos_validos and key in jogos_validos)

            if not base_match:
                sem_match_base += 1
                if EXIGIR_MATCH_BASE_OFICIAL:
                    continue
                importadas_sem_match += 1

            linhas.append({
                'data': data,
                'liga': liga,
                "Round": row.get("Round", ""),
                'jogo': f'{home} x {away}',
                'home': home,
                'away': away,
                'mercado': row.get('mercado', ''),
                'event': row.get('event', row.get('mercado', '')),
                'prob_modelo': to_float(row.get('prob_modelo', row.get('prob', 0))),
                'confianca': to_float(row.get('prob_evento', row.get('confianca', 0))),
                'odd': to_float(row.get('odd_num', row.get('odd', 1.5)), 1.5),
                'valor_apostado': STAKE_FIXA,
                'kelly_pct': RISCO_MAX_POR_APOSTA * 100,
                'roi_bt': to_float(row.get('roi_bt', 0)),
                'resultado': 'pendente',
                'lucro': 0.0,
                'banca_apos': BANCA_INICIAL,
                'base_match': base_match,
                'origem': 'backfill_simulado',
            })

    hist = pd.DataFrame(linhas)
    if not hist.empty:
        hist = hist.drop_duplicates(subset=['data', 'jogo', 'mercado'], keep='first')
    importadas_nesta_execucao = len(hist)

    if erros_leitura:
        salvar_diagnostico(diagnosticos)
        print(f'[ERRO] {erros_leitura} arquivo(s) de previsão não puderam ser lidos.')
        print('[ERRO] Histórico preservado para evitar substituição parcial.')
        return 1

    if hist.empty:
        salvar_diagnostico(diagnosticos)
        print('[AVISO] Nenhuma entrada elegível foi encontrada.')
        print('[AVISO] Histórico existente foi preservado; nenhuma escrita foi realizada.')
        return 0

    if args.date and HISTORICO.exists():
        try:
            hist_atual = pd.read_csv(HISTORICO, low_memory=False)
        except pd.errors.EmptyDataError:
            hist_atual = pd.DataFrame(columns=HIST_COLUMNS)
        if not hist_atual.empty and 'data' in hist_atual.columns:
            hist_atual = hist_atual[hist_atual['data'].astype(str).str.strip() != str(args.date)]
            hist = pd.concat([hist_atual, hist], ignore_index=True)
            hist = hist.drop_duplicates(subset=['data', 'jogo', 'mercado'], keep='last')

    salvar_historico(hist)
    salvar_diagnostico(diagnosticos)

    print('=' * 60)
    print('IMPORTAÇÃO DO BACKFILL FINALIZADA')
    print('=' * 60)
    print(f'Banca inicial           : R$ {BANCA_INICIAL:.2f}')
    print(f'Apostas nesta execução  : {importadas_nesta_execucao}')
    print(f'Linhas no histórico     : {len(hist)}')
    print(f'Sem match base oficial  : {sem_match_base}')
    print(f'Importadas sem match    : {importadas_sem_match}')
    print(f'Exigir match base       : {EXIGIR_MATCH_BASE_OFICIAL}')
    print('Mercados ativos         : ' + (', '.join(sorted(mercados_ativos.keys())) if mercados_ativos else 'perfil ausente/vazio — não bloqueado'))
    print(f'Stake fixa              : R$ {STAKE_FIXA:.2f}')
    print(f'Arquivo salvo           : {HISTORICO}')
    print(f'Origem das previsões    : {origem}')
    print(f'Diagnóstico salvo       : {DIAGNOSTICO}')
    if hist.empty:
        print('-' * 60)
        print('[AVISO] Nenhuma entrada elegível foi importada.')
        print('[AVISO] O CSV foi salvo com cabeçalho para evitar EmptyDataError.')
        print('[AVISO] Consulte o diagnóstico para localizar o filtro responsável.')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
