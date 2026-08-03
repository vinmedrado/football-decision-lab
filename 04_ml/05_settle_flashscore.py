#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completa apostas pendentes usando os IDs locais do Flashscore.

O modo padrão é somente prévia. Use --apply para gravar resultados validados.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for import_path in (PROJECT_ROOT, BASE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.flashscore_settlement import (  # noqa: E402
    FlashscoreClient,
    find_local_match_id,
    validate_match_teams,
)
from utils.settlement_utils import resolver_mercado  # noqa: E402


HISTORY_PATH = BASE_DIR / "banca" / "historico_apostas.csv"
RAW_DAILY_DIR = PROJECT_ROOT / "data" / "raw" / "futpython" / "daily"
CACHE_DIR = BASE_DIR / "banca" / "flashscore_cache"
DEFAULT_REPORT = BASE_DIR / "banca" / "flashscore_settlement_preview.csv"

PROVENANCE_COLUMNS = {
    "settlement_source": "",
    "settlement_match_id": "",
    "settlement_url": "",
    "data_realizacao": "",
    "placar_ft": "",
    "placar_ht": "",
    "settled_at": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consulta páginas públicas do Flashscore para completar apostas pendentes. "
            "Sem --apply, apenas gera uma prévia."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Grava apenas resultados finalizados e validados.")
    parser.add_argument("--date", help="Limita às previsões da data original YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, help="Máximo de pendências a consultar.")
    parser.add_argument("--delay", type=float, default=1.25, help="Intervalo entre consultas reais, em segundos.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout de cada consulta, em segundos.")
    parser.add_argument("--max-retries", type=int, default=2, help="Número de novas tentativas por página.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignora o cache local e consulta novamente o Flashscore.",
    )
    parser.add_argument(
        "--skip-rebuild-bank",
        action="store_true",
        help="Não reconstrói banca_apos e o estado da banca depois de --apply.",
    )
    parser.add_argument("--history", type=Path, default=HISTORY_PATH, help=argparse.SUPPRESS)
    parser.add_argument(
        "--history-indices",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--raw-daily-dir", type=Path, default=RAW_DAILY_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Caminho do relatório CSV.")
    return parser.parse_args()


def _text(row: pd.Series, name: str) -> str:
    value = row.get(name, "")
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split_game(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    if " x " not in text:
        return "", ""
    home, away = text.split(" x ", 1)
    return home.strip(), away.strip()


def _actual_date(played_at: str) -> str:
    value = str(played_at or "").strip()
    for pattern in ("%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return ""


def _report_base(index: Any, row: pd.Series) -> dict[str, Any]:
    return {
        "indice_historico": index,
        "data_original": _text(row, "data"),
        "liga": _text(row, "liga"),
        "home_esperado": _text(row, "home"),
        "away_esperado": _text(row, "away"),
        "mercado": _text(row, "mercado"),
        "odd": _text(row, "odd"),
        "valor_apostado": _text(row, "valor_apostado"),
        "flashscore_id": "",
        "flashscore_url": "",
        "home_flashscore": "",
        "away_flashscore": "",
        "confianca_times": "",
        "status_flashscore": "",
        "data_realizacao": "",
        "placar_ft": "",
        "placar_ht": "",
        "acao": "",
        "resultado_proposto": "",
        "lucro_proposto": "",
        "fonte_id_local": "",
        "cache": "",
        "erro": "",
    }


def _ensure_provenance_columns(history: pd.DataFrame) -> None:
    for column, default in PROVENANCE_COLUMNS.items():
        if column not in history.columns:
            history[column] = default


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_history(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.backup_flashscore_{timestamp}{path.suffix}")
    backup.write_bytes(path.read_bytes())
    return backup


def main() -> int:
    args = parse_args()
    if not args.history.exists():
        print(f"ERRO: histórico não encontrado: {args.history}")
        return 1
    if not args.raw_daily_dir.exists():
        print(f"ERRO: pasta dos jogos diários não encontrada: {args.raw_daily_dir}")
        return 1
    if args.limit is not None and args.limit <= 0:
        print("ERRO: --limit deve ser maior que zero.")
        return 1

    history = pd.read_csv(args.history, low_memory=False)
    if "resultado" not in history.columns:
        print("ERRO: histórico sem a coluna resultado.")
        return 1

    pending_mask = history["resultado"].fillna("").astype(str).str.strip().str.lower().eq("pendente")
    pending = history.loc[pending_mask].copy()
    if args.history_indices:
        try:
            selected_indices = {int(value.strip()) for value in str(args.history_indices).split(",") if value.strip()}
        except ValueError:
            print("ERRO: --history-indices deve conter índices inteiros separados por vírgula.")
            return 1
        pending = pending[pending.index.isin(selected_indices)]
    if args.date:
        pending = pending[pending["data"].fillna("").astype(str).str.strip().eq(str(args.date))]
    if args.limit is not None:
        pending = pending.head(args.limit)

    if pending.empty:
        print("Nenhuma pendência encontrada para o recorte solicitado.")
        return 0

    client = FlashscoreClient(
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        max_retries=args.max_retries,
        min_request_interval=args.delay,
    )
    report_rows: list[dict[str, Any]] = []
    proposed_updates: dict[Any, dict[str, Any]] = {}
    network_requests = 0

    mode = "APLICAÇÃO" if args.apply else "PRÉVIA SEGURA"
    print("=" * 68)
    print(f"SETTLEMENT FLASHSCORE — {mode}")
    print(f"Pendências selecionadas: {len(pending)}")
    print("=" * 68)

    for index, row in pending.iterrows():
        report = _report_base(index, row)
        home = _text(row, "home")
        away = _text(row, "away")
        if not home or not away:
            home, away = _split_game(_text(row, "jogo"))
            report["home_esperado"] = home
            report["away_esperado"] = away
        if not home or not away:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "TIMES_AUSENTES_NO_HISTORICO"
            report_rows.append(report)
            continue

        local = find_local_match_id(
            args.raw_daily_dir,
            _text(row, "data"),
            home,
            away,
        )
        if local is None:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "ID_FLASHSCORE_NAO_ENCONTRADO_NO_CSV_DIARIO"
            report_rows.append(report)
            continue

        report["flashscore_id"] = local.match_id
        report["fonte_id_local"] = local.source_file
        try:
            match, from_cache = client.get_match(local.match_id, refresh=args.refresh_cache)
            if not from_cache:
                network_requests += 1
        except RuntimeError as exc:
            report["acao"] = "TENTAR_NOVAMENTE"
            report["erro"] = str(exc)
            report_rows.append(report)
            continue

        report["cache"] = "SIM" if from_cache else "NAO"
        report["flashscore_url"] = match.source_url
        report["home_flashscore"] = match.home
        report["away_flashscore"] = match.away
        report["status_flashscore"] = match.status
        report["data_realizacao"] = _actual_date(match.played_at)
        if None not in (match.ft_home, match.ft_away):
            report["placar_ft"] = f"{match.ft_home}-{match.ft_away}"
        if match.halftime_score is not None:
            report["placar_ht"] = f"{match.halftime_score[0]}-{match.halftime_score[1]}"

        teams_ok, team_confidence = validate_match_teams(home, away, match)
        report["confianca_times"] = team_confidence
        if not teams_ok:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "TIMES_DIVERGENTES"
            report_rows.append(report)
            continue

        if not match.is_finished:
            report["acao"] = "MANTER_PENDENTE"
            report["erro"] = f"STATUS_NAO_FINAL: {match.status}"
            report_rows.append(report)
            continue

        regulation = match.regulation_score
        if regulation is None:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "PLACAR_REGULAMENTAR_INCOMPLETO"
            report_rows.append(report)
            continue
        market = _text(row, "mercado")
        if "_HT" in market.upper() and match.halftime_score is None:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "PLACAR_HT_AUSENTE"
            report_rows.append(report)
            continue
        halftime = match.halftime_score or (0, 0)
        won = resolver_mercado(
            market,
            regulation[0],
            regulation[1],
            halftime[0],
            halftime[1],
        )
        if won is None:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "MERCADO_NAO_SUPORTADO_OU_PLACAR_HT_AUSENTE"
            report_rows.append(report)
            continue

        try:
            stake = float(row.get("valor_apostado", 0) or 0)
            odd = float(row.get("odd", 0) or 0)
        except (TypeError, ValueError):
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "ODD_OU_STAKE_INVALIDA"
            report_rows.append(report)
            continue
        if stake <= 0 or odd <= 1:
            report["acao"] = "REVISAO_MANUAL"
            report["erro"] = "ODD_OU_STAKE_FORA_DO_INTERVALO"
            report_rows.append(report)
            continue

        result = "ganhou" if won else "perdeu"
        profit = round(stake * (odd - 1), 2) if won else round(-stake, 2)
        report["acao"] = "APLICAR" if args.apply else "PRONTO_PARA_APLICAR"
        report["resultado_proposto"] = result
        report["lucro_proposto"] = profit
        report["placar_ft"] = f"{regulation[0]}-{regulation[1]}"
        proposed_updates[index] = {
            "resultado": result,
            "lucro": profit,
            "settlement_source": "FLASHSCORE_MOBILE",
            "settlement_match_id": match.match_id,
            "settlement_url": match.source_url,
            "data_realizacao": report["data_realizacao"],
            "placar_ft": report["placar_ft"],
            "placar_ht": report["placar_ht"],
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }
        report_rows.append(report)

    report_frame = pd.DataFrame(report_rows)
    _atomic_write_csv(report_frame, args.report)

    backup: Path | None = None
    if args.apply and proposed_updates:
        _ensure_provenance_columns(history)
        backup = _backup_history(args.history)
        for index, updates in proposed_updates.items():
            for column, value in updates.items():
                history.at[index, column] = value
        _atomic_write_csv(history, args.history)

        if not args.skip_rebuild_bank:
            rebuild = subprocess.run(
                [sys.executable, str(BASE_DIR / "04_banca.py"), "--rebuild-bank"],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if rebuild.returncode != 0:
                print("AVISO: resultados foram gravados, mas a reconstrução da banca falhou.")
                print(f"Backup disponível em: {backup}")
                return rebuild.returncode

    action_counts = report_frame["acao"].value_counts().to_dict() if not report_frame.empty else {}
    print("")
    print("=" * 68)
    print("RESUMO")
    print(f"Consultas de rede realizadas : {network_requests}")
    print(f"Resultados validados         : {len(proposed_updates)}")
    for action, count in sorted(action_counts.items()):
        print(f"{action:28}: {count}")
    print(f"Relatório                    : {args.report}")
    if args.apply:
        print(f"Histórico atualizado         : {args.history}")
        if backup is not None:
            print(f"Backup                       : {backup}")
    elif proposed_updates:
        print("")
        print("Prévia concluída. Para gravar exatamente os casos validados, execute novamente com --apply.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
