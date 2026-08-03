#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "04_ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from utils.prediction_paths import PREDICTIONS_DIR, normal_prediction_path  # noqa: E402

DAILY_DIR = ROOT / "data" / "raw" / "futpython" / "daily"
PREDICT_SCRIPT = ML_DIR / "03_predict.py"
REPORT_PATH = ML_DIR / "reports" / "regeneracao_previsoes_2026.csv"
CHECKPOINT_PATH = ML_DIR / "reports" / "progresso_regeneracao_previsoes.json"
STATUS_DASHBOARD_PATH = ML_DIR / "reports" / "status_dashboard.json"

REQUIRED_DAILY_COLUMNS = {"Date", "Home", "Away"}
BOOL_TRUE = {"true", "1", "sim", "s", "yes", "y"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def daterange(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    while current <= last:
        yield current.isoformat()
        current += timedelta(days=1)


def dates_list(start: str, end: str) -> list[str]:
    return list(daterange(start, end))


def bool_count(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    if df[col].dtype == bool:
        return int(df[col].sum())
    return int(df[col].astype(str).str.strip().str.lower().isin(BOOL_TRUE).sum())


def validate_daily(path: Path) -> tuple[bool, int, str]:
    if not path.exists():
        return False, 0, "daily_ausente"
    if path.stat().st_size <= 0:
        return False, 0, "daily_vazio"
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return False, 0, "daily_vazio"
    except Exception as exc:
        return False, 0, f"erro_daily:{exc}"
    if df.empty:
        return False, 0, "daily_sem_linhas"
    missing = sorted(REQUIRED_DAILY_COLUMNS - set(df.columns))
    if missing:
        return False, int(len(df)), "daily_sem_colunas:" + ",".join(missing)
    valid_games = df[["Home", "Away"]].dropna(how="any")
    if valid_games.empty:
        return False, int(len(df)), "daily_sem_jogos_validos"
    return True, int(len(df)), ""


def prediction_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "previsoes": 0,
        "apostar_true": 0,
        "entrada_simulada_true": 0,
        "operacao_real_true": 0,
        "recomendacao_operacional_true": 0,
    }
    if not path.exists() or path.stat().st_size <= 0:
        return metrics
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return metrics
    metrics["previsoes"] = int(len(df))
    metrics["apostar_true"] = bool_count(df, "apostar")
    metrics["entrada_simulada_true"] = bool_count(df, "entrada_simulada")
    metrics["operacao_real_true"] = bool_count(df, "operacao_real")
    metrics["recomendacao_operacional_true"] = bool_count(df, "recomendacao_operacional")
    return metrics


def validate_prediction_file(path: Path, expected_date: str) -> tuple[bool, dict[str, Any], str]:
    if not path.exists():
        return False, {}, "arquivo_ausente"
    if path.stat().st_size <= 0:
        return False, {}, "arquivo_vazio"
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            header = next(csv.reader(handle), None)
        if not header:
            return False, {}, "sem_cabecalho"
    except Exception as exc:
        return False, {}, f"erro_cabecalho:{exc}"
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return False, {}, "csv_vazio"
    except Exception as exc:
        return False, {}, f"erro_leitura:{exc}"
    if df.empty:
        return False, {}, "sem_linhas"
    if "data" not in df.columns:
        return False, {}, "sem_coluna_data"
    dates = set(df["data"].dropna().astype(str).str.strip().unique())
    if dates != {expected_date}:
        return False, {}, "data_conteudo_diferente:" + ",".join(sorted(dates)[:5])
    return True, prediction_metrics(path), ""


def save_report(rows: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prediction_version(files: list[Path]) -> str:
    import hashlib

    parts = []
    for path in files:
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            continue
    if not parts:
        return "sem-previsoes"
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def write_state(
    *,
    status: str,
    etapa_atual: str,
    start_date: str,
    end_date: str,
    all_dates: list[str],
    completed_dates: list[str],
    error_dates: list[str],
    ignored_dates: list[str],
    valid_files: list[str],
    invalid_files: list[dict[str, str]],
    totals: dict[str, int],
    inicio_execucao: str,
    started_perf: float,
    mensagem: str,
    current_next: str | None = None,
) -> None:
    completed_sorted = sorted(set(completed_dates))
    pending = [data for data in all_dates if data not in completed_sorted and data not in set(ignored_dates)]
    proxima = current_next if current_next is not None else (pending[0] if pending else None)
    ultima = completed_sorted[-1] if completed_sorted else None
    payload = {
        "status": status,
        "etapa_atual": etapa_atual,
        "data_inicio_periodo": start_date,
        "data_fim_periodo": end_date,
        "ultima_data_concluida": ultima,
        "proxima_data": proxima,
        "datas_totais": len(all_dates),
        "datas_concluidas": len(completed_sorted),
        "datas_com_erro": sorted(set(error_dates)),
        "datas_ignoradas": sorted(set(ignored_dates)),
        "arquivos_validos": len(valid_files),
        "arquivos_invalidos": invalid_files,
        "previsoes_geradas": int(totals.get("previsoes", 0)),
        "apostar_true": int(totals.get("apostar_true", 0)),
        "entrada_simulada_true": int(totals.get("entrada_simulada_true", 0)),
        "inicio_execucao": inicio_execucao,
        "ultima_atualizacao": now_iso(),
        "tempo_decorrido_segundos": round(time.perf_counter() - started_perf, 3),
        "mensagem": mensagem,
    }
    atomic_write_json(CHECKPOINT_PATH, payload)

    normal_files = sorted(PREDICTIONS_DIR.glob("previsoes_*.csv")) if PREDICTIONS_DIR.exists() else []
    latest = normal_prediction_path(ultima) if ultima else (normal_files[-1] if normal_files else None)
    latest_stat = latest.stat() if latest and latest.exists() else None
    percent = round((len(completed_sorted) / len(all_dates)) * 100, 2) if all_dates else 0.0
    dashboard = {
        "versao_dados": prediction_version([p for p in normal_files if p.exists() and p.stat().st_size > 0]),
        "ultima_previsao_data": ultima,
        "ultimo_arquivo": str(latest.relative_to(ROOT)) if latest and latest.exists() else None,
        "ultimo_arquivo_tamanho": latest_stat.st_size if latest_stat else 0,
        "ultimo_arquivo_modificado_em": datetime.fromtimestamp(latest_stat.st_mtime).isoformat(timespec="seconds") if latest_stat else None,
        "quantidade_arquivos": len([p for p in normal_files if p.exists() and p.stat().st_size > 0]),
        "pipeline_status": status,
        "etapa_atual": etapa_atual,
        "datas_concluidas": len(completed_sorted),
        "datas_totais": len(all_dates),
        "percentual": percent,
        "ultima_atualizacao": now_iso(),
        "nova_previsao_disponivel": True,
        "proxima_data": proxima,
    }
    atomic_write_json(STATUS_DASHBOARD_PATH, dashboard)


def load_existing_report() -> list[dict[str, Any]]:
    if not REPORT_PATH.exists() or REPORT_PATH.stat().st_size <= 0:
        return []
    try:
        return pd.read_csv(REPORT_PATH, low_memory=False).fillna("").to_dict("records")
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenera previsoes normais de 2026 sem fetch.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-13")
    parser.add_argument("--no-resume", action="store_true", help="Ignora relatorio existente e recomeca.")
    args = parser.parse_args()

    inicio_execucao = now_iso()
    started_perf = time.perf_counter()
    all_dates = dates_list(args.start, args.end)
    rows: list[dict[str, Any]] = [] if args.no_resume else load_existing_report()
    total = 0
    ok = 0
    completed_dates: list[str] = []
    error_dates: list[str] = []
    ignored_dates: list[str] = []
    valid_files: list[str] = []
    invalid_files: list[dict[str, str]] = []
    totals = {"previsoes": 0, "apostar_true": 0, "entrada_simulada_true": 0}
    existing_by_date = {str(row.get("data")): row for row in rows if str(row.get("data", "")).strip()}

    for data in all_dates:
        output_path = normal_prediction_path(data)
        valid, metrics, reason = validate_prediction_file(output_path, data)
        if valid:
            completed_dates.append(data)
            valid_files.append(str(output_path.relative_to(ROOT)))
            totals["previsoes"] += int(metrics.get("previsoes", 0))
            totals["apostar_true"] += int(metrics.get("apostar_true", 0))
            totals["entrada_simulada_true"] += int(metrics.get("entrada_simulada_true", 0))
            if data not in existing_by_date:
                row = {
                    "data": data,
                    "status": "ok",
                    "tempo_seg": 0.0,
                    "jogos": 0,
                    **metrics,
                    "arquivo_daily": str((DAILY_DIR / f"jogos_do_dia_{data}.csv").relative_to(ROOT)),
                    "arquivo_previsoes": str(output_path.relative_to(ROOT)),
                    "erro": "preexistente_validado",
                }
                rows.append(row)
        elif output_path.exists():
            invalid_files.append({"data": data, "arquivo": str(output_path.relative_to(ROOT)), "motivo": reason})

    rows = sorted(rows, key=lambda row: str(row.get("data", "")))
    save_report(rows)
    write_state(
        status="em_execucao",
        etapa_atual="validacao_inicial",
        start_date=args.start,
        end_date=args.end,
        all_dates=all_dates,
        completed_dates=completed_dates,
        error_dates=error_dates,
        ignored_dates=ignored_dates,
        valid_files=valid_files,
        invalid_files=invalid_files,
        totals=totals,
        inicio_execucao=inicio_execucao,
        started_perf=started_perf,
        mensagem="Validação inicial concluída; retomando somente pendências.",
    )

    for data in all_dates:
        total += 1
        output_path = normal_prediction_path(data)
        valid_existing, _, _ = validate_prediction_file(output_path, data)
        if valid_existing:
            print(f"{data} | ja_validado")
            continue
        started = time.perf_counter()
        daily_path = DAILY_DIR / f"jogos_do_dia_{data}.csv"
        row: dict[str, Any] = {
            "data": data,
            "status": "",
            "tempo_seg": 0.0,
            "jogos": 0,
            "previsoes": 0,
            "apostar_true": 0,
            "entrada_simulada_true": 0,
            "operacao_real_true": 0,
            "recomendacao_operacional_true": 0,
            "arquivo_daily": str(daily_path.relative_to(ROOT)),
            "arquivo_previsoes": str(output_path.relative_to(ROOT)),
            "erro": "",
        }

        write_state(
            status="em_execucao",
            etapa_atual="processando_data",
            start_date=args.start,
            end_date=args.end,
            all_dates=all_dates,
            completed_dates=completed_dates,
            error_dates=error_dates,
            ignored_dates=ignored_dates,
            valid_files=valid_files,
            invalid_files=invalid_files,
            totals=totals,
            inicio_execucao=inicio_execucao,
            started_perf=started_perf,
            mensagem=f"Processando {data}.",
            current_next=data,
        )

        valid, games, reason = validate_daily(daily_path)
        row["jogos"] = games
        if not valid:
            row["status"] = "ignorado_daily_invalido"
            row["erro"] = reason
            row["tempo_seg"] = round(time.perf_counter() - started, 3)
            rows.append(row)
            save_report(rows)
            ignored_dates.append(data)
            write_state(
                status="em_execucao",
                etapa_atual="data_ignorada",
                start_date=args.start,
                end_date=args.end,
                all_dates=all_dates,
                completed_dates=completed_dates,
                error_dates=error_dates,
                ignored_dates=ignored_dates,
                valid_files=valid_files,
                invalid_files=invalid_files,
                totals=totals,
                inicio_execucao=inicio_execucao,
                started_perf=started_perf,
                mensagem=f"{data} ignorada: {reason}.",
            )
            print(f"{data} | {row['status']} | {reason}")
            continue

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--date", data]
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        metrics = prediction_metrics(output_path)
        row.update(metrics)
        row["tempo_seg"] = round(time.perf_counter() - started, 3)

        if result.returncode != 0:
            row["status"] = "erro_predict"
            row["erro"] = (result.stderr or result.stdout or f"exit_code_{result.returncode}")[-2000:]
            error_dates.append(data)
        elif metrics["previsoes"] <= 0:
            row["status"] = "sem_previsoes_validas"
            row["erro"] = "predict_ok_sem_csv_ou_csv_vazio"
            error_dates.append(data)
        else:
            row["status"] = "ok"
            ok += 1
            completed_dates.append(data)
            valid_files.append(str(output_path.relative_to(ROOT)))
            totals["previsoes"] += int(metrics.get("previsoes", 0))
            totals["apostar_true"] += int(metrics.get("apostar_true", 0))
            totals["entrada_simulada_true"] += int(metrics.get("entrada_simulada_true", 0))

        rows.append(row)
        rows = sorted(rows, key=lambda item: str(item.get("data", "")))
        save_report(rows)
        write_state(
            status="em_execucao",
            etapa_atual="data_concluida" if row["status"] == "ok" else "data_com_erro",
            start_date=args.start,
            end_date=args.end,
            all_dates=all_dates,
            completed_dates=completed_dates,
            error_dates=error_dates,
            ignored_dates=ignored_dates,
            valid_files=valid_files,
            invalid_files=invalid_files,
            totals=totals,
            inicio_execucao=inicio_execucao,
            started_perf=started_perf,
            mensagem=f"{data}: {row['status']}.",
        )
        print(
            f"{data} | {row['status']} | jogos={row['jogos']} | "
            f"previsoes={row['previsoes']} | apostar={row['apostar_true']} | "
            f"{row['tempo_seg']:.1f}s"
        )

    write_state(
        status="concluido",
        etapa_atual="finalizado",
        start_date=args.start,
        end_date=args.end,
        all_dates=all_dates,
        completed_dates=completed_dates,
        error_dates=error_dates,
        ignored_dates=ignored_dates,
        valid_files=valid_files,
        invalid_files=invalid_files,
        totals=totals,
        inicio_execucao=inicio_execucao,
        started_perf=started_perf,
        mensagem="Regeneração finalizada.",
    )
    print(f"Regeneracao finalizada: {len(set(completed_dates))}/{len(all_dates)} dias ok")
    print(f"Relatorio: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
