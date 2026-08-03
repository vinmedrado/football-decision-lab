#!/usr/bin/env python3
"""Registra somente sinais do snapshot paper na banca simulada."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "paper_mode.json"
HISTORY_PATH = BASE_DIR / "banca" / "historico_apostas.csv"


def load_guard():
    path = BASE_DIR / "11_responsible_mode.py"
    spec = importlib.util.spec_from_file_location("responsible_mode_paper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Não foi possível carregar o modo responsável.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return normalized.isin({"true", "1", "sim", "s", "yes", "y"}) | numeric.eq(1)


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prediction_path(config: dict, data_ref: str) -> Path:
    return BASE_DIR / "previsoes_paper" / str(config["cycle_id"]) / f"previsoes_{data_ref}.csv"


def captured_before_kickoff(row: pd.Series, timezone: ZoneInfo) -> bool:
    """Confirma que o sinal ficou imutável antes do início da partida."""
    try:
        generated_at = pd.Timestamp(row.get("paper_generated_at"))
        kickoff_at = pd.Timestamp(row.get("kickoff_at"))
        if pd.isna(generated_at) or pd.isna(kickoff_at):
            return False
        if generated_at.tzinfo is None:
            generated_at = generated_at.tz_localize(timezone)
        else:
            generated_at = generated_at.tz_convert(timezone)
        if kickoff_at.tzinfo is None:
            kickoff_at = kickoff_at.tz_localize(timezone)
        else:
            kickoff_at = kickoff_at.tz_convert(timezone)
        return bool(generated_at < kickoff_at)
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--recover-immutable-captures",
        action="store_true",
        help=(
            "Recupera contabilmente sinais já capturados antes do kickoff. "
            "As entradas ficam marcadas como recuperação operacional."
        ),
    )
    args = parser.parse_args()
    pd.Timestamp(args.date)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("enabled") or config.get("mode") != "paper_only" or config.get("allow_real_bets") is not False:
        raise RuntimeError("Configuração paper inválida ou insegura.")
    load_guard().assert_paper_fill_allowed("registrar_banca_paper")

    source = prediction_path(config, args.date)
    if not source.exists():
        print("Ainda não há captura paper nesta data; histórico preservado.")
        return 0
    predictions = pd.read_csv(source, low_memory=False)
    if predictions.empty:
        print("Snapshot paper vazio; histórico preservado.")
        return 0
    required = {"paper_signal", "paper_cycle_id", "paper_policy_version", "paper_generated_at", "paper_model_sha256"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RuntimeError(f"Snapshot sem colunas de proveniência: {missing}")
    if not predictions["paper_cycle_id"].astype(str).eq(str(config["cycle_id"])).all():
        raise RuntimeError("Snapshot pertence a outro ciclo paper.")

    selected = predictions[bool_mask(predictions["paper_signal"])].copy()
    if selected.empty:
        print("Nenhum sinal paper no dia; histórico preservado.")
        return 0
    timezone = ZoneInfo(str(config["timezone"]))
    prospective_mask = selected.apply(
        lambda row: captured_before_kickoff(row, timezone),
        axis=1,
    )
    if not prospective_mask.all():
        rejected = int((~prospective_mask).sum())
        print(f"Sinais sem prova de captura anterior ao kickoff ignorados: {rejected}")
        selected = selected[prospective_mask].copy()
    if selected.empty:
        print("Nenhum sinal possui captura prospectiva válida; histórico preservado.")
        return 0

    now = pd.Timestamp(datetime.now(timezone))
    kickoff = pd.to_datetime(selected["kickoff_at"], errors="coerce")
    if getattr(kickoff.dt, "tz", None) is None:
        kickoff = kickoff.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        kickoff = kickoff.dt.tz_convert(timezone)
    if not args.recover_immutable_captures:
        future_mask = kickoff.gt(now)
        late_count = int((~future_mask).sum())
        if late_count:
            print(
                "Sinais não registrados cujo jogo já começou foram preservados fora da banca: "
                f"{late_count}. Use a recuperação auditável somente após revisar o incidente."
            )
        selected = selected[future_mask].copy()
    if selected.empty:
        print("Nenhum sinal novo pode ser registrado neste horário; histórico preservado.")
        return 0

    selected["odd"] = pd.to_numeric(selected["odd"], errors="coerce")
    selected["ev_sort"] = pd.to_numeric(selected.get("ev"), errors="coerce").fillna(-999)
    selected = selected[selected["odd"].notna()].sort_values("ev_sort", ascending=False)
    selected = selected.drop_duplicates(["data", "home", "away"], keep="first")
    exposure_limit = float(config["paper_bank_initial"]) * float(config["max_daily_exposure_pct"])
    max_by_exposure = int(exposure_limit // float(config["stake_fixed"]))
    daily_limit = min(int(config["max_bets_per_day"]), max_by_exposure)

    history = pd.read_csv(HISTORY_PATH, low_memory=False) if HISTORY_PATH.exists() else pd.DataFrame()
    if history.empty:
        existing_keys: set[str] = set()
        paper_history = pd.DataFrame()
    else:
        origins = norm(history["origem"]) if "origem" in history.columns else pd.Series("", index=history.index)
        cycles = norm(history["paper_cycle_id"]) if "paper_cycle_id" in history.columns else pd.Series("", index=history.index)
        paper_mask = origins.eq("paper_forward") & cycles.eq(str(config["cycle_id"]).strip().lower())
        paper_history = history.loc[paper_mask].copy()
        existing_keys = set(
            norm(history.loc[paper_mask, "data"]) + "|"
            + norm(history.loc[paper_mask, "home"]) + "|"
            + norm(history.loc[paper_mask, "away"]) + "|"
            + norm(history.loc[paper_mask, "mercado"]) + "|"
            + cycles.loc[paper_mask]
        )

    registered_at = datetime.now(timezone).isoformat(timespec="seconds")
    prediction_hash = file_sha256(source)
    daily_counts: defaultdict[str, int] = defaultdict(int)
    league_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    market_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    daily_exposure: defaultdict[str, float] = defaultdict(float)
    if not paper_history.empty:
        paper_history["_data_key"] = norm(paper_history["data"])
        paper_history["_liga_key"] = norm(paper_history["liga"])
        paper_history["_mercado_key"] = norm(paper_history["mercado"])
        paper_history["_stake"] = pd.to_numeric(paper_history.get("valor_apostado"), errors="coerce").fillna(0)
        daily_counts.update(paper_history.groupby("_data_key").size().astype(int).to_dict())
        league_counts.update(paper_history.groupby(["_data_key", "_liga_key"]).size().astype(int).to_dict())
        market_counts.update(paper_history.groupby(["_data_key", "_mercado_key"]).size().astype(int).to_dict())
        daily_exposure.update(paper_history.groupby("_data_key")["_stake"].sum().astype(float).to_dict())

    records = []
    for _, row in selected.iterrows():
        data_key = str(row.get("data", "")).strip().lower()
        league_key = str(row.get("liga", "")).strip().lower()
        market_key = str(row.get("mercado", "")).strip().lower()
        key = "|".join([
            data_key,
            str(row.get("home", "")).strip().lower(),
            str(row.get("away", "")).strip().lower(),
            market_key,
            str(config["cycle_id"]).strip().lower(),
        ])
        if key in existing_keys:
            continue
        stake_fixed = float(config["stake_fixed"])
        if daily_counts[data_key] >= daily_limit:
            continue
        if daily_exposure[data_key] + stake_fixed > exposure_limit + 1e-9:
            continue
        if league_counts[(data_key, league_key)] >= int(config["max_bets_per_league"]):
            continue
        if market_counts[(data_key, market_key)] >= int(config["max_bets_per_market"]):
            continue
        row_payload = {
            str(column): (None if pd.isna(value) else value)
            for column, value in row.to_dict().items()
        }
        row_hash = hashlib.sha256(
            json.dumps(row_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        records.append({
            "data": row.get("data", args.date),
            "liga": row.get("liga", ""),
            "Round": row.get("Round", ""),
            "jogo": f"{row.get('home', '')} x {row.get('away', '')}",
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "mercado": row.get("mercado", ""),
            "event": row.get("event", row.get("mercado", "")),
            "prob_modelo": row.get("prob_modelo", row.get("prob_evento", "")),
            "confianca": row.get("prob_evento", row.get("confianca", "")),
            "odd": row.get("odd", ""),
            "valor_apostado": stake_fixed,
            "kelly_pct": 0.0,
            "roi_bt": row.get("roi_bt", ""),
            "resultado": "pendente",
            "lucro": 0.0,
            "banca_apos": "",
            "base_match": "",
            "origem": "paper_forward",
            "paper_cycle_id": config["cycle_id"],
            "paper_policy_version": config["policy_version"],
            "paper_model_registry_id": row.get("paper_model_registry_id", ""),
            "paper_model_sha256": row.get("paper_model_sha256", ""),
            "paper_prediction_sha256": prediction_hash,
            "paper_prediction_row_sha256": row_hash,
            "paper_prediction_file": str(source.relative_to(BASE_DIR.parent)),
            "paper_generated_at": row.get("paper_generated_at", ""),
            "kickoff_at": row.get("kickoff_at", ""),
            "paper_registered_at": registered_at,
            "paper_capture_was_prospective": True,
            "paper_registration_recovered": bool(args.recover_immutable_captures),
            "paper_recovery_reason": (
                "CORRECAO_FLAG_NUMERICA_1_0_2026_07_27"
                if args.recover_immutable_captures else ""
            ),
            "operacao_real": False,
        })
        existing_keys.add(key)
        daily_counts[data_key] += 1
        league_counts[(data_key, league_key)] += 1
        market_counts[(data_key, market_key)] += 1
        daily_exposure[data_key] += stake_fixed

    if not records:
        print("Todos os sinais paper já estavam registrados; histórico preservado.")
        return 0
    additions = pd.DataFrame(records)
    for column in sorted(set(history.columns).union(additions.columns)):
        if column not in history.columns:
            history[column] = ""
        if column not in additions.columns:
            additions[column] = ""
    additions = additions[history.columns] if len(history.columns) else additions
    updated = pd.concat([history, additions], ignore_index=True, sort=False)
    if HISTORY_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = HISTORY_PATH.with_name(f"{HISTORY_PATH.stem}.backup_paper_{stamp}{HISTORY_PATH.suffix}")
        backup.write_bytes(HISTORY_PATH.read_bytes())
    atomic_write(updated, HISTORY_PATH)
    print(f"Entradas paper registradas: {len(additions)}")
    if args.recover_immutable_captures:
        print("Recuperação auditável: sim; todas as capturas foram validadas antes do kickoff.")
    print(f"Histórico: {HISTORY_PATH}")
    print("Apostas reais registradas: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
