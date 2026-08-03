#!/usr/bin/env python3
"""Captura previsões paper uma única vez por jogo, perto do kickoff."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from paper_model_manager import bundle_dir, load_config, validate

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DAILY_DIR = ROOT_DIR / "data" / "raw" / "futpython" / "daily"
OUTPUT_ROOT = BASE_DIR / "previsoes_paper"


def bool_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return normalized.isin({"true", "1", "sim", "s", "yes", "y"}) | numeric.eq(1)


def load_predict_module():
    path = BASE_DIR / "03_predict.py"
    spec = importlib.util.spec_from_file_location("football_lab_predict_paper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_path(config: dict, data_ref: str) -> Path:
    return OUTPUT_ROOT / str(config["cycle_id"]) / f"previsoes_{data_ref}.csv"


def _existing_is_valid(path: Path, config: dict) -> bool:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    if frame.empty:
        return True
    return (
        "paper_cycle_id" in frame.columns
        and frame["paper_cycle_id"].astype(str).eq(str(config["cycle_id"])).all()
        and "paper_policy_version" in frame.columns
        and frame["paper_policy_version"].astype(str).eq(str(config["policy_version"])).all()
    )


def _kickoff_series(frame: pd.DataFrame, timezone: ZoneInfo) -> pd.Series:
    kickoff = pd.to_datetime(frame.get("kickoff_at"), errors="coerce")
    if getattr(kickoff.dt, "tz", None) is None:
        return kickoff.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return kickoff.dt.tz_convert(timezone)


def _raw_has_game_in_window(config: dict, data_ref: str, now: datetime) -> bool:
    """Evita carregar o modelo quando não há partida perto da janela de captura."""
    path = DAILY_DIR / f"jogos_do_dia_{data_ref}.csv"
    if not path.exists():
        return True
    try:
        raw = pd.read_csv(path, usecols=["Date", "Time"], low_memory=False)
    except (ValueError, pd.errors.EmptyDataError):
        return True
    kickoff = pd.to_datetime(
        raw["Date"].fillna("").astype(str).str.strip() + " " + raw["Time"].fillna("").astype(str).str.strip(),
        errors="coerce",
    ).dt.tz_localize(now.tzinfo, ambiguous="NaT", nonexistent="NaT")
    minutes = (kickoff - now).dt.total_seconds() / 60
    return bool(minutes.between(
        float(config["capture_window_min_minutes"]),
        float(config["capture_window_max_minutes"]),
        inclusive="both",
    ).any())


def _row_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ["data", "home", "away", "mercado"]:
        if column in frame.columns:
            parts.append(frame[column].fillna("").astype(str).str.strip().str.lower())
        else:
            parts.append(pd.Series("", index=frame.index))
    return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate(data_ref: str, now: datetime | None = None) -> Path:
    config = load_config()
    validate()
    timezone = ZoneInfo(str(config["timezone"]))
    generated_at = now.astimezone(timezone) if now else datetime.now(timezone)
    destination = output_path(config, data_ref)
    if destination.exists() and not _existing_is_valid(destination, config):
        raise RuntimeError(f"Snapshot existente inválido; revisão manual necessária: {destination}")
    if not _raw_has_game_in_window(config, data_ref, generated_at):
        print(
            "Nenhum jogo dentro da janela de captura "
            f"{config['capture_window_min_minutes']}–{config['capture_window_max_minutes']} minutos."
        )
        return destination

    bundle = bundle_dir(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".raw_{destination.name}.{os.getpid()}.tmp")
    predict = load_predict_module()
    predict.MODEL_DIR = bundle / "models"
    predict.DATASET_DIR = bundle / "datasets"
    predict.RESUMO_MODELOS_PATH = predict.MODEL_DIR / "resumo_modelos.pkl"
    predict.ENCODERS_PATH = predict.DATASET_DIR / "label_encoders.pkl"
    predict.historical_prediction_path = lambda *_args, **_kwargs: staging

    try:
        predict.prever(data_ref, modo_auditoria=True)
        frame = pd.read_csv(staging, low_memory=False) if staging.exists() else pd.DataFrame()
    finally:
        if staging.exists() and staging.stat().st_size == 0:
            staging.unlink()

    if frame.empty:
        if staging.exists():
            staging.unlink()
        print("Nenhuma previsão tecnicamente válida nesta janela.")
        return destination

    model_map = {str(item["market"]).upper(): item for item in config["models"]}
    frame["mercado_key"] = frame["mercado"].astype(str).str.upper().str.strip()
    frame = frame[frame["mercado_key"].isin(model_map)].copy()
    frame["paper_threshold"] = frame["mercado_key"].map(lambda value: float(model_map[value]["threshold"]))
    frame["paper_odd_min"] = frame["mercado_key"].map(lambda value: float(model_map[value]["odd_min"]))
    frame["paper_odd_max"] = frame["mercado_key"].map(lambda value: float(model_map[value]["odd_max"]))
    frame["odd"] = pd.to_numeric(frame.get("odd"), errors="coerce")
    frame["prob_evento"] = pd.to_numeric(frame.get("prob_evento"), errors="coerce")
    kickoff = _kickoff_series(frame, timezone)
    frame["paper_minutes_before_kickoff"] = (kickoff - generated_at).dt.total_seconds() / 60
    frame["paper_capture_window_ok"] = frame["paper_minutes_before_kickoff"].between(
        float(config["capture_window_min_minutes"]),
        float(config["capture_window_max_minutes"]),
        inclusive="both",
    )
    frame = frame[frame["paper_capture_window_ok"]].copy()
    if frame.empty:
        if staging.exists():
            staging.unlink()
        print("Os jogos brutos estavam na janela, mas nenhum passou pela normalização/features.")
        return destination

    real_odd = bool_mask(frame["tem_odd_real"]) if "tem_odd_real" in frame.columns else pd.Series(False, index=frame.index)
    frame["paper_lead_time_ok"] = True
    frame["paper_odd_ok"] = real_odd & frame["odd"].between(frame["paper_odd_min"], frame["paper_odd_max"], inclusive="both")
    frame["paper_probability_ok"] = frame["prob_evento"].ge(frame["paper_threshold"])
    frame["paper_signal"] = frame["paper_odd_ok"] & frame["paper_probability_ok"]

    eligible = frame[frame["paper_signal"]].copy()
    eligible["_paper_ev"] = pd.to_numeric(eligible.get("ev"), errors="coerce").fillna(-999)
    keep = eligible.sort_values("_paper_ev", ascending=False).drop_duplicates(["data", "home", "away"], keep="first").index
    frame.loc[frame["paper_signal"] & ~frame.index.isin(keep), "paper_signal"] = False
    frame["apostar"] = frame["paper_signal"]
    frame["entrada_simulada"] = frame["paper_signal"]
    frame["operacao_real"] = False
    frame["recomendacao_operacional"] = False
    frame["modo_paper"] = True
    frame["origem"] = "paper_forward"
    frame["paper_cycle_id"] = str(config["cycle_id"])
    frame["paper_policy_version"] = str(config["policy_version"])
    frame["paper_generated_at"] = generated_at.isoformat(timespec="seconds")
    frame["paper_capture_id"] = generated_at.strftime("%Y%m%dT%H%M%S%z")
    frame["paper_model_registry_id"] = frame["mercado_key"].map(lambda value: model_map[value]["registry_id"])
    frame["paper_model_sha256"] = frame["mercado_key"].map(lambda value: model_map[value]["model_sha256"])
    frame["motivo_nao_apostar"] = frame.apply(
        lambda row: "ok" if bool(row["paper_signal"]) else (
            "ODD_REAL_FORA_DA_POLITICA" if not bool(row["paper_odd_ok"]) else "ABAIXO_THRESHOLD_PAPER"
        ),
        axis=1,
    )
    captured = frame.drop(columns=["mercado_key"], errors="ignore")

    existing = pd.read_csv(destination, low_memory=False) if destination.exists() else pd.DataFrame()
    existing_keys = set(_row_key(existing)) if not existing.empty else set()
    captured = captured.loc[~_row_key(captured).isin(existing_keys)].copy()
    if captured.empty:
        if staging.exists():
            staging.unlink()
        print("Jogos da janela já possuíam captura imutável; nenhuma linha foi alterada.")
        return destination

    all_columns = list(dict.fromkeys([*existing.columns.tolist(), *captured.columns.tolist()]))
    existing = existing.reindex(columns=all_columns)
    captured = captured.reindex(columns=all_columns)
    combined = pd.concat([existing, captured], ignore_index=True, sort=False)
    _atomic_csv(combined, destination)
    if staging.exists():
        staging.unlink()
    signals = int(bool_mask(captured["paper_signal"]).sum())
    print(f"Capturas novas: {len(captured)} | sinais paper: {signals} | apostas reais: 0")
    print(f"Snapshot acumulado e imutável por jogo: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    pd.Timestamp(args.date)
    generate(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
