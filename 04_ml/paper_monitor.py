#!/usr/bin/env python3
"""Métricas prospectivas isoladas do backfill e da banca histórica."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "paper_mode.json"
HISTORY_PATH = BASE_DIR / "banca" / "historico_apostas.csv"
SUMMARY_PATH = BASE_DIR / "reports" / "paper_monitor.json"
DAILY_PATH = BASE_DIR / "reports" / "paper_daily.csv"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_result(value: object) -> float:
    normalized = str(value or "").strip().lower()
    if normalized == "ganhou":
        return 1.0
    if normalized == "perdeu":
        return 0.0
    return math.nan


def max_drawdown(profits: pd.Series, initial_bank: float) -> float:
    curve = initial_bank + profits.cumsum()
    peaks = curve.cummax()
    return float(((curve - peaks) / peaks.replace(0, np.nan)).fillna(0).min()) if len(curve) else 0.0


def roi_bootstrap(profits: np.ndarray, stakes: np.ndarray) -> dict:
    if len(profits) < 2 or stakes.sum() <= 0:
        return {"ci95_low": None, "ci95_high": None, "prob_roi_positive": None}
    rng = np.random.default_rng(20260723)
    indexes = rng.integers(0, len(profits), size=(5000, len(profits)))
    sampled_profit = profits[indexes].sum(axis=1)
    sampled_stake = stakes[indexes].sum(axis=1)
    roi = sampled_profit / np.where(sampled_stake == 0, np.nan, sampled_stake)
    return {
        "ci95_low": round(float(np.nanquantile(roi, 0.025)), 6),
        "ci95_high": round(float(np.nanquantile(roi, 0.975)), 6),
        "prob_roi_positive": round(float(np.nanmean(roi > 0)), 6),
    }


def metrics(frame: pd.DataFrame, initial_bank: float) -> dict:
    if frame.empty:
        return {
            "bets": 0, "settled": 0, "pending": 0, "wins": 0, "losses": 0,
            "stake": 0.0, "profit": 0.0, "roi": None, "winrate": None,
            "drawdown": 0.0, "brier": None, "log_loss": None,
            "calibration_error": None, "ci95_low": None, "ci95_high": None,
            "prob_roi_positive": None,
        }
    result = frame["resultado"].fillna("").astype(str).str.strip().str.lower()
    settled = frame[result.isin(["ganhou", "perdeu"])].copy()
    settled["_y"] = settled["resultado"].map(parse_result)
    settled["_stake"] = pd.to_numeric(settled["valor_apostado"], errors="coerce").fillna(0)
    settled["_profit"] = pd.to_numeric(settled["lucro"], errors="coerce").fillna(0)
    settled["_prob"] = pd.to_numeric(settled.get("confianca"), errors="coerce")
    stake = float(settled["_stake"].sum())
    profit = float(settled["_profit"].sum())
    valid_prob = settled.dropna(subset=["_prob", "_y"]).copy()
    if not valid_prob.empty:
        p = valid_prob["_prob"].clip(1e-9, 1 - 1e-9)
        y = valid_prob["_y"]
        brier = float(((p - y) ** 2).mean())
        log_loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        calibration_error = float(abs(p.mean() - y.mean()))
    else:
        brier = log_loss = calibration_error = None
    bootstrap = roi_bootstrap(settled["_profit"].to_numpy(), settled["_stake"].to_numpy())
    wins = int((settled["_y"] == 1).sum())
    losses = int((settled["_y"] == 0).sum())
    return {
        "bets": int(len(frame)),
        "settled": int(len(settled)),
        "pending": int((result == "pendente").sum()),
        "wins": wins,
        "losses": losses,
        "stake": round(stake, 2),
        "profit": round(profit, 2),
        "roi": round(profit / stake, 6) if stake else None,
        "winrate": round(wins / len(settled), 6) if len(settled) else None,
        "drawdown": round(max_drawdown(settled["_profit"], initial_bank), 6),
        "brier": round(brier, 6) if brier is not None else None,
        "log_loss": round(log_loss, 6) if log_loss is not None else None,
        "calibration_error": round(calibration_error, 6) if calibration_error is not None else None,
        **bootstrap,
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    timezone = ZoneInfo(str(config["timezone"]))
    generated_at = datetime.now(timezone).isoformat(timespec="seconds")
    history = pd.read_csv(HISTORY_PATH, low_memory=False) if HISTORY_PATH.exists() else pd.DataFrame()
    if history.empty or "origem" not in history.columns:
        paper = pd.DataFrame()
    else:
        mask = history["origem"].fillna("").astype(str).str.strip().str.lower().eq("paper_forward")
        if "paper_cycle_id" in history.columns:
            mask &= history["paper_cycle_id"].fillna("").astype(str).eq(str(config["cycle_id"]))
        paper = history[mask].copy()

    overall = metrics(paper, float(config["paper_bank_initial"]))
    n = int(overall["settled"])
    if n < 30:
        status = "COLETANDO_DADOS"
        message = "Amostra inicial; ROI e calibração ainda não devem orientar decisão."
    elif n < 100:
        status = "AMOSTRA_PEQUENA"
        message = "Monitorar, sem recalibrar nem trocar o modelo durante o ciclo."
    elif n < 300:
        status = "EM_VALIDACAO"
        message = "Já permite diagnóstico, mas a decisão principal continua aguardando mais amostra."
    elif overall["ci95_low"] is not None and overall["ci95_low"] > 0:
        status = "EVIDENCIA_POSITIVA"
        message = "Intervalo bootstrap de ROI ficou totalmente acima de zero."
    elif overall["ci95_high"] is not None and overall["ci95_high"] < 0:
        status = "EVIDENCIA_NEGATIVA"
        message = "Intervalo bootstrap de ROI ficou totalmente abaixo de zero."
    else:
        status = "INCONCLUSIVO"
        message = "A amostra cresceu, mas ainda inclui ROI zero no intervalo de incerteza."

    calibration_status = "SEM_AMOSTRA"
    if n >= 50 and overall["calibration_error"] is not None:
        calibration_status = "OK" if overall["calibration_error"] <= 0.10 else "ALERTA"

    daily = pd.DataFrame()
    if not paper.empty:
        settled_mask = paper["resultado"].fillna("").astype(str).str.lower().isin(["ganhou", "perdeu"])
        settled = paper[settled_mask].copy()
        if not settled.empty:
            settled["lucro"] = pd.to_numeric(settled["lucro"], errors="coerce").fillna(0)
            settled["valor_apostado"] = pd.to_numeric(settled["valor_apostado"], errors="coerce").fillna(0)
            daily = settled.groupby("data", dropna=False).agg(
                apostas=("resultado", "size"),
                lucro=("lucro", "sum"),
                stake=("valor_apostado", "sum"),
            ).reset_index()
            daily["roi"] = daily["lucro"] / daily["stake"].replace(0, np.nan)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": generated_at,
        "mode": "PAPER_ONLY",
        "real_bets_allowed": False,
        "cycle_id": config["cycle_id"],
        "policy_version": config["policy_version"],
        "started_at": config["started_at"],
        "review_not_before": config["review_not_before"],
        "status": status,
        "message": message,
        "calibration_status": calibration_status,
        "metrics": overall,
        "sample_progress": {
            "diagnostic_target": 100,
            "decision_target": 300,
            "robust_target": 500,
            "settled": n,
            "decision_pct": round(min(1.0, n / 300), 4),
        },
        "policy": {
            "stake_fixed": config["stake_fixed"],
            "max_bets_per_day": config["max_bets_per_day"],
            "capture_window_minutes": [
                config.get("capture_window_min_minutes"),
                config.get("capture_window_max_minutes"),
            ],
            "settlement_first_attempt_minutes_after_kickoff": config.get(
                "settlement_first_attempt_minutes_after_kickoff"
            ),
            "models": config["models"],
            "automatic_recalibration_during_cycle": False,
            "automatic_model_promotion": False,
        },
    }
    atomic_json(payload, SUMMARY_PATH)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
