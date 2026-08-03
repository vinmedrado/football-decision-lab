#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 15 — Calibration Recovery Engine.

Treina calibradores por mercado com holdout temporal e só recomenda uso quando
há melhora real em Brier/calibration error. Não altera o modelo bruto.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.project_paths import REPORTS_DIR, ROOT_DIR as PROJECT_ROOT, read_json, write_json
from core.result_normalizer import normalize_result

ML_DIR = PROJECT_ROOT / "04_ml"
BANCA_DIR = ML_DIR / "banca"
CALIBRATORS_DIR = ML_DIR / "calibrators"
CURVES_DIR = REPORTS_DIR / "calibration_curves"
HISTORY_CANDIDATES = [BANCA_DIR / "historico_apostas.csv", ML_DIR / "historico_apostas.csv"]
BEST_PATH = REPORTS_DIR / "melhor_calibrador_por_mercado.json"
RECOVERY_PATH = REPORTS_DIR / "relatorio_recuperacao_calibracao.json"
GUARD_PATH = REPORTS_DIR / "estado_guard_calibracao.json"
MIN_MARKET_RECORDS = 30
CALIBRATION_LIMIT = 0.13


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_sep(path: Path) -> str:
    try:
        line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return ","
    return ";" if line.count(";") > line.count(",") else ","


def load_history() -> tuple[pd.DataFrame, str | None]:
    for path in HISTORY_CANDIDATES:
        if not path.exists():
            continue
        for enc in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=detect_sep(path), encoding=enc, low_memory=False)
                if len(df.columns) == 1 and "," in str(df.columns[0]):
                    df = pd.read_csv(path, sep=",", encoding=enc, low_memory=False)
                return df.dropna(how="all"), str(path)
            except Exception:
                continue
    return pd.DataFrame(), None


def find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in by_lower:
            return by_lower[n.lower()]
    return None


def parse_prob(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    num = pd.to_numeric(out, errors="coerce")
    num = num.where(num <= 1, num / 100.0)
    return num.clip(0, 1)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(abs(float(np.mean(y_prob)) - float(np.mean(y_true))))


def stability_penalty(y_prob: np.ndarray) -> float:
    if len(y_prob) < 2:
        return 1.0
    return float(max(0.0, 0.05 - np.std(y_prob)))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == bins - 1 else y_prob < hi)
        count = int(mask.sum())
        pred = float(np.mean(y_prob[mask])) if count else None
        actual = float(np.mean(y_true[mask])) if count else None
        gap = abs(pred - actual) if count else None
        rows.append({
            "prob_bin": f"{lo:.1f}-{hi:.1f}",
            "predicted_mean": round(pred, 6) if pred is not None else None,
            "actual_rate": round(actual, 6) if actual is not None else None,
            "count": count,
            "gap": round(gap, 6) if gap is not None else None,
        })
    return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True)
    n = len(df)
    a = int(n * 0.70)
    b = int(n * 0.85)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def fit_platt(x: np.ndarray, y: np.ndarray) -> Any | None:
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(solver="lbfgs")
    model.fit(x.reshape(-1, 1), y)
    return model

def mercado_modelo_ja_calibrado(mercado: str) -> bool:
    model_path = ML_DIR / "models" / str(mercado).strip() / "model.pkl"
    if not model_path.exists():
        return False

    try:
        with model_path.open("rb") as f:
            model = pickle.load(f)
        return isinstance(model, CalibratedClassifierCV)
    except Exception:
        return False

def apply_model(model: Any, method: str, probs: np.ndarray) -> np.ndarray:
    if method == "platt":
        return model.predict_proba(probs.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return model.predict(probs)
    return probs


def score_candidate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_error": calibration_error(y_true, y_prob),
        "stability_penalty": stability_penalty(y_prob),
    }


def run_recovery() -> dict[str, Any]:
    CALIBRATORS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, source = load_history()
    generated_at = now_iso()
    if df.empty:
        report = {"generated_at": generated_at, "status": "FALHA", "motivo": "NO_HISTORY", "mercados_checked": 0, "source": source}
        write_json(RECOVERY_PATH, report)
        return report

    prob_col = find_col(df, ["probabilidade_calibrada", "prob_modelo", "probabilidade", "prob", "confianca"])
    raw_prob_col = find_col(df, ["prob_modelo", "probabilidade", "prob", "confianca"])
    result_col = find_col(df, ["resultado", "result", "status", "outcome"])
    mercado_col = find_col(df, ["mercado", "mercado"])
    date_col = find_col(df, ["data", "date"])
    if not all([prob_col, raw_prob_col, result_col, mercado_col]):
        report = {"generated_at": generated_at, "status": "FALHA", "motivo": "MISSING_COLUMNS", "columns": list(df.columns)}
        write_json(RECOVERY_PATH, report)
        return report

    work = df.copy()
    if date_col:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values("_date", na_position="first")
    work["_prob"] = parse_prob(work[raw_prob_col])
    work["_result_norm"] = work[result_col].apply(normalize_result)
    work["_y"] = work["_result_norm"].map({"WIN": 1.0, "LOSS": 0.0})
    work["_mercado"] = work[mercado_col].astype(str).str.strip()
    work = work.dropna(subset=["_prob", "_y"])

    if work.empty:
        report = {"generated_at": generated_at, "status": "FALHA", "motivo": "NO_CLOSED_BETS", "source": source}
        write_json(RECOVERY_PATH, report)
        return report

    mercado_results: dict[str, Any] = {}
    recovered = failed = checked = 0
    all_before = []
    all_after = []
    all_y = []

    for mercado, grp in work.groupby("_mercado", sort=False):
        grp = grp.reset_index(drop=True)
        
        if mercado_modelo_ja_calibrado(mercado):
            mercado_results[mercado] = {
                "status": "FALHA",
                "motivo": "MODELO_JA_CALIBRADO_INTERNAMENTE",
                "records": int(len(grp)),
            }
            failed += 1
            continue
        
        if len(grp) < MIN_MARKET_RECORDS or grp["_y"].nunique() < 2:
            mercado_results[mercado] = {"status": "FALHA", "motivo": "INSUFFICIENT_SAMPLE", "records": int(len(grp))}
            failed += 1
            continue
        checked += 1
        _, calib, valid = temporal_split(grp)
        if len(calib) < 5 or len(valid) < 5 or calib["_y"].nunique() < 2 or valid["_y"].nunique() < 2:
            mercado_results[mercado] = {"status": "FALHA", "motivo": "INSUFFICIENT_TEMPORAL_HOLDOUT", "records": int(len(grp))}
            failed += 1
            continue

        x_calib = calib["_prob"].to_numpy(dtype=float)
        y_calib = calib["_y"].to_numpy(dtype=int)
        x_valid = valid["_prob"].to_numpy(dtype=float)
        y_valid = valid["_y"].to_numpy(dtype=int)
        before = score_candidate(y_valid, x_valid)

        candidates = {"baseline": (None, x_valid, before)}
        platt = fit_platt(x_calib, y_calib)
        if platt is not None:
            p = np.clip(apply_model(platt, "platt", x_valid), 0, 1)
            candidates["platt"] = (platt, p, score_candidate(y_valid, p))
        if mercado != "TG_HT_U05":
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_calib, y_calib)
                p = np.clip(apply_model(iso, "isotonic", x_valid), 0, 1)
                candidates["isotonic"] = (iso, p, score_candidate(y_valid, p))
            except Exception:
                pass

        def rank(item):
            method, (_, _, sc) = item
            return (sc["brier"], sc["calibration_error"], sc["stability_penalty"])

        best_method, (best_model, best_probs, best_score) = sorted(candidates.items(), key=rank)[0]
        improved = best_method != "baseline" and best_score["brier"] <= before["brier"] and best_score["calibration_error"] <= before["calibration_error"]
        if improved and best_model is not None:
            path = CALIBRATORS_DIR / f"{mercado}_{best_method}.pkl"
            with path.open("wb") as f:
                pickle.dump({"method": best_method, "mercado": mercado, "model": best_model, "created_at": generated_at}, f)
            reliability_curve(y_valid, best_probs).to_csv(CURVES_DIR / f"{mercado}_reliability.csv", index=False, encoding="utf-8-sig")
            mercado_results[mercado] = {
                "status": "RECOVERED",
                "best_method": best_method,
                "calibrator_path": str(path.relative_to(PROJECT_ROOT)),
                "records": int(len(grp)),
                "validation_records": int(len(valid)),
                "brier_before": round(before["brier"], 6),
                "brier_after": round(best_score["brier"], 6),
                "calibration_error_before": round(before["calibration_error"], 6),
                "calibration_error_after": round(best_score["calibration_error"], 6),
            }
            recovered += 1
            all_after.extend(best_probs.tolist())
        else:
            mercado_results[mercado] = {
                "status": "FALHA",
                "motivo": "RECUPERACAO_CALIBRACAO_FALHOU",
                "records": int(len(grp)),
                "brier_before": round(before["brier"], 6),
                "brier_after": round(best_score["brier"], 6),
                "calibration_error_before": round(before["calibration_error"], 6),
                "calibration_error_after": round(best_score["calibration_error"], 6),
            }
            failed += 1
            all_after.extend(x_valid.tolist())
        all_before.extend(x_valid.tolist())
        all_y.extend(y_valid.tolist())

    global_before = score_candidate(np.array(all_y), np.array(all_before)) if all_y else {"brier": None, "calibration_error": None}
    global_after = score_candidate(np.array(all_y), np.array(all_after)) if all_y else {"brier": None, "calibration_error": None}
    all_skipped_calibrated = (
        checked == 0
        and recovered == 0
        and mercado_results
        and all(r.get("motivo") == "MODELO_JA_CALIBRADO_INTERNAMENTE" for r in mercado_results.values())
    )

    if all_skipped_calibrated:
        status = "OK"
    elif recovered > 0 and global_after.get("calibration_error", 999) <= CALIBRATION_LIMIT:
        status = "OK"
    elif recovered > 0:
        status = "ALERTA"
    else:
        status = "FALHA"
    best = {m: r for m, r in mercado_results.items() if r.get("status") == "RECOVERED"}
    write_json(BEST_PATH, best)
    report = {
        "generated_at": generated_at,
        "status": status,
        "source": source,
        "mercados_checked": checked,
        "mercados_recovered": recovered,
        "mercados_failed": failed,
        "global_before": {"brier": None if global_before["brier"] is None else round(global_before["brier"], 6), "calibration_error": None if global_before["calibration_error"] is None else round(global_before["calibration_error"], 6)},
        "global_after": {"brier": None if global_after["brier"] is None else round(global_after["brier"], 6), "calibration_error": None if global_after["calibration_error"] is None else round(global_after["calibration_error"], 6)},
        "mercados": mercado_results,
    }
    write_json(RECOVERY_PATH, report)

    previous_guard = read_json(GUARD_PATH, {}) or {}
    raw_err = report["global_before"].get("calibration_error")
    cal_err = report["global_after"].get("calibration_error")
    guard = {
        **previous_guard,
        "generated_at": generated_at,
        "raw_calibration_error": raw_err,
        "calibrated_calibration_error": cal_err,
        "calibration_recovery_status": status,
        "blocked": False if all_skipped_calibrated else not (cal_err is not None and cal_err <= CALIBRATION_LIMIT),
        "status": "OK" if all_skipped_calibrated or (cal_err is not None and cal_err <= CALIBRATION_LIMIT) else "CRITICO",
        "motivo": None if all_skipped_calibrated or (cal_err is not None and cal_err <= CALIBRATION_LIMIT) else "RECUPERACAO_CALIBRACAO_FALHOU",
        "limit": CALIBRATION_LIMIT,
    }
    write_json(GUARD_PATH, guard)
    return report


_CALIBRATOR_CACHE: dict[str, tuple[Any | None, str | None]] = {}


def load_mercado_calibrator(mercado: str) -> tuple[Any | None, str | None]:
    mercado_key = str(mercado).strip()

    if mercado_key in _CALIBRATOR_CACHE:
        return _CALIBRATOR_CACHE[mercado_key]

    best = read_json(BEST_PATH, {}) or {}
    info = best.get(mercado_key) or {}
    path_rel = info.get("calibrator_path")
    method = info.get("best_method")

    if not path_rel or not method:
        resultado = (None, None)
        _CALIBRATOR_CACHE[mercado_key] = resultado
        return resultado

    path = PROJECT_ROOT / path_rel

    if not path.exists():
        resultado = (None, None)
        _CALIBRATOR_CACHE[mercado_key] = resultado
        return resultado

    try:
        with path.open("rb") as f:
            payload = pickle.load(f)

        resultado = (payload.get("model"), payload.get("method", method))
        _CALIBRATOR_CACHE[mercado_key] = resultado
        return resultado

    except Exception:
        resultado = (None, None)
        _CALIBRATOR_CACHE[mercado_key] = resultado
        return resultado


def apply_calibrated_probability(mercado: str, raw_probability: float) -> tuple[float, str, bool]:
    model, method = load_mercado_calibrator(mercado)
    raw = float(np.clip(raw_probability, 0, 1))
    if model is None or method is None:
        return raw, "SEM_CALIBRADOR_EXTERNO_PASSTHROUGH", False
    try:
        calibrated = float(np.clip(apply_model(model, method, np.array([raw]))[0], 0, 1))
        return calibrated, method, True
    except Exception:
        return raw, "CALIBRATOR_APPLY_ERRO", False


def calibration_absence_blocks_operation(mercado: str) -> bool:
    return False


def main() -> int:
    report = run_recovery()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") in {"OK", "ALERTA"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
