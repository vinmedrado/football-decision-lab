#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Football Lab — central project paths and boot validation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "04_ml"
REPORTS_DIR = ML_DIR / "reports"
MODELS_DIR = ML_DIR / "models"
MODEL_REGISTRY_DIR = ML_DIR / "model_registry"
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ML_DIR / "config"
BANCA_DIR = ML_DIR / "banca"
ALERTS_DIR = ML_DIR / "alerts"
DASHBOARD_DIR = ML_DIR / "dashboard"

ENVIRONMENT_STATUS_PATH = REPORTS_DIR / "status_ambiente.json"
ENVIRONMENT_EXPLAINABILITY_PATH = REPORTS_DIR / "explicabilidade_ambiente.json"

REQUIRED_DIRECTORIES = {
    "root": ROOT_DIR,
    "ml": ML_DIR,
    "reports": REPORTS_DIR,
    "data": DATA_DIR,
    "config": CONFIG_DIR,
    "banca": BANCA_DIR,
}

OPTIONAL_DIRECTORIES = {
    "models": MODELS_DIR,
    "model_registry": MODEL_REGISTRY_DIR,
    "alerts": ALERTS_DIR,
    "dashboard": DASHBOARD_DIR,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_environment() -> Dict[str, Any]:
    """Validate expected folders and known stale path references.

    This function is intentionally read-only except for creating the reports
    folder and writing status_ambiente.json.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    alertas: List[str] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []
    issues: List[Dict[str, Any]] = []

    for name, path in REQUIRED_DIRECTORIES.items():
        if not path.exists():
            missing_required.append(name)
            sev = "BLOQUEADA" if name in {"reports", "config", "data", "root", "ml", "banca"} else "ALERTA"
            issues.append({
                "component": name,
                "path": str(path),
                "severity": sev,
                "message": f"Diretório obrigatório ausente: {name}.",
            })
    for name, path in OPTIONAL_DIRECTORIES.items():
        if not path.exists():
            missing_optional.append(name)
            sev = "ALERTA" if name in {"models", "model_registry", "alerts", "dashboard"} else "OK"
            issues.append({
                "component": name,
                "path": str(path),
                "severity": sev,
                "message": f"Diretório opcional ausente: {name}.",
            })

    stale_markers = ["/" + "mnt/data/football_lab_phase3/", "/" + "mnt/data/football_lab_next/"]
    stale_refs: List[str] = []
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".py", ".ps1", ".json", ".md", ".txt", ".html"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(marker in text for marker in stale_markers):
                stale_refs.append(str(path.relative_to(ROOT_DIR)))

    if missing_optional:
        alertas.append("OPTIONAL_DIRECTORIES_MISSING:" + ",".join(missing_optional))
    if stale_refs:
        alertas.append("STALE_PATH_REFERENCES_FOUND:" + ",".join(stale_refs[:20]))
        for ref in stale_refs[:20]:
            issues.append({
                "component": "path_validation",
                "path": ref,
                "severity": "ALERTA",
                "message": "Referência a path antigo encontrada.",
            })

    if any(i.get("severity") == "BLOQUEADA" for i in issues):
        env_status = "BLOQUEADA"
    elif any(i.get("severity") == "ALERTA" for i in issues):
        env_status = "ALERTA"
    else:
        env_status = "OK"

    payload = {
        "timestamp": now_iso(),
        "status": env_status,
        "root_dir": str(ROOT_DIR),
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
        "data_dir": str(DATA_DIR),
        "config_dir": str(CONFIG_DIR),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "stale_path_references": stale_refs,
        "issues": issues,
        "alertas": alertas,
    }
    explainability = {
        "timestamp": payload["timestamp"],
        "status": env_status,
        "issues": issues,
        "summary": {
            "blocked": sum(1 for i in issues if i.get("severity") == "BLOQUEADA"),
            "alertas": sum(1 for i in issues if i.get("severity") == "ALERTA"),
            "ok": env_status == "OK",
        },
    }
    write_json(ENVIRONMENT_STATUS_PATH, payload)
    write_json(ENVIRONMENT_EXPLAINABILITY_PATH, explainability)
    return payload


if __name__ == "__main__":
    print(json.dumps(validate_environment(), ensure_ascii=False, indent=2))
