#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 8 — Model Registry analítico.

Cria um inventário versionado dos modelos e métricas disponíveis no projeto.
Este script é somente analítico: não treina, não publica e não libera operação.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REGISTRY_DIR = BASE_DIR / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"


def file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def discover_models() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return rows

    active_summary = read_pickle(MODELS_DIR / "resumo_modelos.pkl")
    active_summary = active_summary if isinstance(active_summary, list) else []
    deployable_markets = {
        str(item.get("mercado") or item.get("market") or "").strip()
        for item in active_summary
        if isinstance(item, dict)
    }

    for model_path in sorted(MODELS_DIR.rglob("model.pkl")):
        model_dir = model_path.parent
        meta_path = model_dir / "meta.pkl"
        meta = read_pickle(meta_path)
        meta = meta if isinstance(meta, dict) else {}
        rel_dir = str(model_dir.relative_to(BASE_DIR)).replace("\\", "/")
        mercado = str(meta.get("mercado") or model_dir.name)
        deployable = mercado in deployable_markets
        rows.append({
            "id": hashlib.sha1(rel_dir.encode("utf-8")).hexdigest()[:12],
            "path": rel_dir,
            "model_file": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "meta_file": str(meta_path.relative_to(BASE_DIR)).replace("\\", "/") if meta_path.exists() else None,
            "sha256": file_hash(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds"),
            "mercado": mercado,
            "model_name": meta.get("model_name") or meta.get("model") or "unknown",
            "status": meta.get("status") or "registered",
            "auc": meta.get("auc"),
            "accuracy": meta.get("accuracy"),
            "brier": meta.get("brier"),
            "coverage": meta.get("coverage"),
            "calibration_method": meta.get("calibration_method"),
            "leakage_suspected": bool(meta.get("leakage_suspected", False)),
            "deployable": deployable,
            "artifact_status": "DEPLOYABLE" if deployable else "STALE_NOT_IN_ACTIVE_SUMMARY",
        })
    return rows


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    deployable_models = [model for model in models if model.get("deployable")]
    baseline = read_json(MODELS_DIR / "baseline_metrics.json", {})
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "analytical_registry_only",
        "models_dir": str(MODELS_DIR),
        "models_found": len(models),
        "deployable_models": len(deployable_models),
        "status": "OK" if deployable_models else "NO_DEPLOYABLE_MODEL",
        "models": models,
        "baseline": baseline,
        "note": "Inventário para rastreabilidade. Não altera modelos nem executa apostas.",
    }
    REGISTRY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "relatorio_registro_modelos.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "models_found": len(models),
        "deployable_models": len(deployable_models),
        "status": payload["status"],
        "registry": str(REGISTRY_FILE),
    }, ensure_ascii=False, indent=2))
    return 0 if deployable_models else 2


if __name__ == "__main__":
    raise SystemExit(main())
