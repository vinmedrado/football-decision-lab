#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# Este arquivo fica em 04_ml/core/perfil_operacional.py.
# Assim funciona tanto rodando a partir da raiz quanto a partir de 04_ml.
ML_DIR = Path(__file__).resolve().parents[1]
PERFIL_PATH = ML_DIR / "reports" / "perfil_operacional_mercados.json"

DEFAULT_ODD_MIN = 1.20
DEFAULT_ODD_MAX = 3.50
DEFAULT_MIN_EV = 0.05


def carregar_perfil_operacional():
    if not PERFIL_PATH.exists():
        return {}

    try:
        data = json.loads(PERFIL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return {
        str(m).upper(): cfg
        for m, cfg in data.items()
        if isinstance(cfg, dict)
    }


def mercado_ativo(mercado: str) -> bool:
    perfil = carregar_perfil_operacional()
    cfg = perfil.get(str(mercado).upper())
    return bool(cfg and cfg.get("ativo") is True)


def config_mercado(mercado: str) -> dict:
    perfil = carregar_perfil_operacional()
    cfg = perfil.get(str(mercado).upper(), {})
    motivo = str(cfg.get("motivo", ""))
    ativo = bool(cfg.get("ativo") is True)

    return {
        "ativo": ativo,
        "odd_min": float(cfg.get("odd_min", DEFAULT_ODD_MIN)),
        "odd_max": float(cfg.get("odd_max", DEFAULT_ODD_MAX)),
        "min_ev": float(cfg.get("min_ev", DEFAULT_MIN_EV)),
        "status": cfg.get("status", "SEM_PERFIL"),
        "motivo": motivo,
        "roi_policy_ok": bool(ativo or (cfg and "LIGA_NAO_ATIVA" not in motivo)),
    }


def mercados_ativos():
    perfil = carregar_perfil_operacional()
    return {
        m: cfg
        for m, cfg in perfil.items()
        if cfg.get("ativo") is True
    }
