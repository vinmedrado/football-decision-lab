#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Football Lab — FASE 11 Responsible Mode

Camada de segurança para impedir execução operacional automática.
O projeto passa a funcionar como simulação/análise por padrão.

Para análise local, os scripts continuam gerando relatórios, dashboards e métricas.
Para qualquer rotina de preenchimento automático, o padrão é bloquear.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / 'reports'
ALERTS_DIR = BASE_DIR / 'alerts'
CONFIG_PATH = BASE_DIR / 'config' / 'responsible_mode.json'
STATE_PATH = REPORTS_DIR / 'estado_modo_responsavel.json'
BLOCK_PATH = ALERTS_DIR / 'AUTO_FILL_BLOQUEADA.txt'

DEFAULT_CONFIG = {
    'mode': 'simulation_only',
    'allow_auto_fill': False,
    'allow_real_bet_registration': False,
    'allow_paper_auto_fill': True,
    'require_manual_export': True,
    'motivo': 'Projeto configurado para análise, auditoria e simulação. Auto-fill operacional fica bloqueado por padrão.',
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except Exception:
        data = DEFAULT_CONFIG.copy()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(data or {})
    return cfg


def is_auto_fill_allowed() -> tuple[bool, str]:
    cfg = load_config()
    env_override = os.getenv('FOOTBALL_LAB_ALLOW_AUTO_FILL', '').strip().lower()
    if env_override in {'1', 'true', 'yes', 'sim'}:
        return False, 'Bloqueado: override por variável de ambiente não é aceito nesta fase segura.'
    allowed = bool(cfg.get('allow_auto_fill')) and bool(cfg.get('allow_real_bet_registration'))
    if allowed:
        return False, 'Bloqueado: configuração operacional real não é permitida nesta versão segura.'
    return False, str(cfg.get('motivo') or DEFAULT_CONFIG['motivo'])


def is_paper_fill_allowed() -> tuple[bool, str]:
    """Autoriza somente o registro contábil da simulação prospectiva."""
    cfg = load_config()
    if bool(cfg.get('allow_real_bet_registration')) or bool(cfg.get('allow_auto_fill')):
        return False, 'Configuração insegura: paper exige apostas reais e auto-fill real desativados.'
    if str(cfg.get('mode', '')).strip().lower() not in {'simulation_only', 'paper_only'}:
        return False, 'Modo responsável incompatível com paper trading.'
    if not bool(cfg.get('allow_paper_auto_fill')):
        return False, 'Registro automático da banca paper está desativado.'
    return True, 'Registro autorizado apenas na banca simulada paper.'


def write_state(action: str, blocked: bool, motivo: str) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'phase': 'FASE 11 — Responsible Simulation Lock',
        'action': action,
        'blocked': blocked,
        'motivo': motivo,
        'mode': load_config().get('mode', 'simulation_only'),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    if blocked:
        BLOCK_PATH.write_text(
            f"AUTO-FILL BLOQUEADO\nData: {state['generated_at']}\nMotivo: {motivo}\n",
            encoding='utf-8',
        )
    return state


def assert_auto_fill_allowed(action: str = 'auto_fill') -> None:
    allowed, motivo = is_auto_fill_allowed()
    write_state(action=action, blocked=not allowed, motivo=motivo)
    if not allowed:
        raise SystemExit(f'🛡️ MODO SIMULAÇÃO ATIVO — {motivo}')


def assert_paper_fill_allowed(action: str = 'paper_auto_fill') -> None:
    allowed, motivo = is_paper_fill_allowed()
    write_state(action=action, blocked=not allowed, motivo=motivo)
    if not allowed:
        raise SystemExit(f'MODO PAPER BLOQUEADO — {motivo}')


if __name__ == '__main__':
    allowed, motivo = is_auto_fill_allowed()
    state = write_state('status_check', blocked=not allowed, motivo=motivo)
    print(json.dumps(state, indent=2, ensure_ascii=False))
