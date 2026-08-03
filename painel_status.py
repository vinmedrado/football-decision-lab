# -*- coding: utf-8 -*-
"""
painel_status.py
─────────────────
Ponte simples entre o terminal.py (CLI) e o painel_web (site local).

Cada vez que terminal.py roda uma etapa via executar(), este módulo:
  - grava o estado atual em logs/pipeline_status.json (o que está rodando agora,
    última linha de saída, histórico das últimas execuções);
  - acrescenta a saída bruta em logs/terminal_live.log (para o painel mostrar
    o "processo andando" ao vivo, tipo tail -f).

Não depende de Flask nem de nada do painel_web — só escreve arquivos.
Assim o terminal.py continua funcionando normalmente mesmo se o site
nunca for aberto.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

STATUS_PATH = LOGS_DIR / "pipeline_status.json"
LIVE_LOG_PATH = LOGS_DIR / "terminal_live.log"

MAX_LIVE_LOG_LINES = 1000
MAX_HISTORICO = 25
_MIN_WRITE_INTERVAL = 0.25  # segundos — evita I/O excessivo em loops muito rápidos

_lock = threading.Lock()
_last_write = 0.0
_live_lines: list[str] = []
_estado = {
    "rodando": False,
    "titulo": None,
    "comando": None,
    "iniciado_em": None,
    "atualizado_em": None,
    "ultima_linha": "",
    "rc": None,
    "historico": [],
}


def _carregar_estado_existente():
    if STATUS_PATH.exists():
        try:
            data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            _estado["historico"] = data.get("historico", [])[-MAX_HISTORICO:]
        except Exception:
            pass


def _carregar_log_existente():
    if LIVE_LOG_PATH.exists():
        try:
            linhas = LIVE_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            _live_lines.extend(linhas[-MAX_LIVE_LOG_LINES:])
        except Exception:
            pass


_carregar_estado_existente()
_carregar_log_existente()


def _persistir(force: bool = False):
    global _last_write
    agora = time.monotonic()
    if not force and (agora - _last_write) < _MIN_WRITE_INTERVAL:
        return
    _last_write = agora
    try:
        STATUS_PATH.write_text(json.dumps(_estado, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        LIVE_LOG_PATH.write_text("\n".join(_live_lines[-MAX_LIVE_LOG_LINES:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def iniciar_etapa(titulo: str, comando: list[str] | str):
    with _lock:
        cmd_str = comando if isinstance(comando, str) else " ".join(comando)
        agora = datetime.now().isoformat(timespec="seconds")
        _estado["rodando"] = True
        _estado["titulo"] = titulo
        _estado["comando"] = cmd_str
        _estado["iniciado_em"] = agora
        _estado["atualizado_em"] = agora
        _estado["ultima_linha"] = ""
        _estado["rc"] = None
        _live_lines.append(f"\n$ [{agora}] {titulo}\n$ {cmd_str}")
        _persistir(force=True)


def atualizar_linha(linha: str):
    if not linha:
        return
    with _lock:
        _estado["ultima_linha"] = linha
        _estado["atualizado_em"] = datetime.now().isoformat(timespec="seconds")
        _live_lines.append(linha)
        if len(_live_lines) > MAX_LIVE_LOG_LINES * 2:
            del _live_lines[: len(_live_lines) - MAX_LIVE_LOG_LINES]
        _persistir()


def finalizar_etapa(rc: int):
    with _lock:
        agora = datetime.now().isoformat(timespec="seconds")
        duracao = None
        try:
            inicio = datetime.fromisoformat(_estado["iniciado_em"])
            duracao = round((datetime.now() - inicio).total_seconds(), 1)
        except Exception:
            pass
        _estado["rodando"] = False
        _estado["rc"] = rc
        _estado["atualizado_em"] = agora
        _estado["historico"].append({
            "titulo": _estado["titulo"],
            "iniciado_em": _estado["iniciado_em"],
            "finalizado_em": agora,
            "duracao_s": duracao,
            "rc": rc,
        })
        _estado["historico"] = _estado["historico"][-MAX_HISTORICO:]
        _live_lines.append(f"$ {'✔ OK' if rc == 0 else f'✘ ERRO ({rc})'}: {_estado['titulo']}  [{duracao}s]")
        _persistir(force=True)
