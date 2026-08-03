#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Wrapper de compatibilidade.
# Implementação principal: 04_ml/controles/operacao/controle_operacional.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "04_ml"
for _p in (ROOT_DIR, ML_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controles.operacao.controle_operacional import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        main()  # type: ignore[name-defined]
    except NameError:
        import json
        print(json.dumps(evaluate_operational_guard(), ensure_ascii=False, indent=2))  # type: ignore[name-defined]
