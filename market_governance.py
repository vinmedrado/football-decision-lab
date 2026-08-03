#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Wrapper de compatibilidade.
# Implementação principal: 04_ml/controles/mercados/governanca.py

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ML_DIR = ROOT_DIR / "04_ml"

if ML_DIR.exists() and str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from controles.mercados.governanca import (
    main,
    evaluate_mercado_governance,
)

if __name__ == "__main__":
    main()