#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Wrapper de compatibilidade.
# Implementação principal: 04_ml/controles/mercados/ciclo_vida.py
from __future__ import annotations

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
ROOT_DIR = ML_DIR.parent
for _p in (ROOT_DIR, ML_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controles.mercados.ciclo_vida import main

if __name__ == "__main__":
    main()
