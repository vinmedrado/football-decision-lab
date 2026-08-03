#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper de compatibilidade.

Implementação principal: 04_ml/controles/operacao/atualizar_controles.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "atualizar_controles.py"), run_name="__main__")
