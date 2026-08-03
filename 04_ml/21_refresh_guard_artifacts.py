#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper de compatibilidade mantido temporariamente.

Novo módulo: controles/operacao/atualizar_status.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "controles/operacao/atualizar_status.py"), run_name="__main__")
