#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper de compatibilidade mantido temporariamente.

Novo módulo: controles/mercados/auditoria.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "controles/mercados/auditoria.py"), run_name="__main__")
