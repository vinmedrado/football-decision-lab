# -*- coding: utf-8 -*-
"""Configuração global de console para o Football Lab.
Evita UnicodeEncodeErro no Windows/PowerShell quando scripts imprimem emojis ou acentos.
"""
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
