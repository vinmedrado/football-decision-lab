#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove artefatos de build/cache antes de distribuir o pacote."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PATTERN_FILES = ["*.pyc", "*.pyo"]
PATTERN_DIRS = ["__pycache__"]


def clean_distribution(root: Path = ROOT_DIR) -> dict:
    removed_dirs = []
    removed_files = []
    for name in PATTERN_DIRS:
        for path in root.rglob(name):
            if path.is_dir():
                removed_dirs.append(str(path.relative_to(root)))
                shutil.rmtree(path, ignore_errors=True)
    for pattern in PATTERN_FILES:
        for path in root.rglob(pattern):
            if path.is_file():
                removed_files.append(str(path.relative_to(root)))
                path.unlink(missing_ok=True)
    return {"removed_dirs": removed_dirs, "removed_files": removed_files}


if __name__ == "__main__":
    print(json.dumps(clean_distribution(), ensure_ascii=False, indent=2))
