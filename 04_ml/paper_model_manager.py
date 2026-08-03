#!/usr/bin/env python3
"""Congela e valida o bundle de modelo usado no experimento paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "paper_mode.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not data.get("enabled") or data.get("mode") != "paper_only":
        raise RuntimeError("paper_mode.json não está habilitado em paper_only.")
    if data.get("allow_real_bets") is not False:
        raise RuntimeError("O experimento paper exige allow_real_bets=false.")
    return data


def bundle_dir(config: dict) -> Path:
    return BASE_DIR / "paper_models" / str(config["cycle_id"])


def source_paths(market: str) -> dict[str, Path]:
    return {
        "model": BASE_DIR / "models" / market / "model.pkl",
        "context_models": BASE_DIR / "models" / market / "context_models.pkl",
        "meta": BASE_DIR / "models" / market / "meta.pkl",
        "features": BASE_DIR / "datasets" / market / "feature_columns.pkl",
        "encoders": BASE_DIR / "datasets" / "label_encoders.pkl",
    }


def target_paths(config: dict, market: str) -> dict[str, Path]:
    root = bundle_dir(config)
    return {
        "model": root / "models" / market / "model.pkl",
        "context_models": root / "models" / market / "context_models.pkl",
        "meta": root / "models" / market / "meta.pkl",
        "features": root / "datasets" / market / "feature_columns.pkl",
        "encoders": root / "datasets" / "label_encoders.pkl",
    }


def expected_hashes(model_cfg: dict) -> dict[str, str]:
    return {
        "model": str(model_cfg["model_sha256"]),
        "context_models": str(model_cfg["context_models_sha256"]),
        "meta": str(model_cfg["meta_sha256"]),
        "features": str(model_cfg["features_sha256"]),
        "encoders": str(model_cfg["encoders_sha256"]),
    }


def verify(paths: dict[str, Path], hashes: dict[str, str]) -> None:
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != hashes[name]:
            raise RuntimeError(f"Hash inválido em {name}: esperado {hashes[name]}, obtido {actual}.")


def freeze() -> Path:
    config = load_config()
    root = bundle_dir(config)
    manifest_models = []
    for model_cfg in config["models"]:
        market = str(model_cfg["market"])
        sources = source_paths(market)
        hashes = expected_hashes(model_cfg)
        verify(sources, hashes)
        targets = target_paths(config, market)
        for name, source in sources.items():
            targets[name].parent.mkdir(parents=True, exist_ok=True)
            if not targets[name].exists():
                shutil.copy2(source, targets[name])
        verify(targets, hashes)
        manifest_models.append(pickle.loads(targets["meta"].read_bytes()))

    summary = root / "models" / "resumo_modelos.pkl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(manifest_models, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, summary)
    manifest = {
        "cycle_id": config["cycle_id"],
        "policy_version": config["policy_version"],
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "real_bets_allowed": False,
        "models": config["models"],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def validate() -> Path:
    config = load_config()
    root = bundle_dir(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Bundle paper ainda não foi congelado.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cycle_id") != config.get("cycle_id"):
        raise RuntimeError("Bundle pertence a outro ciclo paper.")
    for model_cfg in config["models"]:
        verify(target_paths(config, str(model_cfg["market"])), expected_hashes(model_cfg))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "validate"])
    args = parser.parse_args()
    path = freeze() if args.action == "freeze" else validate()
    print(f"Bundle paper válido: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
