from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
import queue
import shutil
import json
from collections import deque
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ML_DIR = ROOT / "04_ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from utils.prediction_paths import historical_prediction_path, normal_prediction_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.rule import Rule
    from rich import box
    from rich.live import Live
    from rich.columns import Columns
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.markup import escape

    HAS_RICH = True
    console = Console(highlight=False)
except Exception:
    HAS_RICH = False
    console = None

# ─── Paleta visual ────────────────────────────────────────────────────────────
# Pitch black + field green + amber score + chalk white
# Inspiração: placar eletrônico de estádio + terminal profissional de trading
C_BG        = "on #0a0f0a"          # pitch preto
C_GREEN     = "#1a7a2e"             # green do campo
C_GREEN_LT  = "#2aad42"            # verde claro (destaque)
C_AMBER     = "#f5a623"             # âmbar de placar
C_AMBER_DIM = "#a36a10"            # âmbar apagado
C_WHITE     = "#e8f0e8"            # chalk white
C_DIM       = "#4a5c4a"            # texto apagado
C_RED       = "#d93636"            # alerta vermelho
C_CYAN      = "#00c8c8"            # azul-ciano de odds
C_PURPLE    = "#9a6dcc"            # roxo de ML/modelo
C_GREY      = "#8a9a8a"            # cinza
C_BLUE      = "#3b82f6"            # azul premium

APP_NAME = "Football Lab"
APP_VERSION = "v1.3"
APP_SUBTITLE = "Sports Analytics Platform"
STATUS_CACHE_TTL = 45
CACHE_DIR = ROOT / "cache"
STATUS_CACHE_FILE = CACHE_DIR / "status.json"
_STATUS_CACHE: dict[str, object] = {"ts": 0, "data": None}
_SPLASH_SHOWN = False


def env_base() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Força flush imediato nos subprocessos Python
    env["PYTHONUNBUFFERED"] = "1"
    return env


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def pausar():
    if HAS_RICH:
        console.input(f"\n[{C_DIM}]  ↵  pressione ENTER para continuar[/{C_DIM}]")
    else:
        input("\n  Pressione ENTER para continuar...")


def pedir(label: str, default: str = "") -> str:
    sufixo = f" [{C_DIM}](ENTER = {default})[/{C_DIM}]" if default else ""
    if HAS_RICH:
        return console.input(f"  [{C_AMBER}]›[/{C_AMBER}] [{C_WHITE}]{label}[/{C_WHITE}]{sufixo}: ").strip() or default
    return (input(f"  › {label}{(' [' + default + ']') if default else ''}: ").strip() or default)


def pedir_opcional(label: str) -> str:
    if HAS_RICH:
        return console.input(
            f"  [{C_AMBER}]›[/{C_AMBER}] [{C_WHITE}]{label}[/{C_WHITE}] [{C_DIM}](ENTER para todos)[/{C_DIM}]: "
        ).strip()
    return input(f"  › {label} (ENTER para todos): ").strip()



def terminal_width(default: int = 120) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _fmt_int(n: int | str | None) -> str:
    if n is None or n == "":
        return "—"
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def _fmt_bytes(num: int | float | None) -> str:
    if num is None:
        return "—"
    try:
        num = float(num)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024:
                return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
            num /= 1024
        return f"{num:.1f}PB"
    except Exception:
        return "—"


def _bar(percent: float | int | None, width: int = 10) -> str:
    try:
        pct = max(0, min(100, float(percent)))
    except Exception:
        pct = 0
    filled = round(width * pct / 100)
    return "█" * filled + "░" * (width - filled) + f" {pct:.0f}%"


def _arquivo_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime if path.exists() else 0.0
    except Exception:
        return 0.0


# ─── Cache de glob (evita re-varrer pastas grandes a cada refresh) ────────────
_GLOB_CACHE: dict[Path, tuple[float, int]] = {}


def _safe_count_glob(folder: Path, pattern: str = "*.csv", limit: int = 50_000) -> int:
    if not folder.exists():
        return 0
    try:
        mtime = folder.stat().st_mtime
    except Exception:
        mtime = 0
    cached = _GLOB_CACHE.get(folder)
    if cached and cached[0] == mtime:
        return cached[1]
    total = 0
    try:
        for _ in folder.glob(pattern):
            total += 1
            if total >= limit:
                break
    except Exception:
        return 0
    _GLOB_CACHE[folder] = (mtime, total)
    return total


def _count_csv_header_fast(path: Path, columns: list[str] | None = None, max_rows: int = 5000) -> dict:
    """Resumo leve de CSV: contagem total via bytes (rápida) + amostra parseada limitada a max_rows."""
    result = {"rows": None, "columns": [], "unique": {}}
    if not path.exists() or path.suffix.lower() != ".csv":
        return result
    try:
        import csv

        # Contagem total de linhas via leitura em blocos de bytes — muito mais rápido
        # do que percorrer o arquivo inteiro com csv.DictReader linha a linha.
        with path.open("rb") as f:
            total_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
        result["rows"] = max(0, total_lines - 1)

        # Amostra: só parseia até max_rows linhas para colunas/valores únicos
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with path.open("r", encoding=enc, newline="", errors="replace") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    result["columns"] = fieldnames
                    sets = {c: set() for c in (columns or []) if c in fieldnames}
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        for c in sets:
                            val = (row.get(c) or "").strip()
                            if val:
                                sets[c].add(val)
                    result["unique"] = {c: len(v) for c, v in sets.items()}
                break
            except UnicodeDecodeError:
                continue
    except Exception:
        pass
    return result


def _latest_existing(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=_arquivo_mtime)


def _idade_arquivo(path: Path) -> str:
    if not path.exists():
        return "ausente"
    try:
        age = max(0, time.time() - path.stat().st_mtime)
        if age < 60:
            return "agora"
        if age < 3600:
            return f"{int(age // 60)}min"
        if age < 86400:
            return f"{int(age // 3600)}h"
        return f"{int(age // 86400)}d"
    except Exception:
        return "—"


def _contar_linhas_csv(path: Path, max_bytes: int = 80_000_000) -> str:
    """Conta linhas sem carregar CSV inteiro em memória. Evita travamentos no painel."""
    if not path.exists() or path.suffix.lower() != ".csv":
        return ""
    try:
        if path.stat().st_size > max_bytes:
            return "grande"
        with path.open("rb") as f:
            total = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
        return _fmt_int(max(0, total - 1))
    except Exception:
        return ""


def _status_snapshot(force: bool = False) -> dict:
    now = time.time()
    cached = _STATUS_CACHE.get("data")
    if cached and not force and now - float(_STATUS_CACHE.get("ts", 0)) < STATUS_CACHE_TTL:
        return cached  # type: ignore[return-value]

    if not force:
        file_cached = _load_status_cache(max_age=90)
        if file_cached:
            _STATUS_CACHE["ts"] = now
            _STATUS_CACHE["data"] = file_cached
            return file_cached

    today = date.today().isoformat()
    paths = {
        "catalogo": ROOT / "data" / "catalog" / "ligas_catalog.csv",
        "base_unificada": ROOT / "data" / "base_unificada.csv",
        "base_oficial": ROOT / "data" / "base_oficial.csv",
        "base_times": ROOT / "data" / "base_times_padronizados.csv",
        "backtest": ROOT / "03_backtest" / "results" / "resumo.csv",
        "datasets": ROOT / "04_ml" / "datasets" / "mercados_meta.pkl",
        "modelos": ROOT / "04_ml" / "models" / "resumo_modelos.pkl",
        "previsoes": normal_prediction_path(today),
        "historico": ROOT / "04_ml" / "banca" / "historico_apostas.csv",
        "feature_health": ROOT / "04_ml" / "reports" / "feature_health_report.html",
        "mlops": ROOT / "04_ml" / "reports" / "mlops_dashboard.html",
    }

    raw_dir = ROOT / "data" / "raw" / "futpython" / "ligas"
    eventos_dir = ROOT / "data" / "eventos"
    reports_dir = ROOT / "04_ml" / "reports"
    logs_dir = ROOT / "logs"

    existing = {k: v.exists() for k, v in paths.items()}
    core_keys = ["base_oficial", "backtest", "datasets", "modelos"]
    core_ok = sum(1 for k in core_keys if existing.get(k))
    total_ok = sum(1 for ok in existing.values() if ok)

    pred_info = _count_csv_header_fast(paths["previsoes"], ["mercado", "liga"], max_rows=4000)
    backtest_info = _count_csv_header_fast(paths["backtest"], ["mercado", "liga"], max_rows=4000)
    latest_core = _latest_existing([paths[k] for k in ["base_oficial", "backtest", "modelos", "previsoes", "historico"]])

    # CPU/RAM são opcionais para não criar dependência. Se psutil não existir, o terminal segue normal.
    cpu = ram = None
    try:
        import psutil  # type: ignore
        cpu = psutil.cpu_percent(interval=0.0)
        ram = psutil.virtual_memory().percent
    except Exception:
        pass

    base_age = _idade_arquivo(paths["base_oficial"])
    pred_age = _idade_arquivo(paths["previsoes"])
    backtest_age = _idade_arquivo(paths["backtest"])
    models_age = _idade_arquivo(paths["modelos"])

    issues = []
    if not paths["base_oficial"].exists():
        issues.append("base oficial ausente")
    elif base_age not in ("agora",) and (base_age.endswith("d") or base_age in ["ausente"]):
        issues.append("dados antigos")
    if not paths["backtest"].exists():
        issues.append("backtest ausente")
    if not paths["modelos"].exists():
        issues.append("modelos ausentes")
    if not paths["previsoes"].exists():
        issues.append("previsões de hoje ausentes")

    if core_ok == len(core_keys) and not issues:
        health_level, health_label, health_style, health_icon = "ok", "SAUDÁVEL", C_GREEN_LT, "●"
    elif core_ok >= 2:
        health_level, health_label, health_style, health_icon = "warn", "ATENÇÃO", C_AMBER, "●"
    else:
        health_level, health_label, health_style, health_icon = "bad", "INCOMPLETO", C_RED, "●"

    snap = {
        "paths": paths,
        "existing": existing,
        "ok_count": total_ok,
        "total_count": len(paths),
        "core_ok": core_ok,
        "core_total": len(core_keys),
        "raw_count": _safe_count_glob(raw_dir),
        "eventos_count": _safe_count_glob(eventos_dir),
        "reports_count": _safe_count_glob(reports_dir, "*.html") + _safe_count_glob(reports_dir, "*.csv"),
        "logs_count": _safe_count_glob(logs_dir, "*.log"),
        "mode": (os.environ.get("ML_PREDICT_MODE") or os.environ.get("BT_ML_DATASET_MODE") or "research").upper(),
        "base_age": base_age,
        "pred_age": pred_age,
        "backtest_age": backtest_age,
        "models_age": models_age,
        "latest_age": _idade_arquivo(latest_core) if latest_core else "ausente",
        "latest_path": str(latest_core.relative_to(ROOT)) if latest_core else "—",
        "pred_rows": pred_info.get("rows"),
        "pred_markets": pred_info.get("unique", {}).get("mercado"),
        "pred_leagues": pred_info.get("unique", {}).get("liga"),
        "backtest_rows": backtest_info.get("rows"),
        "backtest_markets": backtest_info.get("unique", {}).get("mercado"),
        "backtest_leagues": backtest_info.get("unique", {}).get("liga"),
        "cpu": cpu,
        "ram": ram,
        "health_level": health_level,
        "health_label": health_label,
        "health_style": health_style,
        "health_icon": health_icon,
        "issues": issues[:3],
        "generated_at": date.today().strftime("%d/%m/%Y"),
    }
    score, label, style, message = _health_score_from_snapshot(snap)
    snap["health_score"] = score
    snap["health_label"] = label
    snap["health_style"] = style
    snap["health_message"] = message
    snap["suggestion"] = _suggestion_from_snapshot(snap)

    _STATUS_CACHE["ts"] = now
    _STATUS_CACHE["data"] = snap
    _save_status_cache(snap)
    return snap


def _status_label_age(label: str, age: str, ok_missing: bool = False) -> str:
    if age == "ausente" and not ok_missing:
        return f"[{C_RED}]ausente[/]"
    if age.endswith("d"):
        return f"[{C_AMBER}]{age}[/]"
    if age == "ausente":
        return f"[{C_DIM}]ausente[/]"
    return f"[{C_GREEN_LT}]{age}[/]"

def _as_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "—":
            return default
        return int(value)
    except Exception:
        return default


def _cacheable_snapshot(snap: dict) -> dict:
    """Remove objetos Path/estilos Rich para gravar cache leve em JSON."""
    keep = {
        "mode", "base_age", "pred_age", "backtest_age", "models_age", "latest_age",
        "latest_path", "pred_rows", "pred_markets", "pred_leagues", "backtest_rows",
        "backtest_markets", "backtest_leagues", "raw_count", "eventos_count",
        "reports_count", "logs_count", "health_level", "health_label", "health_style", "issues",
        "generated_at", "health_score", "health_message", "suggestion",
        "ok_count", "total_count", "core_ok", "core_total"
    }
    return {k: snap.get(k) for k in keep if k in snap}


def _load_status_cache(max_age: int = 90) -> dict | None:
    try:
        if not STATUS_CACHE_FILE.exists():
            return None
        age = time.time() - STATUS_CACHE_FILE.stat().st_mtime
        if age > max_age:
            return None
        data = json.loads(STATUS_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        data["from_cache"] = True
        return data
    except Exception:
        return None


def _save_status_cache(snap: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = _cacheable_snapshot(snap)
        payload["cache_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temp_path = STATUS_CACHE_FILE.with_name(f".{STATUS_CACHE_FILE.name}.{os.getpid()}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_path, STATUS_CACHE_FILE)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except Exception:
        pass


def _health_score_from_snapshot(snap: dict) -> tuple[int, str, str, str]:
    """Score simples e explicável para o cabeçalho."""
    score = 100
    issues = list(snap.get("issues") or [])
    existing = snap.get("existing") or {}
    if not existing.get("base_oficial", False):
        score -= 25
    if not existing.get("backtest", False):
        score -= 20
    if not existing.get("modelos", False):
        score -= 20
    if not existing.get("previsoes", False):
        score -= 10
    if str(snap.get("base_age", "")).endswith("d"):
        score -= 10
    if str(snap.get("backtest_age", "")).endswith("d"):
        score -= 8
    score = max(0, min(100, score))
    if score >= 90:
        return score, "SAUDÁVEL", C_GREEN_LT, "Sistema pronto para análise."
    if score >= 70:
        return score, "PESQUISA PENDENTE", C_AMBER, "Há pendências leves, mas o modo pesquisa pode seguir."
    if score >= 45:
        return score, "ATENÇÃO", C_AMBER, "Execute coleta/backtest/ML antes de confiar no painel."
    return score, "INCOMPLETO", C_RED, "Artefatos críticos ausentes. Rode o pipeline assistido."


def _suggestion_from_snapshot(snap: dict) -> str:
    issues = list(snap.get("issues") or [])
    if any("previsões" in i for i in issues):
        return "Sugestão: opção 6 › Prever jogos de hoje ou opção 2 › Modo assistido."
    if any("modelos" in i for i in issues):
        return "Sugestão: opção 5 › Rodar dataset + treino."
    if any("backtest" in i for i in issues):
        return "Sugestão: opção 4 › Rodar backtest."
    if any("base oficial" in i or "dados" in i for i in issues):
        return "Sugestão: opção 3 › Atualizar dados."
    return "Sugestão: acompanhe o painel ou rode --auto quando quiser atualizar tudo."


def _bar_status(percent: float | int | None, width: int = 8) -> str:
    try:
        pct = max(0, min(100, float(percent)))
    except Exception:
        return "—"
    filled = round(width * pct / 100)
    label = "🟢" if pct < 70 else "🟡" if pct < 88 else "🔴"
    return f"{label} {'█' * filled}{'░' * (width - filled)} {pct:.0f}%"


def _age_label_natural(age: str) -> str:
    if age == "agora":
        return "agora"
    if age == "ausente":
        return "ausente"
    if age.endswith("min"):
        return f"há {age}"
    if age.endswith("h"):
        return f"há {age}"
    if age.endswith("d"):
        return f"há {age}"
    return age


def imprimir_status_compacto(snap: dict | None = None):
    if not HAS_RICH:
        return
    snap = snap or _status_snapshot()
    issues = snap.get("issues") or []
    motivo = "; ".join(issues) if issues else str(snap.get("health_message", "operação normal"))
    score = _as_int(snap.get("health_score"), 0)
    style = snap.get("health_style") or (C_GREEN_LT if score >= 90 else C_AMBER if score >= 60 else C_RED)

    grid = Table.grid(expand=True)
    for _ in range(6):
        grid.add_column(ratio=1)
    grid.add_row(
        f"[{C_DIM}]Modo[/]\n[bold {C_AMBER}]{snap.get('mode', 'RESEARCH')}[/]",
        f"[{C_DIM}]Health[/]\n[bold {style}]{score}% · {snap.get('health_label', '—')}[/]",
        f"[{C_DIM}]Dados[/]\n[bold {C_WHITE}]{_age_label_natural(str(snap.get('base_age', '—')))}[/]",
        f"[{C_DIM}]Backtest[/]\n[bold {C_WHITE}]{_age_label_natural(str(snap.get('backtest_age', '—')))}[/]",
        f"[{C_DIM}]Modelos[/]\n[bold {C_WHITE}]{_age_label_natural(str(snap.get('models_age', '—')))}[/]",
        f"[{C_DIM}]Previsões[/]\n[bold {C_WHITE}]{_fmt_int(snap.get('pred_rows')) if snap.get('pred_rows') else _age_label_natural(str(snap.get('pred_age', 'ausente')))}[/]",
    )

    grid2 = Table.grid(expand=True)
    for _ in range(6):
        grid2.add_column(ratio=1)
    grid2.add_row(
        f"[{C_DIM}]CSVs/Eventos[/]\n[bold {C_GREEN_LT}]{snap.get('raw_count', 0)} / {snap.get('eventos_count', 0)}[/]",
        f"[{C_DIM}]Mercados[/]\n[bold {C_CYAN}]{_fmt_int(snap.get('backtest_markets'))} ativos[/]",
        f"[{C_DIM}]Relatórios[/]\n[bold {C_PURPLE}]{_fmt_int(snap.get('reports_count'))}[/]",
        f"[{C_DIM}]Último artefato[/]\n[bold {C_WHITE}]{_age_label_natural(str(snap.get('latest_age', '—')))}[/]",
        f"[{C_DIM}]CPU[/]\n[bold {C_WHITE}]{_bar_status(snap.get('cpu'), 7) if snap.get('cpu') is not None else '—'}[/]",
        f"[{C_DIM}]RAM[/]\n[bold {C_WHITE}]{_bar_status(snap.get('ram'), 7) if snap.get('ram') is not None else '—'}[/]",
    )

    stack = Table.grid(expand=True)
    stack.add_column()
    stack.add_row(grid)
    stack.add_row(grid2)
    stack.add_row(f"[{C_DIM}]Diagnóstico:[/] [{C_WHITE}]{escape(motivo)}[/]")
    stack.add_row(f"[{C_DIM}]{escape(str(snap.get('suggestion', '')))}[/]")
    if snap.get("from_cache"):
        stack.add_row(f"[{C_DIM}]status via cache rápido · atualize em Painel operacional para recálculo completo[/]")
    console.print(Panel(stack, border_style=style, box=box.ROUNDED, padding=(0, 1)))

def painel_premium_status(snap: dict | None = None):
    """Painel operacional mais rico, mas ainda leve para abrir rápido."""
    if not HAS_RICH:
        return
    snap = snap or _status_snapshot(force=True)
    checklist = Table.grid(padding=(0, 2))
    checklist.add_column(style=C_WHITE)
    checklist.add_column(justify="center")
    checklist.add_column(style=C_DIM)

    items = [
        ("Base oficial", "base_oficial", snap["base_age"]),
        ("Backtest", "backtest", snap["backtest_age"]),
        ("Datasets ML", "datasets", _idade_arquivo(snap["paths"]["datasets"])),
        ("Modelos", "modelos", snap["models_age"]),
        ("Previsões hoje", "previsoes", snap["pred_age"]),
        ("MLOps dashboard", "mlops", _idade_arquivo(snap["paths"]["mlops"])),
    ]
    for label, key, age in items:
        ok = snap["existing"].get(key, False)
        icon = f"[{C_GREEN_LT}]● OK[/]" if ok else f"[{C_RED}]○ AUSENTE[/]"
        checklist.add_row(label, icon, str(age))

    kpis = Table.grid(padding=(0, 4))
    kpis.add_column(style=C_DIM)
    kpis.add_column(style=f"bold {C_AMBER}")
    kpis.add_row("Linhas backtest", _fmt_int(snap.get("backtest_rows")))
    kpis.add_row("Mercados backtest", _fmt_int(snap.get("backtest_markets")))
    kpis.add_row("Ligas backtest", _fmt_int(snap.get("backtest_leagues")))
    kpis.add_row("Previsões hoje", _fmt_int(snap.get("pred_rows")))
    kpis.add_row("Mercados previsão", _fmt_int(snap.get("pred_markets")))
    kpis.add_row("Ligas previsão", _fmt_int(snap.get("pred_leagues")))

    layout = Table.grid(expand=True)
    layout.add_column(ratio=2)
    layout.add_column(ratio=1)
    layout.add_row(
        Panel(checklist, title=f"[{C_WHITE}]Health Check[/]", border_style=snap["health_style"], box=box.ROUNDED),
        Panel(kpis, title=f"[{C_WHITE}]KPIs rápidos[/]", border_style=C_CYAN, box=box.ROUNDED),
    )
    console.print(layout)
# ─── Cabeçalho ────────────────────────────────────────────────────────────────

LOGO = r"""
  ___          _   _           _ _   _          _
 | __|___  ___| |_| |__  __ _| | | | |   __ _ | |__
 | _|/ _ \/ _ \  _| '_ \/ _` | | | | |__/ _` || '_ \
 |_| \___/\___/\__|_.__/\__,_|_|_| |____\__,_||_.__/
"""

def _regua(char: str = "─", cor: str = C_GREEN) -> Rule:
    return Rule(characters=char, style=cor)


def print_header():
    if not HAS_RICH:
        print("=" * 72)
        print("  FOOTBALL LAB — Terminal Operacional")
        print("=" * 72)
        return

    limpar_tela()
    logo_text = Text(LOGO, style=f"bold {C_GREEN_LT}")
    badge = Text(f"  {APP_SUBTITLE.upper()}  ", style=f"bold {C_WHITE} on {C_GREEN}")
    versao = Text(f"  {APP_VERSION} · {date.today().strftime('%d/%m/%Y')}  ", style=f"italic {C_DIM}")

    console.print()
    console.print(Align.center(logo_text))
    console.print(Align.center(badge))
    console.print(Align.center(versao))
    console.print(_regua("═", C_GREEN))
    console.print()


# ─── Executor de subprocessos ─────────────────────────────────────────────────

_PATTERN_LIGA   = re.compile(r"\[(\d+)/(\d+)\]")
_PATTERN_ETAPA  = re.compile(r"Etapa atual:\s*(.+)")
_PATTERN_OK_STP = re.compile(r"\[OK\]\s(.+?)\s->")
_PATTERN_NOVAS  = re.compile(r"novas=(\d+)")
_PATTERN_ROI    = re.compile(r"ROI|AUC|Backtest|Treinando|Processando|mercados|modelo", re.I)

# Ruídos comuns de bibliotecas/rotinas longas. Eles continuam existindo,
# mas o terminal resume em vez de repetir dezenas de linhas iguais.
_NOISE_PATTERNS = [
    re.compile(r"DtypeWarning", re.I),
    re.compile(r"Columns \(\d+\) have mixed types", re.I),
    re.compile(r"Specify dtype option", re.I),
    re.compile(r"low_memory=False", re.I),
]
_HISTORY_WARN_RE = re.compile(r"Sem histórico anterior suficiente para '([^']+)'", re.I)

def _is_noise_line(raw: str) -> str | None:
    if any(p.search(raw) for p in _NOISE_PATTERNS):
        return "DtypeWarning / leitura CSV"
    m = _HISTORY_WARN_RE.search(raw)
    if m:
        return f"Sem histórico suficiente: {m.group(1)}"
    return None


def _classificar_linha(raw: str):
    """Retorna (estilo_rich, raw) para colorir a linha."""
    if "Traceback" in raw or "[ERRO]" in raw or "Erro" in raw or "Exception" in raw:
        return f"bold {C_RED}", raw
    if "concluído" in raw.lower() or "finalizado" in raw.lower() or "salvo" in raw.lower():
        return f"bold {C_GREEN_LT}", raw
    if "aviso" in raw.lower() or "warning" in raw.lower():
        return C_AMBER, raw
    if _PATTERN_ROI.search(raw):
        return C_CYAN, raw
    if _PATTERN_OK_STP.search(raw):
        return C_GREEN_LT, raw
    return C_DIM, raw


def _invalidate_status_async():
    """Invalida o cache em memória e recalcula em background, sem travar a volta ao menu.
    O próximo desenho de tela usa o cache em disco (rápido) enquanto o snapshot
    completo é recomputado em paralelo."""
    _STATUS_CACHE["ts"] = 0
    _STATUS_CACHE["data"] = None
    threading.Thread(target=lambda: _status_snapshot(force=True), daemon=True).start()


def executar(cmd: list[str], titulo: str) -> int:
    """Executor visual sem travar: usa queue, limita logs renderizados e evita loop agressivo."""
    started = time.perf_counter()

    if not HAS_RICH:
        print(f"\n{'─'*72}\n▶ {titulo}\n$ {' '.join(cmd)}\n{'─'*72}\n")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env_base(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()
        rc = proc.wait()
        elapsed = time.perf_counter() - started
        print(f"\n{'─'*72}")
        print(f"{'OK' if rc == 0 else 'ERRO'}: {titulo} | tempo: {elapsed:.1f}s")
        return rc

    console.print()
    console.print(Panel(
        f"[{C_AMBER}]▶[/] [{C_WHITE}]{escape(titulo)}[/]\n[{C_DIM}]$ {escape(' '.join(cmd))}[/]",
        border_style=C_GREEN,
        box=box.ROUNDED,
        padding=(0, 2),
    ))

    progress = Progress(
        SpinnerColumn(style=C_GREEN_LT),
        TextColumn(f"[bold {C_AMBER}]{{task.description}}[/bold {C_AMBER}]"),
        BarColumn(bar_width=28, style=C_GREEN, complete_style=C_GREEN_LT, finished_style=C_GREEN_LT),
        TextColumn(f"[{C_WHITE}]{{task.percentage:>5.1f}}%[/{C_WHITE}]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env_base(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None

    q: queue.Queue[str | None] = queue.Queue(maxsize=2000)

    def _reader():
        try:
            for raw in iter(proc.stdout.readline, ""):
                try:
                    q.put(raw.rstrip(), timeout=0.2)
                except queue.Full:
                    # Se o terminal estiver atrasado, não deixa o subprocesso travar.
                    pass
        finally:
            q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    total = None
    current = 0
    novos_total = 0
    task_id = None
    etapa_atual = titulo
    last_line = ""
    finished = False
    visible_log: deque[str] = deque(maxlen=10)
    suppressed = 0
    noise_counts: dict[str, int] = {}

    def _registrar_ruido(kind: str):
        noise_counts[kind] = noise_counts.get(kind, 0) + 1
        qtd = noise_counts[kind]
        # Mostra só o primeiro aviso e depois em marcos; evita poluir backfill/previsões.
        if qtd in (1, 10, 25, 50, 100) or qtd % 250 == 0:
            console.print(f"  [{C_DIM}]… aviso resumido: {escape(kind)} ({qtd}x)[/]")

    def _print_line(raw: str, force: bool = False):
        nonlocal last_line, suppressed
        if not raw or raw == last_line:
            return
        estilo, texto = _classificar_linha(raw)
        # Reduz ruído de linhas genéricas em loops longos, mas preserva erros/avisos/métricas.
        important = force or estilo != C_DIM or any(k in raw.lower() for k in ["erro", "warning", "aviso", "auc", "roi", "salvo", "finalizado"])
        if important:
            if suppressed:
                console.print(f"  [{C_DIM}]… {suppressed} linhas técnicas ocultadas[/]")
                suppressed = 0
            console.print(f"  [{estilo}]{escape(texto)}[/{estilo}]")
        else:
            visible_log.append(raw)
            suppressed += 1
            if suppressed % 60 == 0:
                console.print(f"  [{C_DIM}]… processando ({suppressed} linhas técnicas ocultadas)[/]")
        last_line = raw

    with progress:
        while not finished:
            try:
                raw = q.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    finished = True
                continue

            if raw is None:
                finished = True
                continue
            if not raw:
                continue

            noise_kind = _is_noise_line(raw)
            if noise_kind:
                _registrar_ruido(noise_kind)
                continue

            m = _PATTERN_LIGA.search(raw)
            if m:
                current = int(m.group(1))
                total = int(m.group(2))
                if task_id is None:
                    task_id = progress.add_task(etapa_atual, total=total)
                progress.update(task_id, completed=current)
                if current == 1 or current % 50 == 0 or current == total:
                    _print_line(raw, force=True)
                continue

            etapa = _PATTERN_ETAPA.search(raw)
            if etapa:
                etapa_atual = etapa.group(1).strip()
                if task_id is not None:
                    progress.update(task_id, description=etapa_atual)
                console.print(f"  [{C_AMBER}]◆[/] [{C_WHITE}]{escape(etapa_atual)}[/]")
                continue

            novas = _PATTERN_NOVAS.search(raw)
            if novas:
                qtd = int(novas.group(1))
                novos_total += qtd
                if qtd > 0:
                    console.print(f"  [{C_GREEN_LT}]+{qtd} registros novos[/]  [{C_DIM}]{escape(raw)}[/]")
                continue

            if "OK incremental" in raw and "novas=0" in raw:
                continue

            _print_line(raw)

    rc = proc.wait()
    elapsed = time.perf_counter() - started

    console.print()
    if novos_total > 0:
        console.print(f"  [{C_GREEN_LT}]● Novos registros adicionados: [bold]{novos_total}[/bold][/{C_GREEN_LT}]")

    if noise_counts:
        resumo_ruidos = " · ".join(f"{escape(k)}: {v}x" for k, v in sorted(noise_counts.items()))
        console.print(f"  [{C_DIM}]● Avisos técnicos resumidos: {resumo_ruidos}[/]")

    status_style = C_GREEN_LT if rc == 0 else C_RED
    status_icon = "✔" if rc == 0 else "✘"
    status_txt = "Concluído" if rc == 0 else f"Erro código {rc}"
    console.print(Panel(
        f"[bold {status_style}]{status_icon} {status_txt}[/]\n"
        f"[{C_WHITE}]{escape(titulo)}[/]\n"
        f"[{C_DIM}]Tempo de execução: {elapsed:.1f}s[/]",
        border_style=status_style,
        box=box.ROUNDED,
        padding=(0, 2),
    ))
    # Após qualquer etapa, invalida o cache e recalcula em background (não bloqueia a volta ao menu).
    _invalidate_status_async()
    return rc

# ─── Status / Painel ──────────────────────────────────────────────────────────

def status_projeto():
    arquivos = {
        "Catálogo FutPython":  ROOT / "data" / "catalog" / "ligas_catalog.csv",
        "Base unificada":      ROOT / "data" / "base_unificada.csv",
        "Base oficial":        ROOT / "data" / "base_oficial.csv",
        "Base de times":       ROOT / "data" / "base_times_padronizados.csv",
        "Backtest resumo":     ROOT / "03_backtest" / "results" / "resumo.csv",
        "ML datasets":         ROOT / "04_ml" / "datasets" / "mercados_meta.pkl",
        "ML modelos":          ROOT / "04_ml" / "models" / "resumo_modelos.pkl",
    }

    # Único snapshot completo, reaproveitado pelos dois painéis abaixo.
    snap = _status_snapshot(force=True)

    if HAS_RICH:
        imprimir_status_compacto(snap)
        painel_premium_status(snap)
        console.print()
        table = Table(
            title=f"[bold {C_AMBER}]⬡  PAINEL OPERACIONAL[/bold {C_AMBER}]",
            box=box.SIMPLE_HEAD,
            header_style=f"bold {C_WHITE}",
            border_style=C_GREEN,
            show_edge=True,
            padding=(0, 2),
        )
        table.add_column("Artefato", style=C_WHITE, no_wrap=True)
        table.add_column("Estado",   justify="center", no_wrap=True)
        table.add_column("Linhas",   justify="right",  style=C_AMBER, no_wrap=True)
        table.add_column("Atualizado", justify="right", style=C_DIM, no_wrap=True)

        for nome, path in arquivos.items():
            existe = path.exists()
            detalhe = _contar_linhas_csv(path) if existe else ""
            status_txt = f"[{C_GREEN_LT}]●  OK[/]" if existe else f"[{C_RED}]○  ausente[/]"
            table.add_row(nome, status_txt, detalhe, _idade_arquivo(path))

        console.print(table)
        console.print()
    else:
        print("\n  STATUS DO PROJETO")
        for nome, path in arquivos.items():
            print(f"  {'OK' if path.exists() else '??'} — {nome} — {_idade_arquivo(path)}")

    raw_dir     = ROOT / "data" / "raw" / "futpython" / "ligas"
    eventos_dir = ROOT / "data" / "eventos"

    raw_count     = snap.get("raw_count", 0)
    eventos_count = snap.get("eventos_count", 0)

    if HAS_RICH:
        grid = Table.grid(padding=(0, 4))
        grid.add_column(style=f"bold {C_DIM}")
        grid.add_column(style=f"bold {C_AMBER}")
        grid.add_row("CSVs FutPython", str(raw_count))
        grid.add_row("Eventos analíticos", str(eventos_count))
        console.print(Panel(grid, title=f"[{C_WHITE}]Resumo de arquivos[/]", border_style=C_GREEN, padding=(0, 2)))
        console.print()
    else:
        print(f"  CSVs FutPython: {raw_count}")
        print(f"  Eventos analíticos: {eventos_count}")

    imprimir_resumo_previsoes()

# ─── Previsões ────────────────────────────────────────────────────────────────

def _ler_csv_seguro(path: Path):
    try:
        import pandas as pd
    except Exception:
        return None, "pandas indisponível"
    if not path.exists():
        return None, "arquivo inexistente"
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False), ""
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False), ""
        except Exception as exc:
            return None, str(exc)
    except Exception as exc:
        return None, str(exc)


def _contar_true(df, coluna: str) -> int:
    if df is None or coluna not in df.columns:
        return 0
    serie = df[coluna]
    if getattr(serie, "dtype", None) == bool:
        return int(serie.sum())
    return int(serie.astype(str).str.strip().str.lower().isin(["true", "1", "sim", "s", "yes"]).sum())


def _resumo_previsoes_do_dia():
    hoje = date.today().isoformat()
    previsoes_path = normal_prediction_path(hoje)
    historico_path = ROOT / "04_ml" / "banca" / "historico_apostas.csv"

    linhas = []
    df_prev, _ = _ler_csv_seguro(previsoes_path)
    if df_prev is not None:
        total    = len(df_prev)
        apostas  = _contar_true(df_prev, "apostar")
        mercados = df_prev["mercado"].dropna().nunique() if "mercado" in df_prev.columns else 0
        ligas    = df_prev["liga"].dropna().nunique()    if "liga"    in df_prev.columns else 0
        linhas  += [
            ("Arquivo",              str(previsoes_path.relative_to(ROOT))),
            ("Total de previsões",   str(total)),
            ("Com apostar = True",   str(apostas)),
            ("Mercados distintos",   str(mercados)),
            ("Ligas distintas",      str(ligas)),
        ]
        return "Previsões de hoje", linhas

    df_hist, _ = _ler_csv_seguro(historico_path)
    if df_hist is not None:
        linhas.append(("Arquivo", str(historico_path.relative_to(ROOT))))
        linhas.append(("Total de apostas", str(len(df_hist))))
        if not df_hist.empty:
            cols = [c for c in ["data", "liga", "home", "away", "mercado", "apostar", "prob", "odd", "ev"] if c in df_hist.columns]
            for idx, row in df_hist.tail(5)[cols].fillna("").iterrows():
                resumo = "  ".join(f"{c}={row[c]}" for c in cols)
                linhas.append((f"Reg. {idx + 1}", resumo))
        return "Histórico de apostas", linhas

    return "Previsões", [("Status", "Nenhuma previsão registrada ainda.")]


def imprimir_resumo_previsoes():
    titulo, linhas = _resumo_previsoes_do_dia()
    if HAS_RICH:
        table = Table(
            title=f"[bold {C_CYAN}]🔮  {titulo}[/bold {C_CYAN}]",
            box=box.SIMPLE_HEAD,
            header_style=f"bold {C_WHITE}",
            border_style=C_CYAN,
        )
        table.add_column("Item",  style=C_WHITE)
        table.add_column("Valor", style=C_AMBER)
        for item, valor in linhas:
            table.add_row(item, valor)
        console.print(table)
    else:
        print(f"\n  {titulo}")
        for item, valor in linhas:
            print(f"  {item}: {valor}")


# ─── ML env ───────────────────────────────────────────────────────────────────

def configurar_ml_env(automatico: bool = False):
    if automatico:
        os.environ["ML_MIN_APOSTAS"]  = "1000"
        os.environ["ML_MIN_ROI"]      = "0.03"
        os.environ["ML_TRAIN_MIN_ROI"] = "0.03"
        os.environ.pop("ML_ONLY_MARKET", None)
        os.environ.pop("ML_MAX_ROWS", None)
        return

    if HAS_RICH:
        console.print(Panel(
            f"[{C_WHITE}]Defina os filtros de treinamento do modelo ML.[/{C_WHITE}]",
            title=f"[bold {C_PURPLE}]⚙  Filtros do ML[/bold {C_PURPLE}]",
            border_style=C_PURPLE,
        ))
    else:
        print("\n  Filtros do ML")

    min_apostas   = pedir("Mínimo de apostas", "1000")
    min_roi       = pedir("ROI mínimo",        "0.03").replace(",", ".")
    only_mercado  = pedir("Mercado específico", "")
    max_rows      = pedir("Máximo de linhas por mercado", "")

    os.environ["ML_MIN_APOSTAS"]   = min_apostas
    os.environ["ML_MIN_ROI"]       = min_roi
    os.environ["ML_TRAIN_MIN_ROI"] = min_roi

    if only_mercado:
        os.environ["ML_ONLY_MARKET"] = only_mercado
    else:
        os.environ.pop("ML_ONLY_MARKET", None)

    if max_rows:
        os.environ["ML_MAX_ROWS"] = max_rows
    else:
        os.environ.pop("ML_MAX_ROWS", None)


# ─── Montagem de pipeline ─────────────────────────────────────────────────────

def montar_pipeline(
    fetch_all=False,
    run_backtest=False,
    run_ml=False,
    incremental_default=False,
    ml_auto=False,
):
    cmd = [PYTHON, "01_scripts/run_pipeline.py"]
    cmd.append("--fetch-all" if fetch_all else "--fetch")

    country     = pedir_opcional("País")
    season      = pedir_opcional("Temporada")
    sleep       = pedir("Pausa entre downloads (segundos)", "0.5")
    force       = pedir("Forçar re-download de arquivos existentes? (s/N)", "n").lower() == "s"
    incremental = False
    if not force:
        default_inc = "s" if incremental_default else "n"
        incremental = pedir("Atualizar incrementalmente? (s/N)", default_inc).lower() == "s"

    if country:     cmd += ["--country",  country]
    if season:      cmd += ["--season",   season]
    if sleep:       cmd += ["--sleep",    sleep]
    if force:       cmd.append("--force")
    if incremental: cmd.append("--incremental")
    if run_backtest: cmd.append("--run-backtest")
    if run_ml:
        configurar_ml_env(automatico=ml_auto)
        cmd.append("--run-ml")

    return cmd


# ─── Funções de menu ──────────────────────────────────────────────────────────

def menu_teste():
    country = pedir("País para teste", "Brazil")
    season  = pedir("Temporada", "2025")
    limit   = pedir("Limite de ligas", "5")
    cmd = [
        PYTHON, "01_scripts/run_pipeline.py", "--fetch",
        "--country", country, "--season", season, "--limit", limit,
    ]
    executar(cmd, "Pipeline de teste")


def menu_dry_run():
    cmd = [PYTHON, "01_scripts/run_pipeline.py", "--fetch-all", "--dry-run"]
    country = pedir_opcional("País")
    season  = pedir_opcional("Temporada")
    if country: cmd += ["--country", country]
    if season:  cmd += ["--season",  season]
    executar(cmd, "Simular download do catálogo")


def menu_jogos_do_dia():
    if HAS_RICH:
        dias_txt = console.input(
            f"  [{C_AMBER}]›[/{C_AMBER}] [{C_WHITE}]Buscar jogos de quantos dias?[/{C_WHITE}] "
            f"[{C_DIM}](0=hoje  1=hoje+amanhã  2=3 dias | ENTER=0)[/{C_DIM}]: "
        ).strip() or "0"
    else:
        dias_txt = input("  › Buscar jogos de quantos dias? (0/1/2 | ENTER=0): ").strip() or "0"

    try:
        dias = max(0, int(dias_txt))
    except ValueError:
        dias = 0

    hoje = date.today()
    for i in range(dias + 1):
        data = hoje + timedelta(days=i)
        executar(
            [PYTHON, "01_scripts/01_fetch_futpython_daily.py", "--date", str(data)],
            f"Atualizar jogos — {data.strftime('%d/%m/%Y')}",
        )


def menu_prever_jogos_hoje():
    data_alvo   = date.today().isoformat()
    output_path = normal_prediction_path(data_alvo)

    if HAS_RICH:
        console.print(Panel(
            f"[{C_WHITE}]Data alvo:[/{C_WHITE}]  [{C_AMBER}]{date.today().strftime('%d/%m/%Y')}[/{C_AMBER}]",
            title=f"[bold {C_CYAN}]🔮  Prever jogos de hoje[/bold {C_CYAN}]",
            border_style=C_CYAN,
        ))
    else:
        print(f"\n  Prever jogos de hoje — {data_alvo}")

    rc_fetch = executar(
        [PYTHON, "01_scripts/01_fetch_futpython_daily.py", "--date", data_alvo],
        f"Atualizar jogos via FutPython ({data_alvo})",
    )
    if rc_fetch != 0:
        if HAS_RICH:
            console.print(f"  [{C_RED}]✘  Coleta FutPython falhou. Abortando fluxo.[/{C_RED}]")
        else:
            print("  Coleta FutPython falhou. Abortando.")
        return

    rc_predict = executar(
        [PYTHON, "04_ml/03_predict.py", "--date", data_alvo],
        f"Gerar previsões ML ({data_alvo})",
    )
    if rc_predict != 0:
        if HAS_RICH:
            console.print(f"  [{C_RED}]✘  Previsão ML falhou.[/{C_RED}]")
        else:
            print("  Previsão ML falhou.")
        return

    rel_path = output_path.relative_to(ROOT)
    if HAS_RICH:
        if output_path.exists():
            console.print(f"\n  [{C_GREEN_LT}]✔  Arquivo gerado:[/{C_GREEN_LT}]  [{C_AMBER}]{rel_path}[/{C_AMBER}]")
        else:
            console.print(f"\n  [{C_AMBER}]⚠  Previsão concluída mas arquivo não encontrado: {rel_path}[/{C_AMBER}]")
    else:
        print(f"\n  {'OK' if output_path.exists() else '⚠ Ausente'}: {rel_path}")

    console.print()
    imprimir_resumo_previsoes()


def menu_monitor_drift():
    executar([PYTHON, "04_ml/04_monitor_drift.py"], "Monitorar drift / calibração")


def menu_recuperar_calibracao():
    executar([PYTHON, "04_ml/20_calibration_recovery.py"], "Recuperar calibração dos modelos")
    if HAS_RICH:
        console.print(
            f"  [{C_DIM}]Próxima sequência sugerida:[/{C_DIM}]\n"
            f"  [{C_AMBER}]19[/{C_AMBER}] — Sincronizar guards e relatórios\n"
            f"  [{C_AMBER}]16[/{C_AMBER}] — Monitorar drift / calibração\n"
            f"  [{C_AMBER}]20[/{C_AMBER}] — Auditoria lacuna operacional"
        )
    else:
        print("  Sugestão: 19 → 16 → 20")


def menu_refresh_guards():
    executar([PYTHON, "04_ml/controles/operacao/atualizar_controles.py"], "Sincronizar guards e relatórios")


def menu_atualizar_perfil_mercados():
    executar([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Atualizar perfil operacional de mercados")


# ─── Relatórios analíticos (dashboards somente leitura) ───────────────────────

RELATORIOS_ETAPAS = [
    ("04_ml/07_banca_dashboard.py",              "Painel HTML da simulação"),
    ("04_ml/08_backtest_segmentado.py",          "Backtest segmentado por mercado/liga"),
    ("04_ml/10_operational_audit.py",            "Auditoria operacional (qualidade + performance)"),
    ("04_ml/11_observability_report.py",         "Observabilidade histórica (evolução, drawdown, ROI)"),
    ("04_ml/14_model_registry.py",               "Registro de modelos"),
    ("04_ml/15_model_baseline_compare.py",       "Comparação modelo atual vs baseline"),
    ("04_ml/16_feature_health_report.py",        "Saúde das features/datasets"),
    ("04_ml/17_mlops_dashboard.py",               "Painel HTML MLOps"),
    ("04_ml/18_operational_health_dashboard.py", "Painel executivo de saúde operacional"),
]


def menu_relatorio(script: str, titulo: str):
    executar([PYTHON, script], titulo)


def menu_rodar_todos_relatorios():
    etapas = [([PYTHON, script], titulo) for script, titulo in RELATORIOS_ETAPAS]
    for cmd_etapa, titulo in etapas:
        executar(cmd_etapa, titulo)
    if HAS_RICH:
        console.print(f"\n  [{C_GREEN_LT}]✔  Todos os relatórios foram atualizados em 04_ml/reports/[/{C_GREEN_LT}]")
    else:
        print("\n  Todos os relatórios foram atualizados em 04_ml/reports/")


def menu_reality_audit():
    executar([PYTHON, "04_ml/controles/mercados/auditoria.py"], "Auditoria lacuna operacional por mercado")


def menu_reality_governance():
    executar([PYTHON, "04_ml/controles/mercados/ciclo_vida.py"], "Governança operacional — simulação")


def menu_mercado_promotion():
    executar([PYTHON, "04_ml/controles/mercados/promocao.py"], "Avaliação de promoção de mercados — simulação")


def menu_liga_governance():
    executar([PYTHON, "04_ml/controles/ligas/auditoria.py"], "Auditoria e governança por liga — simulação")


def menu_governanca_confianca():
    executar([PYTHON, "04_ml/controles/confianca/auditoria.py"], "Governança de confiança — simulação")


def menu_governanca_odds():
    executar([PYTHON, "04_ml/controles/odds/auditoria.py"], "Governança de odds — simulação")


def menu_matriz_confianca_odds():
    executar([PYTHON, "04_ml/controles/odds/matriz_confianca_odds.py"], "Matriz confiança × odds — simulação")


def menu_limpeza():
    if HAS_RICH:
        console.print(Panel(
            f"[{C_WHITE}]Apaga bases geradas, eventos e resultados.\n"
            f"Preserva base.xlsx, .env e scripts.[/{C_WHITE}]",
            title=f"[bold {C_RED}]⚠  Limpeza de dados[/bold {C_RED}]",
            border_style=C_RED,
        ))
    else:
        print("\n  Limpeza — apaga bases geradas. Preserva scripts e .env.")

    confirmar = pedir("Para confirmar, digite LIMPAR", "")
    if confirmar != "LIMPAR":
        if HAS_RICH:
            console.print(f"  [{C_AMBER}]Limpeza cancelada.[/{C_AMBER}]")
        else:
            print("  Cancelada.")
        return

    caminhos = [
        ROOT / "data" / "base_unificada.csv",
        ROOT / "data" / "base_ligas.csv",
        ROOT / "data" / "base_times_padronizados.csv",
        ROOT / "data" / "base_oficial.csv",
        ROOT / "data" / "dicionario_ligas.csv",
        ROOT / "data" / "dicionario_times.csv",
    ]
    for path in caminhos:
        if path.exists():
            path.unlink()
            msg = f"apagado: {path.relative_to(ROOT)}"
            if HAS_RICH:
                console.print(f"  [{C_RED}]✘  {msg}[/{C_RED}]")
            else:
                print(f"  {msg}")

    for pasta in [ROOT / "data" / "eventos", ROOT / "03_backtest" / "results"]:
        if pasta.exists():
            for f in list(pasta.glob("*.csv")) + list(pasta.glob("*.txt")):
                f.unlink()
                msg = f"apagado: {f.relative_to(ROOT)}"
                if HAS_RICH:
                    console.print(f"  [{C_RED}]✘  {msg}[/{C_RED}]")
                else:
                    print(f"  {msg}")

    if HAS_RICH:
        console.print(f"\n  [{C_GREEN_LT}]✔  Limpeza concluída.[/{C_GREEN_LT}]")


def menu_backfill_previsoes():
    inicio  = pedir("Data início [YYYY-MM-DD]", "")
    fim     = pedir("Data fim    [YYYY-MM-DD]", "")
    audit   = pedir("Executar como auditoria histórica? (s/N)", "s").lower() == "s"
    retomar = pedir("Retomar e processar somente datas pendentes? (S/n)", "s").lower() != "n"
    comandos = montar_comandos_backfill(inicio, fim, audit=audit, retomar=retomar)
    if not comandos:
        print("\n  ✅ Todos os arquivos do período já estão válidos; nada a retomar.")
        return
    executar_sequencia([
        (cmd, f"Backfill de previsões ({ini} → {fim_bloco})")
        for cmd, ini, fim_bloco in comandos
    ])


def _arquivo_previsao_valido(data_ref: str, audit: bool) -> bool:
    path = historical_prediction_path(data_ref) if audit else normal_prediction_path(data_ref)
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            header_norm = {str(col).strip().lower() for col in header}
            if "mercado" not in header_norm or not ({"data", "date"} & header_norm):
                return False
            linhas = 0
            for row in reader:
                if len(row) != len(header):
                    return False
                linhas += 1
        return linhas > 0
    except (OSError, csv.Error):
        return False


def montar_comandos_backfill(
    inicio: str,
    fim: str,
    *,
    audit: bool,
    retomar: bool = True,
) -> list[tuple[list[str], str, str]]:
    """Monta blocos contíguos apenas para datas sem CSV válido."""
    data_inicio = date.fromisoformat(inicio)
    data_fim = date.fromisoformat(fim)
    if data_fim < data_inicio:
        raise ValueError("data final anterior à data inicial")

    pendentes: list[date] = []
    atual = data_inicio
    while atual <= data_fim:
        if not retomar or not _arquivo_previsao_valido(atual.isoformat(), audit):
            pendentes.append(atual)
        atual += timedelta(days=1)

    blocos: list[tuple[date, date]] = []
    for data_pendente in pendentes:
        if not blocos or data_pendente > blocos[-1][1] + timedelta(days=1):
            blocos.append((data_pendente, data_pendente))
        else:
            blocos[-1] = (blocos[-1][0], data_pendente)

    comandos = []
    for bloco_inicio, bloco_fim in blocos:
        cmd = [
            PYTHON,
            "01_scripts/backfill_daily_predictions.py",
            "--start",
            bloco_inicio.isoformat(),
            "--end",
            bloco_fim.isoformat(),
        ]
        if audit:
            cmd += ["--modo-auditoria", "--usar-flashscore-2025"]
        comandos.append((cmd, bloco_inicio.isoformat(), bloco_fim.isoformat()))
    return comandos


def executar_sequencia(etapas: list[tuple[list[str], str]], parar_em_erro: bool = True) -> bool:
    """Executa várias etapas do Football Lab usando o mesmo executor visual do terminal."""
    for cmd, titulo in etapas:
        rc = executar(cmd, titulo)
        if rc != 0 and parar_em_erro:
            if HAS_RICH:
                console.print(f"\n  [{C_RED}]✘ Sequência interrompida em:[/{C_RED}] [{C_WHITE}]{titulo}[/{C_WHITE}]")
            else:
                print(f"\n  Sequência interrompida em: {titulo}")
            return False
    return True


def menu_backfill_banca_completo():
    """Backfill histórico -> importar banca -> settlement -> scores -> relatório."""
    inicio = pedir("Data início [YYYY-MM-DD]", "2026-01-01")
    fim = pedir("Data fim    [YYYY-MM-DD]", date.today().isoformat())
    audit = pedir("Executar como auditoria histórica? (s/N)", "s").lower() == "s"
    retomar = pedir("Retomar e processar somente datas pendentes? (S/n)", "s").lower() != "n"
    comandos_backfill = montar_comandos_backfill(inicio, fim, audit=audit, retomar=retomar)
    origem_importacao = "historicas" if audit else "normais"

    etapas = [
        ([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Sincronizar perfil antes das previsões"),
        *[
            (cmd, f"Backfill de previsões ({ini} → {fim_bloco})")
            for cmd, ini, fim_bloco in comandos_backfill
        ],
        ([PYTHON, "04_ml/06_importar_backfill_banca.py", "--source", origem_importacao], "Importar backfill para banca"),
        ([PYTHON, "04_ml/05_settle_historico.py", "--skip-post-update"], "Liquidar histórico da banca"),
        ([PYTHON, "04_ml/04_banca.py", "--rebuild-bank"], "Reconstruir estado da banca"),
        ([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Atualizar perfil após settlement"),
        ([PYTHON, "04_ml/10_gerar_scores_contexto.py"], "Atualizar scores contextuais"),
        ([PYTHON, "04_ml/11_relatorio_contexto_operacional.py"], "Gerar relatório contextual operacional"),
    ]
    executar_sequencia(etapas)


def menu_validar_backfill_banca():
    inicio = pedir("Data início [YYYY-MM-DD]", "2026-01-01")
    fim = pedir("Data fim    [YYYY-MM-DD]", date.today().isoformat())
    audit = pedir("Executar backfill em auditoria histórica? (s/N)", "s").lower() == "s"
    retomar = pedir("Retomar e processar somente datas pendentes? (S/n)", "s").lower() != "n"
    comandos_backfill = montar_comandos_backfill(inicio, fim, audit=audit, retomar=retomar)
    origem_importacao = "historicas" if audit else "normais"

    etapas = [
        ([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Sincronizar perfil antes das previsões"),
        *[
            (cmd, f"Backfill de previsões ({ini} → {fim_bloco})")
            for cmd, ini, fim_bloco in comandos_backfill
        ],
        ([PYTHON, "04_ml/06_importar_backfill_banca.py", "--source", origem_importacao], "Importar backfill para banca"),
        ([PYTHON, "04_ml/05_settle_historico.py", "--skip-post-update"], "Liquidar histórico"),
        ([PYTHON, "04_ml/04_banca.py", "--rebuild-bank"], "Reconstruir estado da banca"),
        ([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Atualizar perfil após settlement"),
        ([PYTHON, "04_ml/10_gerar_scores_contexto.py"], "Atualizar scores contextuais"),
        ([PYTHON, "04_ml/11_relatorio_contexto_operacional.py"], "Gerar relatório contextual operacional"),
    ]

    for cmd_etapa, titulo in etapas:
        rc = executar(cmd_etapa, titulo)
        if rc != 0:
            if HAS_RICH:
                console.print(f"  [{C_RED}]✘ Pipeline interrompido em: {titulo}[/{C_RED}]")
            else:
                print(f"Pipeline interrompido em: {titulo}")
            return


def menu_pipeline_ml_completo():
    etapas = [
        ([PYTHON, "03_backtest/runner.py"], "Backtest completo"),
        ([PYTHON, "04_ml/01_dataset_builder.py"], "Dataset ML contextual"),
        ([PYTHON, "04_ml/02_train_model.py"], "Treinar ML contextual"),
        ([PYTHON, "04_ml/09_gerar_perfil_operacional_mercados.py"], "Atualizar perfil operacional de mercados"),
        ([PYTHON, "04_ml/03_predict.py", "--date", date.today().isoformat()], "Predict de hoje"),
    ]
    for cmd_etapa, titulo in etapas:
        rc = executar(cmd_etapa, titulo)
        if rc != 0:
            if HAS_RICH:
                console.print(f"  [{C_RED}]✘ Pipeline interrompido em: {titulo}[/{C_RED}]")
            else:
                print(f"Pipeline interrompido em: {titulo}")
            return


def menu_autopilot():
    data_alvo = date.today().isoformat()
    executar([PYTHON, "00_autopilot.py", "--mode", "research", "--date", data_alvo], "Piloto automático seguro — modo pesquisa")


def executar_autopilot_cli(mode: str = "research", data_alvo: str | None = None, force_retrain: bool = False, no_fetch: bool = False, skip_banca: bool = False) -> int:
    data_alvo = data_alvo or date.today().isoformat()
    cmd = [PYTHON, "00_autopilot.py", "--mode", mode, "--date", data_alvo]
    if force_retrain:
        cmd.append("--force-retrain")
    if no_fetch:
        cmd.append("--no-fetch")
    if skip_banca:
        cmd.append("--skip-banca")
    return executar(cmd, f"Piloto automático seguro — modo {mode}")


def executar_pipeline_research_cli(data_alvo: str | None = None) -> int:
    return executar_autopilot_cli(mode="research", data_alvo=data_alvo, force_retrain=True)


def executar_pipeline_production_cli(data_alvo: str | None = None) -> int:
    return executar_autopilot_cli(mode="production", data_alvo=data_alvo)


def executar_dashboard_cli() -> int:
    etapas = [
        ([PYTHON, "04_ml/07_banca_dashboard.py"], "Painel HTML da simulação"),
        ([PYTHON, "04_ml/17_mlops_dashboard.py"], "Painel MLOps"),
        ([PYTHON, "04_ml/18_operational_health_dashboard.py"], "Saúde operacional"),
    ]
    rc_final = 0
    for cmd_etapa, titulo in etapas:
        rc = executar(cmd_etapa, titulo)
        if rc != 0:
            rc_final = rc
    return rc_final


def menu_assistente_inteligente():
    limpar_tela()
    print_header()
    status_projeto()
    if HAS_RICH:
        console.print(f"\n[{C_AMBER}]Modo assistido[/{C_AMBER}] — escolha o nível de execução.")
    else:
        print("\nModo assistido — escolha o nível de execução.")

    opcoes = [
        ("1", "Somente analisar/prever jogos de hoje"),
        ("2", "Atualizar dados e prever"),
        ("3", "Rodar pipeline completo em research"),
        ("4", "Rodar piloto automático em production"),
        ("5", "Gerar dashboards"),
        ("0", "Voltar"),
    ]
    for cod, label in opcoes:
        print(f"  {cod}. {label}")
    escolha = pedir("Opção", "1")
    data_alvo = pedir("Data alvo", date.today().isoformat())

    if escolha == "0":
        return 0
    if escolha == "1":
        return executar([PYTHON, "04_ml/03_predict.py", "--date", data_alvo], "Predict assistido")
    if escolha == "2":
        return executar_autopilot_cli(mode="research", data_alvo=data_alvo, no_fetch=False, skip_banca=True)
    if escolha == "3":
        return executar_pipeline_research_cli(data_alvo=data_alvo)
    if escolha == "4":
        return executar_pipeline_production_cli(data_alvo=data_alvo)
    if escolha == "5":
        return executar_dashboard_cli()
    print("Opção inválida.")
    return 2


def executar_cli(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description="Football Lab — centro de comando manual, assistido e automático.")
    parser.add_argument("--auto", action="store_true", help="Roda o piloto automático seguro em modo research por padrão.")
    parser.add_argument("--assistant", action="store_true", help="Abre o modo assistido interativo.")
    parser.add_argument("--research", action="store_true", help="Roda pipeline completo em modo research.")
    parser.add_argument("--production", action="store_true", help="Roda pipeline completo em modo production.")
    parser.add_argument("--backtest", action="store_true", help="Roda apenas o backtest.")
    parser.add_argument("--train", action="store_true", help="Roda dataset builder + treino ML.")
    parser.add_argument("--predict", action="store_true", help="Roda apenas predição da data informada.")
    parser.add_argument("--dashboard", action="store_true", help="Gera dashboards principais.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Data alvo no formato YYYY-MM-DD.")
    parser.add_argument("--force-retrain", action="store_true", help="Força retreino no autopilot.")
    parser.add_argument("--no-fetch", action="store_true", help="Não baixa dados antes do autopilot.")
    parser.add_argument("--skip-banca", action="store_true", help="Não executa auto-fill da banca simulada.")

    args = parser.parse_args(argv)

    if args.assistant:
        return menu_assistente_inteligente()
    if args.auto:
        return executar_autopilot_cli("research", args.date, args.force_retrain, args.no_fetch, args.skip_banca)
    if args.research:
        return executar_pipeline_research_cli(args.date)
    if args.production:
        return executar_pipeline_production_cli(args.date)
    if args.backtest:
        return executar([PYTHON, "03_backtest/runner.py"], "Backtest")
    if args.train:
        rc = executar([PYTHON, "04_ml/01_dataset_builder.py"], "Dataset ML")
        if rc != 0:
            return rc
        return executar([PYTHON, "04_ml/02_train_model.py"], "Treino ML")
    if args.predict:
        return executar([PYTHON, "04_ml/03_predict.py", "--date", args.date], "Predict")
    if args.dashboard:
        return executar_dashboard_cli()

    return None


# ─── Definição do menu ────────────────────────────────────────────────────────

# (codigo, label_display, icone, acao_chave)
CATEGORIAS_MENU = [
    ("1", "Painel operacional",     "📊", "status_projeto"),
    ("2", "Piloto automático",      "🚀", "autopilot"),
    ("3", "Coleta de dados",        "⬇", "coleta"),
    ("4", "Backtest",               "📈", "backtest"),
    ("5", "Machine Learning",       "🤖", "ml"),
    ("6", "Previsões",              "🔮", "previsoes"),
    ("7", "Simulações",             "🧪", "banca"),
    ("8", "Auditorias",             "🔍", "auditorias"),
    ("9", "Controles operacionais", "⚙", "controles"),
    ("10", "Relatórios",            "📑", "relatorios"),
    ("11", "Manutenção",            "🔧", "manutencao"),
    ("0", "Sair",                   "✕", "sair"),
]

SUBMENUS = {
    "autopilot": [
        ("1", "Rodar piloto automático seguro", "🚀", lambda: menu_autopilot()),
        ("2", "Modo assistido", "🧭", lambda: menu_assistente_inteligente()),
        ("3", "Pipeline completo — research", "🧪", lambda: executar_pipeline_research_cli()),
        ("4", "Pipeline completo — production", "🛡", lambda: executar_pipeline_production_cli()),
    ],
    "coleta": [
        ("1", "Explorar / auditar catálogo",           "🗺",  lambda: (status_projeto(), menu_dry_run())),
        ("2", "Rodar pipeline de teste",               "🧪",  lambda: menu_teste()),
        ("3", "Atualizar base completa",               "⬇",  lambda: executar(montar_pipeline(fetch_all=True), "Atualizar base completa")),
        ("4", "Atualizar jogos do dia",                "📅",  lambda: menu_jogos_do_dia()),
        ("5", "Atualização incremental + processar",   "↻",  lambda: executar(montar_pipeline(fetch_all=True, incremental_default=True), "Atualização incremental")),
        ("6", "Simular download do catálogo",          "🔎",  lambda: menu_dry_run()),
    ],
    "backtest": [
        ("1", "Rodar apenas backtest",                 "📈",  lambda: executar([PYTHON, "03_backtest/runner.py"], "Backtest")),
        ("2", "Atualizar base completa + backtest",    "⬇",  lambda: executar(montar_pipeline(fetch_all=True, run_backtest=True), "Base + backtest")),
    ],
    "ml": [
        ("1", "Rodar dataset + treino",                "🤖",  lambda: (configurar_ml_env(automatico=False), executar([PYTHON, "04_ml/01_dataset_builder.py"], "Dataset ML"), executar([PYTHON, "04_ml/02_train_model.py"], "Treinar modelo"))),
        ("2", "Atualizar base + backtest + ML",        "⚡",  lambda: executar(montar_pipeline(fetch_all=True, run_backtest=True, run_ml=True, ml_auto=True), "Base + backtest + ML")),
        ("3", "Pipeline ML completo",                  "🧠",  lambda: menu_pipeline_ml_completo()),
        ("4", "Recuperar calibração dos modelos",      "🔄",  lambda: menu_recuperar_calibracao()),
        ("5", "Monitorar drift / calibração",          "📉",  lambda: menu_monitor_drift()),
    ],
    "previsoes": [
        ("1", "Prever jogos de hoje",                  "🔮",  lambda: menu_prever_jogos_hoje()),
        ("2", "Backfill de previsões por período",     "📅",  lambda: menu_backfill_previsoes()),
        ("3", "Backfill + banca + settlement",         "💼",  lambda: menu_backfill_banca_completo()),
    ],
    "banca": [
        ("1", "Abrir gestão simulada",              "🧪",  lambda: executar([PYTHON, "04_ml/04_banca.py"], "Gestão simulada")),
        ("2", "Validar simulação por backfill",       "📊",  lambda: menu_backfill_banca_completo()),
    ],
    "auditorias": [
        ("1", "Auditoria de mercados",                 "🔍",  lambda: menu_reality_audit()),
        ("2", "Auditoria de ligas",                    "🔍",  lambda: menu_liga_governance()),
        ("3", "Auditoria de confiança",                "🔍",  lambda: menu_governanca_confianca()),
        ("4", "Auditoria de odds",                     "🔍",  lambda: menu_governanca_odds()),
        ("5", "Matriz confiança × odds",               "⊠",  lambda: menu_matriz_confianca_odds()),
    ],
    "controles": [
        ("1", "Sincronizar guards e relatórios",       "⚙",  lambda: menu_refresh_guards()),
        ("2", "Controle de desempenho real",           "📊",  lambda: menu_reality_governance()),
        ("3", "Avaliação de promoção de mercados",     "⬆",  lambda: menu_mercado_promotion()),
        ("4", "Atualizar perfil operacional de mercados", "🗂",  lambda: menu_atualizar_perfil_mercados()),
    ],
    "manutencao": [
        ("1", "Limpar bases geradas",                  "🗑",  lambda: menu_limpeza()),
    ],
    "relatorios": [
        ("1", "Rodar todos os relatórios",             "⚡",  lambda: menu_rodar_todos_relatorios()),
        ("2", "Painel HTML da simulação",                  "💰",  lambda: menu_relatorio("04_ml/07_banca_dashboard.py", "Painel HTML da simulação")),
        ("3", "Backtest segmentado por mercado/liga",  "📊",  lambda: menu_relatorio("04_ml/08_backtest_segmentado.py", "Backtest segmentado por mercado/liga")),
        ("4", "Auditoria operacional (qualidade)",     "🔍",  lambda: menu_relatorio("04_ml/10_operational_audit.py", "Auditoria operacional")),
        ("5", "Observabilidade histórica (HTML)",      "📉",  lambda: menu_relatorio("04_ml/11_observability_report.py", "Observabilidade histórica")),
        ("6", "Registro de modelos",                   "🗃",  lambda: menu_relatorio("04_ml/14_model_registry.py", "Registro de modelos")),
        ("7", "Modelo atual vs baseline",               "⚖",  lambda: menu_relatorio("04_ml/15_model_baseline_compare.py", "Comparação modelo vs baseline")),
        ("8", "Saúde das features/datasets",           "🩺",  lambda: menu_relatorio("04_ml/16_feature_health_report.py", "Saúde das features/datasets")),
        ("9", "Painel HTML MLOps",                     "🧠",  lambda: menu_relatorio("04_ml/17_mlops_dashboard.py", "Painel MLOps")),
        ("10", "Painel executivo de saúde operacional", "🏥",  lambda: menu_relatorio("04_ml/18_operational_health_dashboard.py", "Painel executivo de saúde operacional")),
    ],
}


# ─── Renderização de menus ────────────────────────────────────────────────────

def _imprimir_opcoes_submenu(titulo: str, icone: str, itens):
    if not HAS_RICH:
        print(f"\n{'─'*72}\n  {titulo}\n{'─'*72}")
        for cod, label, *_ in itens:
            print(f"  {cod}  {label}")
        print("  0  Voltar")
        return

    console.print()
    console.print(Panel(
        f"[bold {C_AMBER}]{icone}  {titulo}[/bold {C_AMBER}]\n[{C_DIM}]Escolha uma ação. Use 0 para voltar ao centro de comando.[/]",
        border_style=C_GREEN,
        box=box.ROUNDED,
        padding=(0, 2),
    ))

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style=f"bold {C_AMBER}", no_wrap=True)
    table.add_column(style=C_DIM, no_wrap=True)
    table.add_column(style=C_WHITE)
    for cod, label, ic, *_ in itens:
        table.add_row(cod, ic, label)
    table.add_row("0", "←", f"[{C_DIM}]Voltar[/]")
    console.print(Panel(table, border_style=C_GREEN, box=box.ROUNDED, padding=(1, 2)))
    console.print()


def startup_splash():
    """Splash curto e leve. Não lê CSV grande, só dá sensação de inicialização premium."""
    global _SPLASH_SHOWN
    if _SPLASH_SHOWN or not HAS_RICH:
        return
    _SPLASH_SHOWN = True
    try:
        limpar_tela()
        steps = ["Config", "Cache", "Feature Store", "Model Health", "Reports"]
        table = Table.grid(padding=(0, 2))
        table.add_column(style=C_DIM)
        table.add_column(style=f"bold {C_GREEN_LT}")
        console.print(Panel(f"[bold {C_GREEN_LT}]Inicializando {APP_NAME}[/]\n[{C_DIM}]Command Center {APP_VERSION}[/]", border_style=C_GREEN, box=box.ROUNDED, padding=(1, 4)))
        for step in steps:
            console.print(f"  [{C_GREEN_LT}]✓[/] [{C_WHITE}]{step}[/]")
            time.sleep(0.045)
    except Exception:
        pass


def mostrar_menu_principal():
    if not HAS_RICH:
        print("=" * 72)
        print(f"  {APP_NAME} — {APP_SUBTITLE} {APP_VERSION}")
        print("=" * 72)
        for cod, label, ic, _ in CATEGORIAS_MENU:
            print(f"  {cod}  {ic}  {label}")
        print("=" * 72)
        return

    console.print()
    logo_text = Text(LOGO, style=f"bold {C_GREEN_LT}")
    console.print(Align.center(logo_text))

    badge = Text(f"  {APP_SUBTITLE.upper()}  ", style=f"bold {C_WHITE} on {C_GREEN}")
    data_txt  = Text(f"  {APP_VERSION} · {date.today().strftime('%A, %d de %B de %Y')}  ", style=f"italic {C_DIM}")
    tagline = Text("  Research Mode · Autopilot · Model Governance · Reports  ", style=C_DIM)
    console.print(Align.center(badge))
    console.print(Align.center(data_txt))
    console.print(Align.center(tagline))
    console.print()
    console.print(_regua("═", C_GREEN))
    console.print()

    imprimir_status_compacto()
    console.print()

    esquerda = CATEGORIAS_MENU[:6]
    direita  = CATEGORIAS_MENU[6:]

    def _menu_lines(items):
        return "\n".join(
            f"  [bold {C_AMBER}]{cod:>2}[/]  [{C_DIM}]{ic}[/]  [{C_WHITE}]{label}[/]"
            for cod, label, ic, _ in items
        )

    grid = Table.grid(padding=(0, 5), expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(_menu_lines(esquerda), _menu_lines(direita))

    footer = (
        f"[{C_DIM}]ENTER selecionar · 0 sair · refresh atualizar · Ctrl+C cancelar · "
        f"comandos: --auto  --assistant  --research  --production[/]"
    )
    console.print(Panel(
        grid,
        title=f"[bold {C_GREEN_LT}]Centro de Comando[/]",
        subtitle=footer,
        border_style=C_GREEN,
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()
# ─── Loop principal ───────────────────────────────────────────────────────────

def executar_submenu(nome: str, titulo: str, icone: str):
    itens = SUBMENUS.get(nome, [])
    while True:
        limpar_tela()
        _imprimir_opcoes_submenu(titulo, icone, itens)
        opcao = pedir("Opção", "")
        if opcao == "0":
            return
        match = next((x for x in itens if x[0] == opcao), None)
        if match:
            _cod, _label, _ic, acao = match
            acao()
            pausar()
        else:
            if HAS_RICH:
                console.print(f"  [{C_RED}]Opção inválida.[/{C_RED}]")
            else:
                print("  Opção inválida.")
            time.sleep(0.8)


def menu_principal():
    startup_splash()
    while True:
        limpar_tela()
        mostrar_menu_principal()
        opcao = pedir("Opção", "")
        if opcao.lower() in {"f5", "r", "refresh"}:
            _status_snapshot(force=True)
            continue
        if opcao.lower() in {"f9", "auto", "autopilot"}:
            menu_autopilot()
            pausar()
            continue
        item = next((x for x in CATEGORIAS_MENU if x[0] == opcao), None)
        if not item:
            if HAS_RICH:
                console.print(f"  [{C_RED}]Opção inválida.[/{C_RED}]")
            else:
                print("  Opção inválida.")
            time.sleep(0.8)
            continue

        cod, label, ic, acao = item

        if acao == "sair":
            if HAS_RICH:
                console.print(f"\n  [{C_GREEN_LT}]Até a próxima.[/{C_GREEN_LT}]\n")
            else:
                print("\n  Até a próxima.\n")
            break
        elif acao == "status_projeto":
            limpar_tela()
            print_header()
            status_projeto()
            pausar()
        else:
            executar_submenu(acao, label, ic)


if __name__ == "__main__":
    if not HAS_RICH:
        print("  Dica: pip install rich  — para a interface visual completa\n")
    rc = executar_cli()
    if rc is None:
        menu_principal()
    else:
        raise SystemExit(rc)
