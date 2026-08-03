#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode


HERE = Path(__file__).resolve()
WEB_DIR = HERE.parents[1]
ROOT = WEB_DIR.parents[0]

CRESTS_DIR = WEB_DIR / "static" / "crests"
LEAGUES_DIR = WEB_DIR / "static" / "leagues"
MANIFEST_PATH = WEB_DIR / "static" / "crest-manifest.json"
NOT_FOUND_PATH = WEB_DIR / "static" / "not_found.txt"
CACHE_PATH = WEB_DIR / "static" / "crest_fetch_cache.json"

DICIONARIO = ROOT / "data" / "dicionario_times.csv"
EXTRA_TEAMS = HERE.parent / "extra_teams.txt"
EXTRA_LEAGUES = HERE.parent / "extra_leagues.txt"
KEY_FILE = HERE.parent / "api_football_key.txt"

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/123"
APIF_BASE = "https://v3.football.api-sports.io"

NOT_FOUND = []
_TSDB_LEAGUE_CACHE = None


def slugify(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name or ""))
    n = n.encode("ascii", "ignore").decode("ascii").lower()
    n = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    return n or "item"


def read_lines(path: Path):
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]


def load_cache():
    if not CACHE_PATH.exists():
        return {"ok": {}, "not_found": {}}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        data.setdefault("ok", {})
        data.setdefault("not_found", {})
        return data
    except Exception:
        return {"ok": {}, "not_found": {}}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_api_football_key(cli_key):
    if cli_key:
        return cli_key.strip()
    if os.environ.get("API_FOOTBALL_KEY"):
        return os.environ["API_FOOTBALL_KEY"].strip()
    if KEY_FILE.exists():
        txt = KEY_FILE.read_text(encoding="utf-8", errors="replace").strip()
        if txt and "COLOQUE" not in txt.upper():
            return txt.splitlines()[0].strip()
    return None


def collect_from_dicionario():
    teams, leagues = [], []

    if DICIONARIO.exists() and DICIONARIO.stat().st_size > 0:
        try:
            with DICIONARIO.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pad = (row.get("Time_padronizado") or "").strip()
                    ori = (row.get("Time_original") or "").strip()
                    if pad or ori:
                        teams.append((pad, ori))

                    liga = (row.get("League_padronizada") or "").strip()
                    country = (row.get("country") or "").strip()
                    if liga or country:
                        leagues.append((liga, country))
        except Exception as e:
            print(f"aviso: não consegui ler {DICIONARIO}: {e}")

    for t in read_lines(EXTRA_TEAMS):
        teams.append((t, ""))

    for l in read_lines(EXTRA_LEAGUES):
        leagues.append((l, ""))

    def dedupe(items):
        out, seen = [], set()
        for a, b in items:
            key = (slugify(a), slugify(b))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
        return sorted(out, key=lambda x: x[0] or x[1])

    return dedupe(teams), dedupe(leagues)


def http_json(url, headers=None, timeout=8, retries=2):
    last = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "football-lab-crest-fetcher/2.0", **(headers or {})},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 429):
                raise
            time.sleep(0.5 * i)
        except Exception as e:
            last = e
            time.sleep(0.5 * i)
    raise last or RuntimeError("erro HTTP")


def download_image(url, dest, timeout=8, retries=2):
    last = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "football-lab-crest-fetcher/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data:
                return False
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except Exception as e:
            last = e
            time.sleep(0.5 * i)
    print(f"    falha imagem: {last}")
    return False


def tsdb_team(name, timeout, retries):
    try:
        data = http_json(f"{TSDB_BASE}/searchteams.php?t={quote(name)}", timeout=timeout, retries=retries)
        teams = (data or {}).get("teams") or []
        if teams:
            return teams[0].get("strBadge") or teams[0].get("strLogo")
    except Exception:
        return None
    return None


def apif_team(name, key, timeout, retries):
    try:
        url = f"{APIF_BASE}/teams?" + urlencode({"search": name})
        data = http_json(url, headers={"x-apisports-key": key}, timeout=timeout, retries=retries)
        resp = (data or {}).get("response") or []
        if resp:
            return resp[0].get("team", {}).get("logo")
    except Exception:
        return None
    return None


def tsdb_leagues(timeout, retries):
    global _TSDB_LEAGUE_CACHE
    if _TSDB_LEAGUE_CACHE is not None:
        return _TSDB_LEAGUE_CACHE
    try:
        data = http_json(f"{TSDB_BASE}/search_all_leagues.php?s=Soccer", timeout=timeout, retries=retries)
        _TSDB_LEAGUE_CACHE = (data or {}).get("countries") or []
    except Exception as e:
        print(f"aviso: ligas TheSportsDB indisponíveis ({e})")
        _TSDB_LEAGUE_CACHE = []
    return _TSDB_LEAGUE_CACHE


def tsdb_league(name, timeout, retries):
    target = name.strip().lower()
    if not target:
        return None
    for lg in tsdb_leagues(timeout, retries):
        lname = (lg.get("strLeague") or "").lower()
        if lname == target or target in lname or lname in target:
            return lg.get("strBadge")
    return None


def apif_league(name, key, timeout, retries):
    try:
        url = f"{APIF_BASE}/leagues?" + urlencode({"search": name})
        data = http_json(url, headers={"x-apisports-key": key}, timeout=timeout, retries=retries)
        resp = (data or {}).get("response") or []
        if resp:
            return resp[0].get("league", {}).get("logo")
    except Exception:
        return None
    return None


def progress(i, total, kind, name, msg):
    pct = (i / total * 100) if total else 0
    print(f"[{i:>5}/{total:<5}] {pct:6.2f}% | {kind:<5} | {name[:44]:<44} | {msg}")


def fetch_item(i, total, kind, a, b, key, cache, args):
    names = [x for x in [a, b] if x]
    display = names[0] if names else "-"

    folder = CRESTS_DIR if kind == "time" else LEAGUES_DIR
    cache_key = f"{kind}:{slugify(a)}:{slugify(b)}"

    for name in names:
        dest = folder / f"{slugify(name)}.png"
        if dest.exists():
            cache["ok"][cache_key] = str(dest.relative_to(WEB_DIR))
            progress(i, total, kind, name, f"já existe -> {dest.name}")
            return True

    if args.resume and cache_key in cache.get("not_found", {}):
        progress(i, total, kind, display, "skip cache not_found")
        return False

    for name in names:
        dest = folder / f"{slugify(name)}.png"

        if kind == "time":
            badge = tsdb_team(name, args.timeout, args.retries)
            source = "TheSportsDB"
            time.sleep(args.delay)
            if not badge and key:
                badge = apif_team(name, key, args.timeout, args.retries)
                source = "API-Football"
                time.sleep(args.delay)
        else:
            badge = tsdb_league(name, args.timeout, args.retries)
            source = "TheSportsDB"
            time.sleep(args.delay)
            if not badge and key:
                badge = apif_league(name, key, args.timeout, args.retries)
                source = "API-Football"
                time.sleep(args.delay)

        if badge and download_image(badge, dest, args.timeout, args.retries):
            cache["ok"][cache_key] = str(dest.relative_to(WEB_DIR))
            progress(i, total, kind, name, f"ok {source} -> {dest.name}")
            return True

    NOT_FOUND.append(f"{kind}: {a} ({b})")
    cache["not_found"][cache_key] = {"a": a, "b": b, "kind": kind}
    progress(i, total, kind, display, "não encontrado")
    return False


def write_manifest():
    teams = sorted(p.stem for p in CRESTS_DIR.glob("*.png"))
    leagues = sorted(p.stem for p in LEAGUES_DIR.glob("*.png"))
    MANIFEST_PATH.write_text(json.dumps({"teams": teams, "leagues": leagues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest atualizado: {len(teams)} times, {len(leagues)} ligas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-football-key", default=None)
    ap.add_argument("--delay", type=float, default=1.25)
    ap.add_argument("--timeout", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teams-only", action="store_true")
    ap.add_argument("--leagues-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    CRESTS_DIR.mkdir(parents=True, exist_ok=True)
    LEAGUES_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()

    cache = load_cache()
    key = get_api_football_key(args.api_football_key)
    teams, leagues = collect_from_dicionario()

    jobs = []
    if not args.leagues_only:
        jobs += [("time", a, b) for a, b in teams]
    if not args.teams_only:
        jobs += [("liga", a, b) for a, b in leagues]

    if args.limit > 0:
        jobs = jobs[:args.limit]

    total = len(jobs)
    if not total:
        print("Nenhum item encontrado.")
        return 1

    print("=" * 90)
    print("FOOTBALL LAB — CREST FETCHER SEGURO")
    print("=" * 90)
    print(f"Total a processar : {total}")
    print(f"Times catálogo    : {len(teams)}")
    print(f"Ligas catálogo    : {len(leagues)}")
    print(f"Delay             : {args.delay}s")
    print(f"Timeout           : {args.timeout}s")
    print(f"Retries           : {args.retries}")
    print(f"Resume            : {args.resume}")
    print(f"API-Football      : {'ativada' if key else 'desativada'}")
    print("=" * 90)

    ok = fail = 0
    try:
        for i, (kind, a, b) in enumerate(jobs, 1):
            if fetch_item(i, total, kind, a, b, key, cache, args):
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                save_cache(cache)
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando cache...")
    finally:
        save_cache(cache)
        write_manifest()
        if NOT_FOUND:
            NOT_FOUND_PATH.write_text("\n".join(NOT_FOUND), encoding="utf-8")

    print("=" * 90)
    print(f"OK              : {ok}")
    print(f"Não encontrados : {fail}")
    print(f"Cache           : {CACHE_PATH}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
