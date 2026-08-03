from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from futpython_client import FutPythonClient, save_text


CATALOG_PATH = Path(os.getenv("FUTPYTHON_CATALOG_PATH", "data/catalog/ligas_catalog.csv"))
RAW_DIR = Path(os.getenv("FUTPYTHON_RAW_DIR", "data/raw/futpython/leagues"))


def slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9áàâãéêíóôõúüçñ]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sem_nome"


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Catálogo não encontrado: {path}. Rode: python 01_scripts/00_build_ligas_catalog.py"
        )
    return pd.read_csv(path, dtype=str).fillna("")


def filter_catalog(df: pd.DataFrame, country: str | None, liga: str | None, season: str | None, limit: int | None) -> pd.DataFrame:
    out = df.copy()
    if country:
        out = out[out["country"].str.contains(country, case=False, na=False)]
    if liga:
        out = out[out["liga"].str.contains(liga, case=False, na=False)]
    if season:
        out = out[out["season"].astype(str) == str(season)]
    if limit:
        out = out.head(limit)
    return out.reset_index(drop=True)


def output_path(row: pd.Series, raw_dir: Path = RAW_DIR) -> Path:
    filename = f"{slug(row['country'])}__{slug(row['liga'])}__{slug(row['season'])}.csv"
    return raw_dir / filename



def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Lê CSV tentando encodings comuns da FutPython/Windows."""
    last_exc = None
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=enc).fillna("")
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError(f"Não foi possível ler {path}")


def _read_csv_text_flexible(content: str) -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(content), dtype=str, low_memory=False).fillna("")


def _find_key_column(df: pd.DataFrame) -> str | None:
    aliases = ["Game_ID", "Id_Jogo", "Match_ID", "Fixture_ID", "fixture_id", "game_id", "id", "ID"]
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for a in aliases:
        c = cols_lower.get(a.lower())
        if c is not None:
            return c
    return None


def _find_date_column(df: pd.DataFrame) -> str | None:
    aliases = ["Date", "Data", "date", "data"]
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for a in aliases:
        c = cols_lower.get(a.lower())
        if c is not None:
            return c
    return None


def _normalize_key_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def merge_incremental(path: Path, content: str) -> tuple[int, int, int, int]:
    """Atualiza CSV existente com dados remotos via upsert local."""
    remoto = _read_csv_text_flexible(content)
    local = _read_csv_flexible(path) if path.exists() else pd.DataFrame(columns=remoto.columns)

    all_cols = list(dict.fromkeys(list(local.columns) + list(remoto.columns)))
    local = local.reindex(columns=all_cols).fillna("")
    remoto = remoto.reindex(columns=all_cols).fillna("")

    key_col = _find_key_column(remoto) or _find_key_column(local)
    if key_col:
        local_keys = set(_normalize_key_series(local[key_col])) if key_col in local.columns else set()
        remote_keys = set(_normalize_key_series(remoto[key_col])) if key_col in remoto.columns else set()
        novas = len([k for k in remote_keys if k and k not in local_keys])
        combinado = pd.concat([local, remoto], ignore_index=True)
        combinado[key_col] = _normalize_key_series(combinado[key_col])
        combinado = combinado.drop_duplicates(subset=[key_col], keep="last")
    else:
        date_col = _find_date_column(remoto) or _find_date_column(local)
        possible = [c for c in [date_col, "Home", "Away", "League", "Season"] if c and c in all_cols]
        if len(possible) >= 3:
            local_key = local[possible].fillna("").astype(str).agg("__".join, axis=1)
            remote_key = remoto[possible].fillna("").astype(str).agg("__".join, axis=1)
            novas = len(set(remote_key) - set(local_key))
            combinado = pd.concat([local, remoto], ignore_index=True)
            combinado["__tmp_key__"] = combinado[possible].fillna("").astype(str).agg("__".join, axis=1)
            combinado = combinado.drop_duplicates(subset=["__tmp_key__"], keep="last").drop(columns=["__tmp_key__"])
        else:
            combinado = pd.concat([local, remoto], ignore_index=True).drop_duplicates(keep="last")
            novas = max(0, len(combinado) - len(local))

    path.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_csv(path, index=False, encoding="utf-8-sig")
    return len(local), len(remoto), len(combinado), novas


def fetch(args: argparse.Namespace) -> None:
    df = filter_catalog(load_catalog(), args.country, args.liga, args.season, args.limit)
    if df.empty:
        print("Nenhuma liga encontrada com os filtros informados.")
        return

    client = FutPythonClient(timeout=args.timeout)
    print(f"Ligas selecionadas: {len(df)}")

    def show(msg: str) -> None:
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)

    iterator = df.iterrows()
    pbar = tqdm(total=len(df), desc="FutPython", unit="liga", dynamic_ncols=True) if tqdm is not None else None

    for i, row in iterator:
        country, liga, season = row["country"], row["liga"], row["season"]
        url = client.build_download_url(country, liga, season)
        path = output_path(row)

        if args.dry_run:
            show(f"[{i+1}/{len(df)}] [DRY-RUN] {country} | {liga} | {season} -> {url}")
            if pbar is not None: pbar.update(1)
            continue

        if path.exists() and not args.force and not args.incremental:
            show(f"[{i+1}/{len(df)}] Já existe, pulando: {path.name}")
            if pbar is not None: pbar.update(1)
            continue

        acao = "Atualizando incremental" if path.exists() and args.incremental and not args.force else "Baixando"
        show(f"[{i+1}/{len(df)}] {acao}: {country} | {liga} | {season}")
        try:
            content = client.download_liga_csv(country, liga, season)
            if path.exists() and args.incremental and not args.force:
                antes, remoto, depois, novas = merge_incremental(path, content)
                show(f"    OK incremental -> {path.name} | local={antes} remoto={remoto} final={depois} novas={novas}")
            else:
                save_text(path, content)
                show(f"    OK -> {path}")

            # Salva metadados simples para rastrear origem sem alterar o CSV recebido.
            meta_path = path.with_suffix(".meta.txt")
            meta_path.write_text(
                f"country={country}\nliga={liga}\nseason={season}\nliga_id={row.get('liga_id','')}\nurl={url}\nmodo={'incremental' if args.incremental else 'full'}\n",
                encoding="utf-8",
            )
        except Exception as exc:
            show(f"    ERRO -> {country} | {liga} | {season}: {exc}")

        if pbar is not None:
            pbar.update(1)
        if args.sleep > 0:
            time.sleep(args.sleep)

    if pbar is not None:
        pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa bases históricas da FutPython por país/liga/temporada.")
    parser.add_argument("--country", help="Filtro por país. Ex: Brazil")
    parser.add_argument("--liga", help="Filtro por liga. Ex: Serie A")
    parser.add_argument("--season", help="Filtro por temporada. Ex: 2025")
    parser.add_argument("--limit", type=int, help="Limite de ligas para teste")
    parser.add_argument("--dry-run", action="store_true", help="Mostra URLs sem baixar nada")
    parser.add_argument("--force", action="store_true", help="Sobrescreve CSVs já baixados")
    parser.add_argument("--incremental", action="store_true", help="Atualiza CSVs existentes via upsert, preservando histórico local")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pausa entre requisições")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout HTTP em segundos")
    args = parser.parse_args()
    fetch(args)


if __name__ == "__main__":
    main()
