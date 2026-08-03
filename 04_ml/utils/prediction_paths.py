from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = ROOT_DIR / "04_ml"
PREDICTIONS_DIR = ML_DIR / "previsoes"
HISTORICAL_PREDICTIONS_DIR = ML_DIR / "previsoes_historicas"
LEGACY_PREDICTIONS_DIR = ML_DIR


def _date_str(data_ref: str | date | None = None) -> str:
    return str(data_ref or date.today())


def prediction_filename(data_ref: str | date | None = None) -> str:
    return f"previsoes_{_date_str(data_ref)}.csv"


def normal_prediction_path(data_ref: str | date | None = None, *, ensure_dir: bool = False) -> Path:
    if ensure_dir:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_DIR / prediction_filename(data_ref)


def historical_prediction_path(data_ref: str | date | None = None, *, ensure_dir: bool = False) -> Path:
    if ensure_dir:
        HISTORICAL_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORICAL_PREDICTIONS_DIR / prediction_filename(data_ref)


def legacy_prediction_path(data_ref: str | date | None = None) -> Path:
    return LEGACY_PREDICTIONS_DIR / prediction_filename(data_ref)


def prediction_date_from_path(path: str | Path, fallback: str | date | None = None) -> str:
    name = Path(path).name
    if name.startswith("previsoes_") and name.endswith(".csv"):
        return name.replace("previsoes_", "").replace(".csv", "")
    return _date_str(fallback)


def normal_prediction_files(pattern: str = "previsoes_*.csv", *, include_legacy: bool = False) -> list[Path]:
    files = list(PREDICTIONS_DIR.glob(pattern)) if PREDICTIONS_DIR.exists() else []
    if include_legacy and LEGACY_PREDICTIONS_DIR.exists():
        by_name = {path.name: path for path in sorted(LEGACY_PREDICTIONS_DIR.glob(pattern))}
        by_name.update({path.name: path for path in files})
        files = list(by_name.values())
    return sorted(files)


def historical_prediction_files(pattern: str = "previsoes_*.csv") -> list[Path]:
    if not HISTORICAL_PREDICTIONS_DIR.exists():
        return []
    return sorted(HISTORICAL_PREDICTIONS_DIR.glob(pattern))


def prediction_path_for_date(
    data_ref: str | date | None,
    *,
    include_historical: bool = True,
    include_legacy: bool = True,
) -> Path | None:
    if not data_ref:
        return None
    data = _date_str(data_ref)
    candidates = [normal_prediction_path(data)]
    if include_historical:
        candidates.append(historical_prediction_path(data))
    if include_legacy:
        candidates.append(legacy_prediction_path(data))
    return next((path for path in candidates if path.exists()), None)


def available_prediction_dates(*, include_historical: bool = True, include_legacy: bool = True) -> list[str]:
    dates: set[str] = set()
    files = normal_prediction_files(include_legacy=include_legacy)
    if include_historical:
        files.extend(historical_prediction_files())
    for path in files:
        data = prediction_date_from_path(path, fallback="")
        if data:
            dates.add(data)
    return sorted(dates)


def latest_prediction_path(
    selected_date: str | date | None = None,
    *,
    include_historical: bool = True,
    include_legacy: bool = True,
) -> Path | None:
    selected = prediction_path_for_date(
        selected_date,
        include_historical=include_historical,
        include_legacy=include_legacy,
    )
    if selected_date:
        return selected

    today = prediction_path_for_date(
        date.today().isoformat(),
        include_historical=False,
        include_legacy=include_legacy,
    )
    if today:
        return today

    files = normal_prediction_files(include_legacy=include_legacy)
    if include_historical:
        files.extend(historical_prediction_files())
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)
