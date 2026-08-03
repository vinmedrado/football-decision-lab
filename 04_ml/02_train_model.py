import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import pandas as pd
import numpy as np
import json
import os
import pickle
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if "BASE_DIR" in globals() else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(ROOT_DIR, "04_ml")

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from controles.mercados.status import get_mercado_lifecycle

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss

import lightgbm as lgb
import xgboost as xgb

# ==============================
# CONFIG
# ==============================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.getenv("ML_DATASET_INPUT_DIR", os.path.join(BASE_DIR, "datasets"))
MODEL_DIR   = os.getenv("ML_MODEL_OUTPUT_DIR", os.path.join(BASE_DIR, "models"))

os.makedirs(MODEL_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.60

# ROI mínimo do backtest para treinar modelo.
# Antes disso já existe filtro real: o 01_dataset_builder.py só gera dataset para
# um mercado quando ele passa no corte global OU tem liga elegível resgatada
# (04_ml/controles/mercados/elegibilidade_liga.py). Um mercado como R_FT_H pode
# chegar aqui com roi_bt global negativo mesmo tendo sido resgatado por uma boa
# liga -- esse número é a média de TODAS as ligas, boas e ruins juntas, não o
# resultado do subconjunto de dados que realmente foi usado para treinar.
# Por isso o padrão aqui é permissivo (não filtra de novo por cima). Se quiser
# reativar um corte extra nesta etapa, defina ML_TRAIN_MIN_ROI no ambiente.
MIN_ROI_BT = float(
    os.getenv("ML_TRAIN_MIN_ROI", "-1").replace(",", ".")
)

CALIBRATION_PCT = float(os.getenv("ML_CALIBRATION_PCT", "0.2").replace(",", "."))
MIN_CALIBRATION_SAMPLES = int(float(os.getenv("ML_MIN_CALIBRATION_SAMPLES", "100")))
MIN_CLASS_COUNT_FOR_ISOTONIC = int(float(os.getenv("ML_MIN_CLASS_COUNT_FOR_ISOTONIC", "50")))
VALIDATION_PCT = float(os.getenv("ML_VALIDATION_PCT", "0.2").replace(",", "."))
MIN_VALIDATION_SAMPLES = int(float(os.getenv("ML_MIN_VALIDATION_SAMPLES", "100")))

# ==============================
# MODELOS ESPECIALISTAS CONTEXTUAIS
# ==============================
# Treina modelos adicionais por mercado+liga, mercado+mandante, mercado+visitante
# e combinações liga+mandante/liga+visitante quando houver amostra suficiente.
ML_CONTEXT_MODELS_ENABLED = os.getenv("ML_CONTEXT_MODELS_ENABLED", "1").strip().lower() in {"1", "true", "sim", "s", "yes", "y"}
ML_CONTEXT_MIN_TRAIN = int(float(os.getenv("ML_CONTEXT_MIN_TRAIN", "500")))
ML_CONTEXT_MIN_TEST = int(float(os.getenv("ML_CONTEXT_MIN_TEST", "60")))

# Limiares mínimos por granularidade de contexto.
# "liga" continua alto porque uma liga tem muitos jogos (usa o default acima).
# Os demais níveis (por time e combinações time+liga) foram calibrados mais
# baixo de propósito: um time real dificilmente acumula 500 jogos históricos
# num único contexto, então usar o mesmo limiar da liga na prática impedia
# esses especialistas de nascer. Cada um pode ser ajustado via env var própria
# (ex.: ML_CONTEXT_MIN_TRAIN_HOME, ML_CONTEXT_MIN_TEST_CONFRONTO).
ML_CONTEXT_MIN_TRAIN_DEFAULTS = {
    "liga": ML_CONTEXT_MIN_TRAIN,
    "liga_confronto": 120,
    "confronto": 120,
    "liga_home": 150,
    "liga_away": 150,
    "home": 120,
    "away": 120,
}
ML_CONTEXT_MIN_TEST_DEFAULTS = {
    "liga": ML_CONTEXT_MIN_TEST,
    "liga_confronto": 20,
    "confronto": 20,
    "liga_home": 25,
    "liga_away": 25,
    "home": 20,
    "away": 20,
}


def _context_min_train(context_type: str) -> int:
    default = ML_CONTEXT_MIN_TRAIN_DEFAULTS.get(context_type, ML_CONTEXT_MIN_TRAIN)
    return int(float(os.getenv(f"ML_CONTEXT_MIN_TRAIN_{context_type.upper()}", str(default))))


def _context_min_validation(context_type: str) -> int:
    default = ML_CONTEXT_MIN_TEST_DEFAULTS.get(context_type, ML_CONTEXT_MIN_TEST)
    legacy_value = os.getenv(f"ML_CONTEXT_MIN_TEST_{context_type.upper()}", str(default))
    return int(float(os.getenv(f"ML_CONTEXT_MIN_VALIDATION_{context_type.upper()}", legacy_value)))
ML_CONTEXT_MAX_PER_TYPE = int(float(os.getenv("ML_CONTEXT_MAX_PER_TYPE", "80")))
ML_CONTEXT_MIN_AUC = float(os.getenv("ML_CONTEXT_MIN_AUC", "0.52").replace(",", "."))
# Piso absoluto: um especialista com AUC abaixo disso não tem poder de
# discriminação nenhum (pior que aleatório em muitos casos), então nenhum
# acc_confident alto o salva — em mercados de taxa-base alta (ex.: dupla
# chance), acc_confident pode ficar em 0.70-0.80 mesmo com AUC de 0.43,
# porque a métrica reflete a base rate do mercado, não o acerto do modelo.
ML_CONTEXT_MIN_AUC_FLOOR = float(os.getenv("ML_CONTEXT_MIN_AUC_FLOOR", "0.50").replace(",", "."))
ML_CONTEXT_MIN_ACC_CONF = float(os.getenv("ML_CONTEXT_MIN_ACC_CONF", "0.52").replace(",", "."))
ML_CONTEXT_RF_TREES = int(float(os.getenv("ML_CONTEXT_RF_TREES", "180")))
ML_CONTEXT_TYPES = tuple(
    x.strip()
    for x in os.getenv(
        "ML_CONTEXT_TYPES",
        "liga,home,away,confronto,liga_home,liga_away,liga_confronto",
    ).split(",")
    if x.strip()
)


# ==============================
# VALIDAÇÃO ANTI-LEAKAGE
# ==============================
SAFE_HISTORICAL_PREFIXES = ("ma_", "std_", "media_", "rolling_", "hist_", "lag_", "context_", "ctx_")

LEAKAGE_PATTERNS = (
    "target", "label", "y_", "result", "resultado", "placar", "score",
    "gols", "goals", "ft", "ht", "value", "profit", "lucro", "stake",
    "odd_result", "win", "won", "loss", "perdeu", "ganhou", "green", "red",
)

IMPOSSIBLE_AUC = 0.995
IMPOSSIBLE_ACCURACY = 0.995
IMPOSSIBLE_ACC_CONFIDENT = 0.995
IMPOSSIBLE_BRIER = 0.001

REPROVADA_MODELS_PATH = os.path.join(MODEL_DIR, "rejected_models.csv")
TRAINING_DIAGNOSTICS_PATH = os.path.join(MODEL_DIR, "training_diagnostics.csv")
BASELINE_METRICS_PATH = os.path.join(MODEL_DIR, "baseline_metrics.json")


def _normalizar_nome_coluna(coluna):
    return str(coluna).strip().lower()


def _coluna_tem_prefixo_historico_seguro(coluna):
    nome = _normalizar_nome_coluna(coluna)
    return nome.startswith(SAFE_HISTORICAL_PREFIXES)


def _coluna_suspeita_leakage(coluna):
    """Detecta nomes de coluna com alto risco de target/feature leakage.

    Prefixos históricos seguros são tolerados. As features contextuais
    context_* também são seguras porque foram calculadas no backtest com
    shift(1), usando apenas jogos anteriores daquele mercado/liga/time.
    """
    if _coluna_tem_prefixo_historico_seguro(coluna):
        return False

    nome = _normalizar_nome_coluna(coluna)
    tokens = [t for t in re.split(r"[^a-z0-9]+", nome) if t]

    for pattern in LEAKAGE_PATTERNS:
        pattern_norm = pattern.lower()
        if pattern_norm.endswith("_"):
            if nome.startswith(pattern_norm) or f"_{pattern_norm}" in nome:
                return True
            continue

        if pattern_norm in {"ft", "ht", "win", "won", "red"}:
            if pattern_norm in tokens:
                return True
            continue

        if pattern_norm in nome:
            return True

    return False


def validar_features_anti_leakage(X_train, X_reference):
    colunas = list(dict.fromkeys(list(X_train.columns) + list(X_reference.columns)))
    suspeitas = [c for c in colunas if _coluna_suspeita_leakage(c)]
    return suspeitas


def detectar_metricas_impossiveis(auc, accuracy, acc_confident, brier):
    motivos = []
    if auc is not None and auc >= IMPOSSIBLE_AUC:
        motivos.append(f"AUC >= {IMPOSSIBLE_AUC}")
    if accuracy is not None and accuracy >= IMPOSSIBLE_ACCURACY:
        motivos.append(f"accuracy >= {IMPOSSIBLE_ACCURACY}")
    if acc_confident is not None and acc_confident >= IMPOSSIBLE_ACC_CONFIDENT:
        motivos.append(f"acc_confident >= {IMPOSSIBLE_ACC_CONFIDENT}")
    if brier is not None and brier <= IMPOSSIBLE_BRIER:
        motivos.append(f"Brier Score <= {IMPOSSIBLE_BRIER}")
    return motivos


def remover_modelo_pickle_se_existir(model_mercado_dir):
    model_path = os.path.join(model_mercado_dir, "model.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)


def salvar_meta_rejeitado(model_mercado_dir, mercado, event, motivo, suspicious_columns=None, metrics=None):
    os.makedirs(model_mercado_dir, exist_ok=True)
    remover_modelo_pickle_se_existir(model_mercado_dir)

    metrics = metrics or {}
    meta_rejeitado = {
        "mercado": mercado,
        "event": event,
        "status": "rejected_leakage_suspected",
        "motivo": motivo,
        "suspicious_columns": suspicious_columns or [],
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "acc_confident": metrics.get("acc_confident"),
        "brier": metrics.get("brier"),
        "leakage_suspected": True,
    }
    with open(os.path.join(model_mercado_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_rejeitado, f)
    return meta_rejeitado


def append_rejection(rejected_rows, mercado, event, motivo, suspicious_columns=None, metrics=None):
    metrics = metrics or {}
    rejected_rows.append({
        "mercado": mercado,
        "event": event,
        "motivo": motivo,
        "suspicious_columns": ";".join(map(str, suspicious_columns or [])),
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "acc_confident": metrics.get("acc_confident"),
        "brier": metrics.get("brier"),
        "leakage_suspected": True,
    })


def append_diagnostic(diagnostics_rows, mercado, model_name, metrics=None, leakage_suspected=False, rejected=False, motivo="", calibration_method=None, calibration_status=None):
    metrics = metrics or {}
    diagnostics_rows.append({
        "mercado": mercado,
        "model_name": model_name,
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "acc_confident": metrics.get("acc_confident"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "coverage": metrics.get("coverage"),
        "calibration_method": calibration_method if calibration_method is not None else metrics.get("calibration_method"),
        "calibration_status": calibration_status if calibration_status is not None else metrics.get("calibration_status"),
        "evaluation_split": metrics.get("selection_metric_source", "pre_train_schema"),
        "leakage_suspected": bool(leakage_suspected),
        "rejected": bool(rejected),
        "motivo": motivo,
    })


def salvar_relatorios_diagnostico(rejected_rows, diagnostics_rows):
    rejected_columns = [
        "mercado", "event", "motivo", "suspicious_columns", "auc",
        "accuracy", "acc_confident", "brier", "leakage_suspected",
    ]
    diagnostics_columns = [
        "mercado", "model_name", "auc", "accuracy", "acc_confident", "brier",
        "log_loss", "coverage", "calibration_method", "calibration_status",
        "evaluation_split", "leakage_suspected", "rejected", "motivo",
    ]

    pd.DataFrame(rejected_rows, columns=rejected_columns).to_csv(REPROVADA_MODELS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics_rows, columns=diagnostics_columns).to_csv(TRAINING_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")



def salvar_baseline_metrics(resumo_modelos):
    """Salva baseline agregado de calibração para comparar runs futuras.

    O Brier menor é melhor. Este arquivo é usado pelo 09_calibration_guard.py
    para detectar degradação relevante antes de liberar operação automática.
    """
    import json
    from datetime import datetime, timezone

    df = pd.DataFrame(resumo_modelos)
    if df.empty:
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "EMPTY",
            "overall": {"brier_score": None, "auc": None, "accuracy": None},
            "mercados": [],
        }
    else:
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "OK",
            "overall": {
                "brier_score": round(float(pd.to_numeric(df["brier"], errors="coerce").mean()), 6),
                "auc": round(float(pd.to_numeric(df["auc"], errors="coerce").mean()), 6),
                "accuracy": round(float(pd.to_numeric(df["accuracy"], errors="coerce").mean()), 6),
                "models_count": int(len(df)),
            },
            "mercados": df[["mercado", "model_name", "auc", "accuracy", "brier", "roi_bt", "score"]].to_dict(orient="records"),
        }

    with open(BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


# ==============================
# LOAD META DOS MERCADOS
# ==============================
print("=" * 55)
print("📂 Carregando meta dos mercados...")

meta_path = os.path.join(DATASET_DIR, "mercados_meta.pkl")
if not os.path.exists(meta_path):
    raise FileNotFoundError("❌ mercados_meta.pkl não encontrado. Rode 01_dataset_builder.py primeiro.")

with open(meta_path, "rb") as f:
    mercados_meta = pickle.load(f)

mercados_meta = [m for m in mercados_meta if m["roi_bt"] >= MIN_ROI_BT]

lifecycle_removed = []
lifecycle_kept = []
for meta in list(mercados_meta):
    mercado = str(meta.get("mercado", ""))
    info = get_mercado_lifecycle(mercado)
    status = str(info.get("status_ciclo_vida") or "DESCONHECIDA").upper()
    if status in {"APOSENTADA", "BLOQUEADA"}:
        lifecycle_removed.append((mercado, status, info.get("motivo")))
    else:
        if status == "OBSERVACAO":
            meta["experimental_watchlist"] = True
        lifecycle_kept.append(meta)
mercados_meta = lifecycle_kept

if lifecycle_removed:
    print("  🛑 Mercados removidos por lifecycle antes do treino:")
    for mercado, status, motivo in lifecycle_removed:
        print(f"    - {mercado} — {status} — {motivo}")

print(f"  ✅ {len(mercados_meta)} mercados para treinar (ROI_bt >= {MIN_ROI_BT})")

if len(mercados_meta) == 0:
    print("\n⚠️ Nenhum mercado elegível para treinar.")
    print("   Isso acontece quando o dataset builder não encontra mercados com ROI/apostas mínimos.")
    print("   Rode o backtest completo ou ajuste os filtros do 01_dataset_builder.py.")
    vazio_path = os.path.join(MODEL_DIR, "resumo_modelos.pkl")
    with open(vazio_path, "wb") as f:
        pickle.dump([], f)
    salvar_relatorios_diagnostico([], [])
    salvar_baseline_metrics([])
    status_path = os.path.join(MODEL_DIR, "training_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "NO_DEPLOYABLE_MODEL",
                "deployable_models": 0,
                "reason": "nenhum_mercado_elegivel",
                "resumo_modelos": vazio_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"🛑 Resumo vazio salvo em: {vazio_path}")
    print(f"🛑 Nenhum modelo publicável. Estado salvo em: {status_path}")
    raise SystemExit(2)

# ==============================
# DEFINIÇÃO DOS MODELOS
# ==============================
def get_models(y_train):
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=20,
            scale_pos_weight=scale,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }


def safe_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def safe_log_loss(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return None
    return log_loss(y_true, y_prob, labels=[0, 1])


def evaluate_classifier(model, X_eval, y_eval):
    """Calcula metricas sem executar qualquer decisao de selecao."""
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    mask_conf = (y_prob >= CONFIDENCE_THRESHOLD) | (y_prob <= (1 - CONFIDENCE_THRESHOLD))
    coverage = float(mask_conf.mean())
    acc_confident = (
        accuracy_score(y_eval[mask_conf], y_pred[mask_conf])
        if mask_conf.sum() > 0
        else 0.0
    )
    auc = safe_auc(y_eval, y_prob)
    return {
        "auc": auc,
        "accuracy": accuracy_score(y_eval, y_pred),
        "log_loss": safe_log_loss(y_eval, y_prob),
        "brier": brier_score_loss(y_eval, y_prob),
        "coverage": coverage,
        "acc_confident": acc_confident,
        "score": 0.6 * auc + 0.4 * acc_confident,
        "n_samples": int(len(y_eval)),
    }


def _normalizar_datas_split(dates, expected_size, nome):
    series = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    if len(series) != expected_size:
        raise ValueError(f"{nome}: datas={len(series)} diferente de linhas={expected_size}")
    if series.isna().any():
        raise ValueError(f"{nome}: existem datas invalidas")
    return series


def validar_particoes_temporais(partitions):
    """Prova disjuncao e ordem cronologica entre particoes nomeadas."""
    normalized = []
    for name, dates in partitions:
        series = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
        if series.empty:
            continue
        normalized.append((name, series))

    for index, (left_name, left_dates) in enumerate(normalized):
        left_unique = set(left_dates.dt.normalize())
        for right_name, right_dates in normalized[index + 1:]:
            overlap = left_unique.intersection(set(right_dates.dt.normalize()))
            if overlap:
                sample = sorted(pd.Timestamp(value).date().isoformat() for value in overlap)[:10]
                raise AssertionError(
                    f"leakage temporal entre {left_name} e {right_name}: {sample}"
                )

    for (left_name, left_dates), (right_name, right_dates) in zip(normalized, normalized[1:]):
        if left_dates.max() >= right_dates.min():
            raise AssertionError(
                f"ordem temporal invalida: {left_name} termina em {left_dates.max()} "
                f"e {right_name} inicia em {right_dates.min()}"
            )


def split_train_validation_temporal(X_train, y_train, dates_train):
    """Reserva a janela pre-teste mais recente apenas para selecao/validacao."""
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    dates = _normalizar_datas_split(dates_train, len(X_train), "pre_teste")
    counts = dates.value_counts(sort=False).sort_index()
    if len(counts) < 3:
        raise ValueError("split treino/validacao exige pelo menos tres datas distintas")

    validation_size = max(int(len(X_train) * VALIDATION_PCT), MIN_VALIDATION_SAMPLES)
    validation_size = min(validation_size, len(X_train) - 1)
    target_train_size = len(X_train) - validation_size
    cumulative = counts.cumsum().iloc[:-1]
    cutoff_date = (cumulative - target_train_size).abs().idxmin()
    train_mask = dates <= cutoff_date

    X_selection_train = X_train.loc[train_mask].copy()
    y_selection_train = y_train.loc[train_mask].copy()
    dates_selection_train = dates.loc[train_mask].copy()
    X_validation = X_train.loc[~train_mask].copy()
    y_validation = y_train.loc[~train_mask].copy()
    dates_validation = dates.loc[~train_mask].copy()

    validar_particoes_temporais([
        ("treino_pre_selecao", dates_selection_train),
        ("validacao", dates_validation),
    ])
    if y_selection_train.nunique() < 2:
        raise ValueError("treino pre-selecao sem duas classes")
    if y_validation.nunique() < 2:
        raise ValueError("validacao sem duas classes")

    return (
        X_selection_train,
        y_selection_train,
        dates_selection_train,
        X_validation,
        y_validation,
        dates_validation,
    )


def split_train_calibration_temporal(X_train, y_train, dates_train):
    """Separa treino-base e calibracao por grupos completos de datas."""
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    n_total = len(X_train)
    if n_total < 3:
        return X_train, y_train, None, None, None, None, "sem_calibracao_amostra_insuficiente"

    dates = _normalizar_datas_split(dates_train, n_total, "treino")
    counts = dates.value_counts(sort=False).sort_index()
    if len(counts) < 2:
        return X_train, y_train, None, None, dates, None, "sem_calibracao_datas_insuficientes"

    calib_size = max(int(n_total * CALIBRATION_PCT), MIN_CALIBRATION_SAMPLES)
    calib_size = min(calib_size, n_total - 1)
    target_base_size = n_total - calib_size
    cumulative = counts.cumsum().iloc[:-1]
    cutoff_date = (cumulative - target_base_size).abs().idxmin()
    base_mask = dates <= cutoff_date

    X_base = X_train.loc[base_mask].copy()
    y_base = y_train.loc[base_mask].copy()
    dates_base = dates.loc[base_mask].copy()
    X_calib = X_train.loc[~base_mask].copy()
    y_calib = y_train.loc[~base_mask].copy()
    dates_calib = dates.loc[~base_mask].copy()

    validar_particoes_temporais([
        ("treino_base", dates_base),
        ("calibracao", dates_calib),
    ])

    if y_base.nunique() < 2:
        return X_train, y_train, None, None, dates, None, "sem_calibracao_treino_base_sem_duas_classes"
    if y_calib.nunique() < 2:
        return X_train, y_train, None, None, dates, None, "sem_calibracao_calibracao_sem_duas_classes"

    return X_base, y_base, X_calib, y_calib, dates_base, dates_calib, "calibracao_temporal_por_data"


def calibrate_prefit_model(model, X_calib, y_calib):
    """Calibra modelo já treinado sem usar o teste final.

    Usa isotonic quando há amostra mínima por classe; caso contrário usa
    sigmoid, que é mais estável para amostras menores.
    """
    class_counts = y_calib.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    method = "isotonic" if min_class_count >= MIN_CLASS_COUNT_FOR_ISOTONIC else "sigmoid"

    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except Exception:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")

    calibrated.fit(X_calib, y_calib)
    return calibrated, method



# ==============================
# TREINO DE ESPECIALISTAS CONTEXTUAIS
# ==============================
def _safe_context_key(value):
    return str(value).strip().upper()


def _decode_label_value(encoders, col, value):
    """Converte código do LabelEncoder para texto original quando possível."""
    try:
        if col in encoders:
            idx = int(float(value))
            classes = list(encoders[col].classes_)
            if 0 <= idx < len(classes):
                return str(classes[idx])
    except Exception:
        pass
    return str(value)


def _load_label_encoders():
    path = os.path.join(DATASET_DIR, "label_encoders.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _context_group_specs():
    return {
        "liga": ["League_std"],
        "home": ["Home"],
        "away": ["Away"],
        "confronto": ["Home", "Away"],
        "liga_home": ["League_std", "Home"],
        "liga_away": ["League_std", "Away"],
        "liga_confronto": ["League_std", "Home", "Away"],
    }


def _context_raw_key(encoders, cols, values):
    raw_parts = []
    for col, value in zip(cols, values):
        raw_parts.append(_safe_context_key(_decode_label_value(encoders, col, value)))
    return "||".join(raw_parts)


def _build_context_model(y_train):
    return RandomForestClassifier(
        n_estimators=ML_CONTEXT_RF_TREES,
        max_depth=9,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _fit_eval_context_model(
    *,
    X_train_ctx,
    y_train_ctx,
    dates_train_ctx,
    X_validation_ctx,
    y_validation_ctx,
    global_score,
    min_train,
    min_validation,
):
    """Treina especialista no treino e decide sua aceitacao apenas na validacao."""
    if len(X_train_ctx) < min_train or len(X_validation_ctx) < min_validation:
        return None
    if pd.Series(y_train_ctx).nunique() < 2 or pd.Series(y_validation_ctx).nunique() < 2:
        return None

    X_base, y_base, X_calib, y_calib, _, _, calibration_status = split_train_calibration_temporal(
        X_train_ctx,
        y_train_ctx,
        dates_train_ctx,
    )
    if pd.Series(y_base).nunique() < 2:
        return None

    model = _build_context_model(y_base)
    model.fit(X_base, y_base)

    if X_calib is not None and y_calib is not None:
        try:
            final_model, calibration_method = calibrate_prefit_model(model, X_calib, y_calib)
        except Exception:
            final_model = model
            calibration_method = "none"
    else:
        final_model = model
        calibration_method = "none"

    validation_metrics = evaluate_classifier(final_model, X_validation_ctx, y_validation_ctx)
    auc = validation_metrics["auc"]
    acc = validation_metrics["accuracy"]
    brier = validation_metrics["brier"]
    lloss = validation_metrics["log_loss"]
    coverage = validation_metrics["coverage"]
    acc_confident = validation_metrics["acc_confident"]
    score = validation_metrics["score"]
    y_prob = final_model.predict_proba(X_validation_ctx)[:, 1]
    n_confident = int(
        ((y_prob >= CONFIDENCE_THRESHOLD) | (y_prob <= (1 - CONFIDENCE_THRESHOLD))).sum()
    )

    # Mesma blindagem anti-leakage/métricas impossíveis do modelo global, agora
    # aplicada ao especialista. Amostras pequenas por contexto (time/confronto)
    # são exatamente onde uma métrica "boa demais" é mais provável de ser fluke
    # em vez de sinal real.
    motivos_impossiveis = detectar_metricas_impossiveis(auc, acc, acc_confident, brier)
    if motivos_impossiveis:
        return None

    # acc_confident so e criterio de aprovacao quando a validacao contem amostra
    # minima de previsoes confiantes; o teste final nao participa desta decisao.
    # por sorte (ex.: 2 de 2) inflam a métrica sem significância nenhuma.
    MIN_CONFIDENT_SAMPLES_CONTEXT = 10
    acc_confident_valido = n_confident >= MIN_CONFIDENT_SAMPLES_CONTEXT

    # Piso absoluto de discriminação: abaixo disso, o especialista não entra
    # de jeito nenhum, mesmo que acc_confident pareça bom (base rate do
    # mercado, não sinal real).
    if auc < ML_CONTEXT_MIN_AUC_FLOOR:
        return None

    if auc < ML_CONTEXT_MIN_AUC and not (acc_confident_valido and acc_confident >= ML_CONTEXT_MIN_ACC_CONF):
        return None

    return {
        "model": final_model,
        "auc": float(auc),
        "accuracy": float(acc),
        "brier": float(brier),
        "log_loss": lloss,
        "coverage": float(coverage),
        "acc_confident": float(acc_confident),
        "score": float(score),
        "global_score": float(global_score),
        "calibration_method": calibration_method,
        "calibration_status": calibration_status,
        "n_train": int(len(X_train_ctx)),
        "n_validation": int(len(X_validation_ctx)),
        "n_test": 0,
        "selection_metric_source": "validation",
        "fusion_sample_source": "fixed_no_test",
    }


def treinar_especialistas_contextuais(
    *,
    mercado,
    event,
    X_train,
    y_train,
    dates_train,
    X_validation,
    y_validation,
    best_global,
    model_mercado_dir,
):
    """Treina especialistas por contexto e salva registry em models/<mercado>/context_models.pkl.

    Hierarquia usada depois pelo predict:
    liga_confronto -> confronto -> liga_home -> liga_away -> liga -> home -> away -> global.
    """
    if not ML_CONTEXT_MODELS_ENABLED:
        return []

    encoders = _load_label_encoders()
    specs = _context_group_specs()
    specialists = []
    used_keys = set()

    for context_type in ML_CONTEXT_TYPES:
        cols = specs.get(context_type)
        if not cols or not all(c in X_train.columns and c in X_validation.columns for c in cols):
            continue

        # Prioriza contextos com maior amostra no treino.
        group_sizes = (
            X_train.groupby(cols, dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(max(ML_CONTEXT_MAX_PER_TYPE, 1))
        )

        trained_this_type = 0
        for group_key, _ in group_sizes.items():
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            raw_key = _context_raw_key(encoders, cols, values)
            registry_key = (context_type, raw_key)
            if registry_key in used_keys:
                continue
            used_keys.add(registry_key)

            train_mask = pd.Series(True, index=X_train.index)
            validation_mask = pd.Series(True, index=X_validation.index)
            for col, value in zip(cols, values):
                train_mask &= X_train[col].eq(value)
                validation_mask &= X_validation[col].eq(value)

            X_train_ctx = X_train.loc[train_mask].copy()
            y_train_ctx = y_train.loc[train_mask].copy()
            dates_train_ctx = dates_train.loc[train_mask].copy()
            X_validation_ctx = X_validation.loc[validation_mask].copy()
            y_validation_ctx = y_validation.loc[validation_mask].copy()

            result = _fit_eval_context_model(
                X_train_ctx=X_train_ctx,
                y_train_ctx=y_train_ctx,
                dates_train_ctx=dates_train_ctx,
                X_validation_ctx=X_validation_ctx,
                y_validation_ctx=y_validation_ctx,
                global_score=float(best_global.get("score", 0.0)),
                min_train=_context_min_train(context_type),
                min_validation=_context_min_validation(context_type),
            )
            if result is None:
                continue

            specialists.append({
                "mercado": mercado,
                "event": event,
                "context_type": context_type,
                "context_columns": cols,
                "context_values_encoded": [float(v) for v in values],
                "context_key": raw_key,
                "model": result.pop("model"),
                **result,
            })
            trained_this_type += 1

    context_path = os.path.join(model_mercado_dir, "context_models.pkl")
    with open(context_path, "wb") as f:
        pickle.dump(specialists, f)

    if specialists:
        print(f"     🧠 Especialistas contextuais salvos: {len(specialists)}")
        by_type = pd.Series([s["context_type"] for s in specialists]).value_counts().to_dict()
        print(f"        Por tipo: {by_type}")
    else:
        print("     🧠 Nenhum especialista contextual aprovado para este mercado.")

    return specialists


# ==============================
# TREINO POR MERCADO
# ==============================
print("\n" + "=" * 55)
print("🏋️  Treinando modelos por mercado...\n")

resumo_modelos = []
rejected_rows = []
diagnostics_rows = []

for meta in mercados_meta:

    mercado = str(meta.get("mercado") or meta.get("market") or "").strip()
    event = str(meta.get("event") or mercado).strip()
    roi_bt = float(meta.get("roi_bt", 0))

    if not mercado:
        print("  [AVISO] Meta sem mercado/market. Ignorando registro.")
        continue

    mercado_dir = os.path.join(DATASET_DIR, mercado)

    print(f"  🔄 [{mercado}] {event} | ROI_bt={roi_bt:+.3f}")

    # --- Carrega datasets ---
    X_train = pd.read_csv(os.path.join(mercado_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(mercado_dir, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(mercado_dir, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(mercado_dir, "y_test.csv")).squeeze()
    dates_train_path = os.path.join(mercado_dir, "dates_train.csv")
    dates_test_path = os.path.join(mercado_dir, "dates_test.csv")
    if not os.path.exists(dates_train_path) or not os.path.exists(dates_test_path):
        raise FileNotFoundError(
            f"{mercado}: metadados temporais ausentes. Rode 01_dataset_builder.py novamente."
        )
    dates_train = _normalizar_datas_split(
        pd.read_csv(dates_train_path)["Date"], len(X_train), f"{mercado}/treino"
    )
    dates_test = _normalizar_datas_split(
        pd.read_csv(dates_test_path)["Date"], len(X_test), f"{mercado}/teste"
    )
    validar_particoes_temporais([
        ("treino_total", dates_train),
        ("teste", dates_test),
    ])

    suspicious_columns = validar_features_anti_leakage(X_train, X_train)
    if suspicious_columns:
        motivo = "colunas_suspeitas_de_leakage"
        print("     ⛔ Mercado rejeitado por possível leakage nas features:")
        for col in suspicious_columns:
            print(f"        - {col}")

        model_mercado_dir = os.path.join(MODEL_DIR, mercado)
        salvar_meta_rejeitado(model_mercado_dir, mercado, event, motivo, suspicious_columns=suspicious_columns)
        append_rejection(rejected_rows, mercado, event, motivo, suspicious_columns=suspicious_columns)
        append_diagnostic(
            diagnostics_rows,
            mercado=mercado,
            model_name="anti_leakage_pre_train",
            leakage_suspected=True,
            rejected=True,
            motivo=f"{motivo}: {';'.join(map(str, suspicious_columns))}",
        )
        continue

    (
        X_selection_train,
        y_selection_train,
        dates_selection_train,
        X_validation,
        y_validation,
        dates_validation,
    ) = split_train_validation_temporal(X_train, y_train, dates_train)

    X_base, y_base, X_calib, y_calib, dates_base, dates_calib, calibration_status = split_train_calibration_temporal(
        X_selection_train,
        y_selection_train,
        dates_selection_train,
    )
    validar_particoes_temporais([
        ("treino_base", dates_base),
        ("calibracao", dates_calib),
        ("validacao", dates_validation),
        ("teste_final", dates_test),
    ])

    print(
        f"     Split temporal ML: base={len(X_base)} | "
        f"calibracao={0 if X_calib is None else len(X_calib)} | "
        f"validacao={len(X_validation)} | teste_final={len(X_test)} | "
        f"status={calibration_status}"
    )

    models  = get_models(y_base)
    results = {}

    for name, model in models.items():
        model.fit(X_base, y_base)

        if X_calib is not None and y_calib is not None:
            final_model, calibration_method = calibrate_prefit_model(model, X_calib, y_calib)
        else:
            final_model = model
            calibration_method = "none"

        validation_metrics = evaluate_classifier(final_model, X_validation, y_validation)
        metricas_impossiveis = detectar_metricas_impossiveis(
            validation_metrics["auc"],
            validation_metrics["accuracy"],
            validation_metrics["acc_confident"],
            validation_metrics["brier"],
        )
        leakage_suspected = len(metricas_impossiveis) > 0
        motivo = "; ".join(metricas_impossiveis) if leakage_suspected else ""

        results[name] = {
            "model"             : final_model,
            **validation_metrics,
            "calibration_method": calibration_method,
            "calibration_status": calibration_status,
            "selection_metric_source": "validation",
            "leakage_suspected" : leakage_suspected,
            "rejected"          : leakage_suspected,
            "motivo"            : motivo,
        }

        append_diagnostic(
            diagnostics_rows,
            mercado=mercado,
            model_name=name,
            metrics=results[name],
            leakage_suspected=leakage_suspected,
            rejected=leakage_suspected,
            motivo=motivo,
        )

        if leakage_suspected:
            print(
                f"     ⛔ {name} rejeitado por métricas impossíveis: {motivo}"
            )

    # --- Melhor modelo aprovado ---
    approved_results = {n: r for n, r in results.items() if not r.get("leakage_suspected", False)}
    model_mercado_dir = os.path.join(MODEL_DIR, mercado)

    if not approved_results:
        motivo = "todos_os_modelos_com_suspeita_de_leakage"
        metricas_referencia = max(results.values(), key=lambda r: r["score"]) if results else {}
        print(f"     ⛔ Mercado rejeitado: {motivo}")
        salvar_meta_rejeitado(
            model_mercado_dir,
            mercado,
            event,
            motivo,
            suspicious_columns=[],
            metrics=metricas_referencia,
        )
        append_rejection(
            rejected_rows,
            mercado,
            event,
            motivo,
            suspicious_columns=[],
            metrics=metricas_referencia,
        )
        continue

    best_name = max(approved_results, key=lambda n: approved_results[n]["score"])
    best      = approved_results[best_name]

    print(
        f"     Selecionado na validacao: {best_name} | AUC={best['auc']:.4f} | "
        f"Acuracia>={int(CONFIDENCE_THRESHOLD*100)}%={best['acc_confident']:.4f} | "
        f"Cobertura={best['coverage']:.1%}"
    )

    # Especialistas e parametros consumidos pela Fusion sao congelados usando
    # somente treino/calibracao/validacao, antes de abrir o teste final.
    os.makedirs(model_mercado_dir, exist_ok=True)
    especialistas = treinar_especialistas_contextuais(
        mercado=mercado,
        event=event,
        X_train=X_selection_train,
        y_train=y_selection_train,
        dates_train=dates_selection_train,
        X_validation=X_validation,
        y_validation=y_validation,
        best_global=best,
        model_mercado_dir=model_mercado_dir,
    )

    # O vencedor ja esta congelado. Estas metricas finais sao somente reportadas
    # e nao participam de aprovacao, selecao, calibracao, thresholds ou fusion.
    final_test_metrics = evaluate_classifier(best["model"], X_test, y_test)
    final_test_alerts = detectar_metricas_impossiveis(
        final_test_metrics["auc"],
        final_test_metrics["accuracy"],
        final_test_metrics["acc_confident"],
        final_test_metrics["brier"],
    )
    print(
        f"     Teste final (somente medicao): AUC={final_test_metrics['auc']:.4f} | "
        f"Brier={final_test_metrics['brier']:.4f} | "
        f"LogLoss={final_test_metrics['log_loss']}"
    )

    # --- Salva modelo aprovado do mercado ---
    os.makedirs(model_mercado_dir, exist_ok=True)

    with open(os.path.join(model_mercado_dir, "model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)

    model_meta = {
        "mercado"              : mercado,
        "event"               : event,
        "model_name"          : best_name,
        "auc"                 : best["auc"],
        "accuracy"            : best["accuracy"],
        "brier"               : best["brier"],
        "log_loss"            : best["log_loss"],
        "score"               : best["score"],
        "coverage"            : best["coverage"],
        "acc_confident"       : best["acc_confident"],
        "roi_bt"              : roi_bt,
        "winrate_bt"          : meta.get("winrate_bt", 0),
        "confianca_threshold": CONFIDENCE_THRESHOLD,
        "split_type"          : "temporal_train_calibration_validation_test_final",
        "n_train_total"       : len(X_train),
        "n_selection_train"   : len(X_selection_train),
        "n_train_base"        : len(X_base),
        "n_calibration"       : 0 if X_calib is None else len(X_calib),
        "n_validation"        : len(X_validation),
        "n_test"              : len(X_test),
        "calibration_method"  : best["calibration_method"],
        "calibration_status"  : best["calibration_status"],
        "leakage_suspected"   : False,
        "status"              : "approved",
        "dataset_split_date"  : meta.get("split_date"),
        "train_start"         : meta.get("train_start"),
        "train_end"           : meta.get("train_end"),
        "test_start"          : meta.get("test_start"),
        "test_end"            : meta.get("test_end"),
        "base_train_start"    : None if dates_base is None else str(dates_base.min().date()),
        "base_train_end"      : None if dates_base is None else str(dates_base.max().date()),
        "calibration_start"   : None if dates_calib is None else str(dates_calib.min().date()),
        "calibration_end"     : None if dates_calib is None else str(dates_calib.max().date()),
        "validation_start"    : str(dates_validation.min().date()),
        "validation_end"      : str(dates_validation.max().date()),
        "algorithm_selection_source": "validation",
        "specialist_selection_source": "validation",
        "selection_metric_source": "validation",
        "fusion_metric_source": "validation",
        "hyperparameter_source": "fixed_configuration",
        "threshold_source"    : "fixed_configuration",
        "test_metrics_used_for_selection": False,
        "test_auc"            : final_test_metrics["auc"],
        "test_accuracy"       : final_test_metrics["accuracy"],
        "test_brier"          : final_test_metrics["brier"],
        "test_log_loss"       : final_test_metrics["log_loss"],
        "test_coverage"       : final_test_metrics["coverage"],
        "test_acc_confident"  : final_test_metrics["acc_confident"],
        "test_score_report_only": final_test_metrics["score"],
        "test_quality_alerts_report_only": final_test_alerts,
        "date_overlap_verified": True,
    }

    with open(os.path.join(model_mercado_dir, "meta.pkl"), "wb") as f:
        pickle.dump(model_meta, f)

    model_meta["context_models_count"] = len(especialistas)
    model_meta["context_models_enabled"] = bool(ML_CONTEXT_MODELS_ENABLED)

    # Atualiza meta com quantidade de especialistas.
    with open(os.path.join(model_mercado_dir, "meta.pkl"), "wb") as f:
        pickle.dump(model_meta, f)

    resumo_modelos.append(model_meta)

# ==============================
# RELATÓRIO FINAL
# ==============================
print("\n" + "=" * 55)
print("📊 RESUMO DOS MODELOS POR MERCADO:")
print("=" * 55)

if not resumo_modelos:
    print("⚠️ Nenhum modelo aprovado: suspeita de leakage.")
    resumo_path = os.path.join(MODEL_DIR, "resumo_modelos.pkl")
    with open(resumo_path, "wb") as f:
        pickle.dump([], f)
    salvar_relatorios_diagnostico(rejected_rows, diagnostics_rows)
    salvar_baseline_metrics([])
    status_path = os.path.join(MODEL_DIR, "training_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "NO_DEPLOYABLE_MODEL",
                "deployable_models": 0,
                "reason": "nenhum_modelo_aprovado",
                "resumo_modelos": resumo_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"🛑 Resumo vazio salvo em: {resumo_path}")
    print(f"🛑 Nenhum modelo publicável. Estado salvo em: {status_path}")
    raise SystemExit(2)

df_resumo = pd.DataFrame(resumo_modelos).sort_values("score", ascending=False)

for _, r in df_resumo.iterrows():
    print(
        f"  {r['mercado']:<12} | {r['model_name']:<15} | "
        f"AUC={r['auc']:.4f} | "
        f"Acurácia≥{int(CONFIDENCE_THRESHOLD*100)}%={r['acc_confident']:.4f} | "
        f"ROI_bt={r['roi_bt']:+.3f} | "
        f"ctx={int(r.get('context_models_count', 0))}"
    )

# Salva resumo global dos modelos
resumo_path = os.path.join(MODEL_DIR, "resumo_modelos.pkl")
with open(resumo_path, "wb") as f:
    pickle.dump(resumo_modelos, f)

salvar_relatorios_diagnostico(rejected_rows, diagnostics_rows)
baseline_metrics = salvar_baseline_metrics(resumo_modelos)
print(f"✅ Baseline de calibração salvo em: {BASELINE_METRICS_PATH}")
print(f"   Brier médio baseline: {baseline_metrics.get('overall', {}).get('brier_score')}")

print(f"\n✅ {len(resumo_modelos)} modelos treinados e salvos em: {MODEL_DIR}")
print("=" * 55)
