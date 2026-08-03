import math
import numpy as np
import pandas as pd


# ==============================
# HELPERS TEMPORAIS SEM LEAKAGE
# ==============================
def _sorted_copy(df):
    return df.sort_values("Date").reset_index(drop=True)


def _safe_std(series):
    return series.replace(0, 0.0001).fillna(0.0001)


# ==============================
# MÉDIA MÓVEL GLOBAL (POR LIGA)
# shift(1) dentro da liga garante que só usa jogos ANTERIORES
# ==============================
def calcular_media_movel(df, window):
    df = _sorted_copy(df)
    df["ma"] = df.groupby("League_std", group_keys=False)["value"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


# ==============================
# DESVIO PADRÃO GLOBAL (POR LIGA)
# shift(1) dentro da liga garante que só usa jogos ANTERIORES
# ==============================
def calcular_desvio_padrao(df, window):
    df = _sorted_copy(df)
    df["std"] = df.groupby("League_std", group_keys=False)["value"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=2).std()
    )
    df["std"] = _safe_std(df["std"])
    return df


# ==============================
# Z-SCORE GLOBAL PRÉ-JOGO
# Não usa value atual.
# Compara a média móvel disponível antes do jogo contra o histórico anterior
# dessa própria média dentro da liga.
# ==============================
def calcular_zscore(df):
    df = _sorted_copy(df)
    baseline_mean = df.groupby("League_std", group_keys=False)["ma"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    baseline_std = df.groupby("League_std", group_keys=False)["ma"].apply(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    baseline_std = _safe_std(baseline_std)
    df["zscore"] = ((df["ma"] - baseline_mean) / baseline_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(-5, 5)
    return df


# ==============================
# MÉDIA POR TIME
# shift(1) dentro do time garante que só usa jogos ANTERIORES do time
# ==============================
def calcular_media_movel_grupo(df, group_col, window):
    df = _sorted_copy(df)
    df[f"ma_{group_col}"] = df.groupby(group_col, group_keys=False)["value"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


# ==============================
# DESVIO POR TIME
# shift(1) dentro do time garante que só usa jogos ANTERIORES do time
# ==============================
def calcular_std_grupo(df, group_col, window):
    df = _sorted_copy(df)
    df[f"std_{group_col}"] = df.groupby(group_col, group_keys=False)["value"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=2).std()
    )
    df[f"std_{group_col}"] = _safe_std(df[f"std_{group_col}"])
    return df


# ==============================
# Z-SCORE POR TIME PRÉ-JOGO
# Não usa value atual.
# Compara a média móvel pré-jogo do time contra o histórico anterior
# dessa própria média do time.
# ==============================
def calcular_zscore_grupo(df, group_col):
    df = _sorted_copy(df)
    ma_col = f"ma_{group_col}"
    z_col = f"zscore_{group_col}"
    baseline_mean = df.groupby(group_col, group_keys=False)[ma_col].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    baseline_std = df.groupby(group_col, group_keys=False)[ma_col].apply(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    baseline_std = _safe_std(baseline_std)
    df[z_col] = ((df[ma_col] - baseline_mean) / baseline_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(-5, 5)
    return df


# ==============================
# MÉDIA EXPANSIVA LIGA
# shift(1) dentro da liga garante que só usa jogos ANTERIORES da liga
# ==============================
def calcular_media_liga(df):
    df = _sorted_copy(df)
    df["liga_mean"] = df.groupby("League_std", group_keys=False)["value"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return df


# ==============================
# DESVIO EXPANSIVO LIGA
# shift(1) dentro da liga garante que só usa jogos ANTERIORES da liga
# ==============================
def calcular_std_liga(df):
    df = _sorted_copy(df)
    df["liga_std"] = df.groupby("League_std", group_keys=False)["value"].apply(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    df["liga_std"] = _safe_std(df["liga_std"])
    return df


# ==============================
# Z-SCORE LIGA PRÉ-JOGO
# Não usa value atual.
# Compara a média expansiva disponível antes do jogo contra o histórico
# anterior dessa própria média dentro da liga.
# ==============================
def calcular_zscore_liga(df):
    df = _sorted_copy(df)
    baseline_mean = df.groupby("League_std", group_keys=False)["liga_mean"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    baseline_std = df.groupby("League_std", group_keys=False)["liga_mean"].apply(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    baseline_std = _safe_std(baseline_std)
    df["zscore_liga"] = ((df["liga_mean"] - baseline_mean) / baseline_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(-5, 5)
    return df


# ==============================
# SINAL INTELIGENTE PRÉ-JOGO
# Usa somente z-scores construídos com estatísticas disponíveis até T-1.
# Não lê target, retorno nem value atual.
# ==============================
def gerar_sinal_inteligente(df, z_global=1.5, z_liga=1.0, z_team=1.0):
    z_global_now = df.get("zscore", pd.Series(0, index=df.index)).fillna(0)
    z_liga_now = df.get("zscore_liga", pd.Series(0, index=df.index)).fillna(0)
    z_home_now = df.get("zscore_Home", pd.Series(0, index=df.index)).fillna(0)
    z_away_now = df.get("zscore_Away", pd.Series(0, index=df.index)).fillna(0)
    z_team_combined = (z_home_now + z_away_now) / 2

    score = (
        (z_global_now > z_global).astype(int)
        + (z_liga_now > z_liga).astype(int)
        + (z_team_combined > z_team).astype(int)
    )
    df["signal"] = score >= 2
    return df


# ==============================
# WALK-FORWARD EXPANDING WINDOW
# Treina/estima estatística até T-1 e testa em T.
# ==============================
def aplicar_walk_forward_probabilidade(
    df,
    target_col,
    min_train_games=100,
    probability_floor=0.01,
    probability_cap=0.99,
):
    df = _sorted_copy(df)
    min_train_games = max(int(min_train_games or 1), 1)

    global_prev_sum = df[target_col].shift(1).expanding(min_periods=1).sum()
    global_prev_count = df[target_col].shift(1).expanding(min_periods=1).count()

    liga_prev_sum = df.groupby("League_std", group_keys=False)[target_col].apply(
        lambda s: s.shift(1).expanding(min_periods=1).sum()
    )
    liga_prev_count = df.groupby("League_std", group_keys=False)[target_col].apply(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )

    global_prob = global_prev_sum / global_prev_count.replace(0, np.nan)
    liga_prob = liga_prev_sum / liga_prev_count.replace(0, np.nan)

    df["wf_train_games"] = liga_prev_count.fillna(0).astype(int)
    df["wf_probability"] = liga_prob.fillna(global_prob).fillna(0.5)
    df["wf_probability"] = df["wf_probability"].clip(probability_floor, probability_cap)
    df["wf_ready"] = df["wf_train_games"] >= min_train_games
    return df



# ==============================
# CONTEXTO MERCADO/LIGA/TIME SEM LEAKAGE
# ==============================
def _expanding_group_stat(df, group_cols, value_col, out_prefix, min_periods=1):
    """Cria estatísticas históricas até T-1 por grupo.

    Usa shift(1) antes do expanding, então a linha atual nunca entra na própria feature.
    """
    df = _sorted_copy(df)
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    missing = [c for c in group_cols if c not in df.columns]
    if missing or value_col not in df.columns:
        df[f"{out_prefix}_count"] = 0
        df[f"{out_prefix}_mean"] = 0.0
        df[f"{out_prefix}_std"] = 0.0001
        return df

    value = pd.to_numeric(df[value_col], errors="coerce")
    key = [df[c].astype(str).fillna("").str.strip() for c in group_cols]

    prev_count = value.groupby(key, group_keys=False).apply(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )
    prev_mean = value.groupby(key, group_keys=False).apply(
        lambda s: s.shift(1).expanding(min_periods=min_periods).mean()
    )
    prev_std = value.groupby(key, group_keys=False).apply(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )

    df[f"{out_prefix}_count"] = prev_count.fillna(0).astype(int).values
    df[f"{out_prefix}_mean"] = prev_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    df[f"{out_prefix}_std"] = _safe_std(prev_std).replace([np.inf, -np.inf], np.nan).fillna(0.0001).values
    return df


def calcular_contexto_mercado(df, target_col, min_context_games=5):
    """Adiciona features contextuais por liga/time/confronto para o mercado atual.

    O dataframe recebido é de um único mercado. Portanto, as features abaixo são
    específicas desse mercado, mas segmentadas por liga, mandante, visitante e confronto.

    Features geradas sem leakage:
    - ctx_liga_*: histórico do mercado naquela liga até T-1;
    - ctx_home_*: histórico do mercado com o mandante até T-1;
    - ctx_away_*: histórico do mercado com o visitante até T-1;
    - ctx_liga_home_* / ctx_liga_away_*: time dentro da liga;
    - ctx_matchup_*: confronto mandante x visitante, quando houver amostra;
    - ctx_*_roi: retorno unitário médio histórico usando odd real e target até T-1;
    - ctx_score: combinação defensiva entre probabilidade histórica e ROI contextual.
    """
    df = _sorted_copy(df.copy())
    if target_col not in df.columns:
        return df

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    odd = pd.to_numeric(df.get("odd", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["ctx_ret_unit"] = np.where(y.eq(1), odd - 1.0, -1.0)
    df["ctx_ret_unit"] = pd.to_numeric(df["ctx_ret_unit"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Probabilidade histórica do target por contexto
    for cols, prefix in [
        (["League_std"], "ctx_liga_prob"),
        (["Home"], "ctx_home_prob"),
        (["Away"], "ctx_away_prob"),
        (["League_std", "Home"], "ctx_liga_home_prob"),
        (["League_std", "Away"], "ctx_liga_away_prob"),
        (["Home", "Away"], "ctx_matchup_prob"),
    ]:
        df = _expanding_group_stat(df, cols, target_col, prefix, min_periods=1)

    # ROI histórico por contexto
    for cols, prefix in [
        (["League_std"], "ctx_liga_roi"),
        (["Home"], "ctx_home_roi"),
        (["Away"], "ctx_away_roi"),
        (["League_std", "Home"], "ctx_liga_home_roi"),
        (["League_std", "Away"], "ctx_liga_away_roi"),
        (["Home", "Away"], "ctx_matchup_roi"),
    ]:
        df = _expanding_group_stat(df, cols, "ctx_ret_unit", prefix, min_periods=1)

    # Backward-compatible aliases mais legíveis para o modelo/relatórios
    alias_map = {
        "ctx_liga_apostas": "ctx_liga_prob_count",
        "ctx_home_apostas": "ctx_home_prob_count",
        "ctx_away_apostas": "ctx_away_prob_count",
        "ctx_liga_prob_media": "ctx_liga_prob_mean",
        "ctx_home_prob_media": "ctx_home_prob_mean",
        "ctx_away_prob_media": "ctx_away_prob_mean",
        "ctx_liga_roi_media": "ctx_liga_roi_mean",
        "ctx_home_roi_media": "ctx_home_roi_mean",
        "ctx_away_roi_media": "ctx_away_roi_mean",
    }
    for new_col, old_col in alias_map.items():
        df[new_col] = pd.to_numeric(df.get(old_col, 0), errors="coerce").fillna(0.0)

    # Score contextual: usa somente contextos com amostra mínima; se não houver,
    # mantém neutro para não matar mercado novo.
    ctx_parts = []
    for prob_col, roi_col, count_col, peso in [
        ("ctx_liga_prob_mean", "ctx_liga_roi_mean", "ctx_liga_prob_count", 0.40),
        ("ctx_home_prob_mean", "ctx_home_roi_mean", "ctx_home_prob_count", 0.25),
        ("ctx_away_prob_mean", "ctx_away_roi_mean", "ctx_away_prob_count", 0.25),
        ("ctx_liga_home_prob_mean", "ctx_liga_home_roi_mean", "ctx_liga_home_prob_count", 0.05),
        ("ctx_liga_away_prob_mean", "ctx_liga_away_roi_mean", "ctx_liga_away_prob_count", 0.05),
    ]:
        count = pd.to_numeric(df.get(count_col, 0), errors="coerce").fillna(0)
        prob = pd.to_numeric(df.get(prob_col, 0.5), errors="coerce").fillna(0.5)
        roi = pd.to_numeric(df.get(roi_col, 0.0), errors="coerce").fillna(0.0)
        part = np.where(count >= min_context_games, (prob - 0.5) + roi, 0.0)
        ctx_parts.append(pd.Series(part, index=df.index) * float(peso))

    df["ctx_score"] = sum(ctx_parts).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ctx_amostra_total"] = (
        pd.to_numeric(df.get("ctx_liga_prob_count", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("ctx_home_prob_count", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("ctx_away_prob_count", 0), errors="coerce").fillna(0)
    ).astype(int)

    # Limpa auxiliares que não devem virar feature diretamente.
    df.drop(columns=["ctx_ret_unit"], inplace=True, errors="ignore")
    return df

# ==============================
# KELLY FRACIONÁRIO
# ==============================
def calcular_kelly_fractional(odd, probability, fraction=0.25):
    try:
        odd = float(odd)
        probability = float(probability)
        fraction = float(fraction)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(odd) or not math.isfinite(probability) or odd <= 1.0:
        return 0.0

    probability = min(max(probability, 0.0), 1.0)
    q = 1.0 - probability
    b = odd - 1.0
    kelly = (b * probability - q) / b
    stake_pct = max(0.0, min(1.0, kelly * max(0.0, fraction)))
    return stake_pct


# ==============================
# MÉTRICAS AVANÇADAS
# ==============================
def calcular_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    equity = pd.Series(equity_curve, dtype="float64").fillna(0)
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min()) if len(drawdown) else 0.0


def calcular_metricas_apostas(apostas, target_col, initial_bankroll=1000.0):
    empty = {
        "apostas": 0,
        "lucro": 0.0,
        "winrate": 0.0,
        "roi": 0.0,
        "profit_factor": 0.0,
        "drawdown_max": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "cagr": 0.0,
        "expectancy": 0.0,
        "odd_media": 0.0,
    }
    if apostas is None or apostas.empty:
        return empty

    apostas = apostas.copy()
    retornos = pd.to_numeric(apostas.get("retorno", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
    lucro = float(retornos.sum())
    total = int(len(apostas))
    wins = retornos[retornos > 0]
    losses = retornos[retornos < 0]

    equity = initial_bankroll + retornos.cumsum()
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    std = float(retornos.std(ddof=0)) if total > 1 else 0.0
    sharpe = float(retornos.mean() / std * np.sqrt(total)) if std > 0 else 0.0

    downside = retornos[retornos < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    sortino = float(retornos.mean() / downside_std * np.sqrt(total)) if downside_std > 0 else 0.0

    cagr = 0.0
    if "Date" in apostas.columns and total > 0:
        dates = pd.to_datetime(apostas["Date"], errors="coerce").dropna()
        if len(dates) >= 2:
            years = max((dates.max() - dates.min()).days / 365.25, 0)
            final_bankroll = float(equity.iloc[-1]) if len(equity) else initial_bankroll
            if years > 0 and initial_bankroll > 0 and final_bankroll > 0:
                cagr = (final_bankroll / initial_bankroll) ** (1 / years) - 1

    winrate = float(pd.to_numeric(apostas[target_col], errors="coerce").fillna(0).mean()) if total else 0.0
    odd_media = float(pd.to_numeric(apostas.get("odd", pd.Series(dtype="float64")), errors="coerce").mean()) if total else 0.0

    return {
        "apostas": total,
        "lucro": lucro,
        "winrate": winrate,
        "roi": lucro / total if total else 0.0,
        "profit_factor": profit_factor,
        "drawdown_max": calcular_drawdown(equity),
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": float(cagr),
        "expectancy": float(retornos.mean()) if total else 0.0,
        "odd_media": odd_media if math.isfinite(odd_media) else 0.0,
    }
