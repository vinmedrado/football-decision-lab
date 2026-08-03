import numpy as np
import pandas as pd


def _sort(df):
    if "Date" in df.columns:
        return df.sort_values("Date").reset_index(drop=True)
    return df.reset_index(drop=True)


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index)


def add_strength_features(df):
    df = df.copy()

    ppg_h = _num(df, "PPG_H_Pre", _num(df, "PPG_H", 0.0))
    ppg_a = _num(df, "PPG_A_Pre", _num(df, "PPG_A", 0.0))
    xg_h = _num(df, "XG_H_Pre", 0.0)
    xg_a = _num(df, "XG_A_Pre", 0.0)
    xg_t = _num(df, "XG_T_Pre", xg_h + xg_a)

    df["ppg_h_pre"] = ppg_h
    df["ppg_a_pre"] = ppg_a
    df["ppg_diff"] = ppg_h - ppg_a
    df["ppg_sum"] = ppg_h + ppg_a
    df["ppg_ratio"] = ppg_h / ppg_a.replace(0, np.nan)

    df["xg_h_pre"] = xg_h
    df["xg_a_pre"] = xg_a
    df["xg_t_pre"] = xg_t
    df["xg_diff"] = xg_h - xg_a
    df["xg_sum"] = xg_h + xg_a
    df["xg_ratio"] = xg_h / xg_a.replace(0, np.nan)

    df["ppg_ratio"] = df["ppg_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["xg_ratio"] = df["xg_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return df


def add_liga_pre_features(df, windows=(5, 10, 20)):
    df = _sort(df.copy())

    if "League_std" not in df.columns or "value" not in df.columns:
        return df

    value = pd.to_numeric(df["value"], errors="coerce")

    for w in windows:
        df[f"liga_value_ma_{w}"] = (
            value.groupby(df["League_std"])
            .transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
        )
        df[f"liga_value_std_{w}"] = (
            value.groupby(df["League_std"])
            .transform(lambda s: s.shift(1).rolling(w, min_periods=3).std())
        )

    df["liga_value_exp_mean"] = (
        value.groupby(df["League_std"])
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )

    df["liga_value_exp_std"] = (
        value.groupby(df["League_std"])
        .transform(lambda s: s.shift(1).expanding(min_periods=3).std())
    )

    return df


def add_team_pre_features(df, windows=(3, 5, 10)):
    df = _sort(df.copy())

    if "value" not in df.columns:
        return df

    value = pd.to_numeric(df["value"], errors="coerce")

    for team_col in ["Home", "Away"]:
        if team_col not in df.columns:
            continue

        for w in windows:
            df[f"{team_col.lower()}_value_ma_{w}"] = (
                value.groupby(df[team_col])
                .transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
            )
            df[f"{team_col.lower()}_value_std_{w}"] = (
                value.groupby(df[team_col])
                .transform(lambda s: s.shift(1).rolling(w, min_periods=2).std())
            )

        df[f"{team_col.lower()}_value_exp_mean"] = (
            value.groupby(df[team_col])
            .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
        )

    return df


def add_diff_features(df):
    df = df.copy()

    for w in [3, 5, 10]:
        h = f"home_value_ma_{w}"
        a = f"away_value_ma_{w}"
        if h in df.columns and a in df.columns:
            df[f"home_away_ma_diff_{w}"] = df[h] - df[a]
            df[f"home_away_ma_sum_{w}"] = df[h] + df[a]

    if "liga_value_exp_mean" in df.columns:
        for col in ["home_value_exp_mean", "away_value_exp_mean"]:
            if col in df.columns:
                df[f"{col}_vs_liga"] = df[col] - df["liga_value_exp_mean"]

    return df



def add_market_context_features(df, windows=(5, 10, 20)):
    """Features contextuais por liga/time sem leakage.

    Espera que o dataframe represente um único mercado e possua `value` como
    variável histórica do mercado. Todas as estatísticas usam shift(1).
    """
    df = _sort(df.copy())
    if "value" not in df.columns:
        return df

    value = pd.to_numeric(df["value"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    groups = []
    if "League_std" in df.columns:
        groups.append(("ctx_liga", [df["League_std"].astype(str)]))
    if "Home" in df.columns:
        groups.append(("ctx_home", [df["Home"].astype(str)]))
    if "Away" in df.columns:
        groups.append(("ctx_away", [df["Away"].astype(str)]))
    if "League_std" in df.columns and "Home" in df.columns:
        groups.append(("ctx_liga_home", [df["League_std"].astype(str), df["Home"].astype(str)]))
    if "League_std" in df.columns and "Away" in df.columns:
        groups.append(("ctx_liga_away", [df["League_std"].astype(str), df["Away"].astype(str)]))
    if "Home" in df.columns and "Away" in df.columns:
        groups.append(("ctx_matchup", [df["Home"].astype(str), df["Away"].astype(str)]))

    for prefix, key in groups:
        prev_count = value.groupby(key, group_keys=False).transform(lambda s: s.shift(1).expanding(min_periods=1).count())
        prev_mean = value.groupby(key, group_keys=False).transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
        prev_std = value.groupby(key, group_keys=False).transform(lambda s: s.shift(1).expanding(min_periods=3).std())
        df[f"{prefix}_value_count"] = prev_count.fillna(0).astype(int)
        df[f"{prefix}_value_mean"] = prev_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[f"{prefix}_value_std"] = prev_std.replace([np.inf, -np.inf], np.nan).fillna(0.0001)
        for w in windows:
            df[f"{prefix}_value_ma_{w}"] = value.groupby(key, group_keys=False).transform(
                lambda s, ww=w: s.shift(1).rolling(ww, min_periods=2).mean()
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "ctx_liga_value_mean" in df.columns:
        for col in ["ctx_home_value_mean", "ctx_away_value_mean"]:
            if col in df.columns:
                df[f"{col}_vs_liga"] = df[col] - df["ctx_liga_value_mean"]

    return df


def clean_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df


def apply_all_features(df, window=5):
    df = add_strength_features(df)
    df = add_liga_pre_features(df)
    df = add_team_pre_features(df)
    df = add_diff_features(df)
    df = add_market_context_features(df)
    df = clean_features(df)
    return df