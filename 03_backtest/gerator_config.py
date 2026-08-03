import json
import os
import pandas as pd
from datetime import datetime

# ========================
# Caminhos
# ========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MARKET_FILE = os.path.join(os.path.dirname(BASE_DIR), "02_validation", "mercados_map.json")
DATA_DIR    = os.path.join(os.path.dirname(BASE_DIR), "data", "eventos")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# ========================
# Carrega mercados_map
# ========================
with open(MARKET_FILE, "r", encoding="utf-8") as f:
    mercado_map = json.load(f)

# ========================
# Mapeamento mercado → coluna de odd na base_oficial
# ========================
MAPA_ODDS = {
    "G_H_HT": "O_H_HT",
    "G_A_HT": "O_A_HT",
    "TG_HT" : "O_05_HT",
    "G_H_FT": "O_H_FT",
    "G_A_FT": "O_A_FT",
    "TG_FT" : "O_25_FT",
    "SH_H"  : None,
    "SH_A"  : None,
    "SOT_H" : None,
    "SOT_A" : None,
    "SOF_H" : None,
    "SOF_A" : None,
    "O_C_H" : "O_C_O75",
    "O_C_A" : "O_C_O75",
}

# ========================
# Mercados ativos
# ========================
MERCADOS_ATIVOS = {
    "G_H_HT": "Gols_H_HT",
    "G_A_HT": "Gols_A_HT",
    "TG_HT" : "Total_Gols_HT",
    "G_H_FT": "Gols_H_FT",
    "G_A_FT": "Gols_A_FT",
    "TG_FT" : "Total_Gols_FT",
    "C_H_FT": "Corners_H_FT",
    "C_A_FT": "Corners_A_FT",
    "TC_FT" : "Total_Corners_FT",
    "SH_H"  : "Shots_H",
    "SH_A"  : "Shots_A",
    "SOT_H" : "ShotsOnTarget_H",
    "SOT_A" : "ShotsOnTarget_A",
    "SOF_H" : "ShotsOffTarget_H",
    "SOF_A" : "ShotsOffTarget_A",
    "O_C_H" : "Corners_H",
    "O_C_A" : "Corners_A",
}

# ========================
# Configuração de cores (terminal)
# ========================
COLOR_MARKET = "\033[92m"  # verde
COLOR_LEAGUE = "\033[90m"  # cinza
COLOR_RESET = "\033[0m"

# ========================
# Calcula thresholds por liga e exibe log clean
# ========================
print("Calculando thresholds por mercado...\n")
thresholds_por_liga = {}

for mercado_key, event_name in MERCADOS_ATIVOS.items():
    file_path = os.path.join(DATA_DIR, f"{event_name}.csv")
    hora = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(file_path):
        print(f"[{hora}] ⚠ Arquivo não encontrado: {event_name}.csv")
        thresholds_por_liga[mercado_key] = {}
        continue

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if "League_std" not in df.columns or "value" not in df.columns:
        print(f"[{hora}] ⚠ Colunas ausentes em {event_name}.csv")
        thresholds_por_liga[mercado_key] = {}
        continue

    # Média por liga
    medias = df.groupby("League_std")["value"].mean().round(1).to_dict()
    thresholds_por_liga[mercado_key] = medias

    # Threshold médio do mercado
    threshold_avg = round(pd.Series(list(medias.values())).mean(), 1)
    print(f"[{hora}] {COLOR_MARKET}{mercado_key:<8} | {event_name:<25} | Threshold Médio: {threshold_avg}{COLOR_RESET}")

    # Mostra até 3 melhores ligas em cinza
    melhores = sorted(medias.items(), key=lambda x: x[1], reverse=True)[:3]
    for liga, valor in melhores:
        print(f"    [{hora}] {COLOR_LEAGUE}{liga}: {valor}{COLOR_RESET}")

# ========================
# Cria config.json
# ========================
config = {
    "events": [],
    "window": 5,
    "zscore_threshold": 1.5,
    "strategy": {
        "z_global": 1.5,
        "z_liga": 1.0,
        "z_team"  : 1.0
    }
}

for mercado_key, event_name in MERCADOS_ATIVOS.items():
    if mercado_key not in mercado_map:
        continue

    mercado_info = mercado_map[mercado_key]
    tipo        = mercado_info.get("type")
    if tipo != "quantidade":
        continue

    thresholds_liga = thresholds_por_liga.get(mercado_key, {})
    threshold_default = round(pd.Series(list(thresholds_liga.values())).mean(), 1) if thresholds_liga else 0.5
    odd_col = MAPA_ODDS.get(mercado_key, None)

    event = {
        "name"             : event_name,
        "source"           : mercado_key,
        "threshold_default": threshold_default,
        "thresholds_liga"  : thresholds_liga,
        "odd_col"          : odd_col,
        "odd_default"      : 1.50,
        "type"             : tipo,
        "active"           : True
    }
    config["events"].append(event)

# Salva config.json
with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\nConfig.json gerado com {len(config['events'])} eventos! Arquivo salvo em: {CONFIG_FILE}")