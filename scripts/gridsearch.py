# gridsearch.py
import itertools, json, pandas as pd
import os
from pathlib import Path
import sys, pathlib
from dataclasses import asdict
from tqdm import tqdm
import math
# Ensure the `scripts` directory (where this file and backtesting.py live)
# is on the Python import path so that `import backtesting` works regardless
# of the working directory used to execute the script.
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from backtesting import (
    load_data,
    run_backtest,
    run_walkforward_backtest,
    calculate_performance_metrics,
    StrategyParameters,
)

DATA_FILE = Path("/Users/elgask0/REPOS/algo_meme/data/spread_data.csv")
params = StrategyParameters()
df = load_data(DATA_FILE, params)

# Ajuste dinámico del walk-forward según tamaño de datos (~5-min freq)
# Determinar frecuencia en minutos; si no está definido, asumir 5
try:
    freq_minutes = df.index.freq.delta.total_seconds() / 60
except Exception:
    freq_minutes = 5
total_days = len(df) * freq_minutes / (60 * 24)
walk_train_days = int(total_days * 0.6)
walk_test_days = int(total_days * 0.2)
print(f"Walk-forward settings -> train: {walk_train_days} days, test: {walk_test_days} days (total {total_days:.1f} days)")

 # --- define grid ---------------------------------------------------------
param_grid = {
    # --- Parámetros Temporales y de Frecuencia (Variación Principal) ---
    "RESAMPLE_ALL_DATA_TO_MINUTES": [15, 60, 120, 240], # De más rápido a bastante más lento que tu base de 15 min
    "OLS_WINDOW_DAYS": [3, 7, 15, 30], # Más corta, tu base, más larga, y mucho más larga
    "ZSCORE_WINDOW_DAYS": [3, 7, 15, 30], # Más corta, tu base, más larga, y mucho más larga

    # --- Parámetros de Señal y Filtro (Variación Secundaria) ---
    "ENTRY_ZSCORE": [1.5, 1.75, 2.0, 2.25], # Alrededor de tu base y explorando entradas más exigentes
    "MIN_OLS_R_SQUARED_THRESHOLD": [0.6], # Tu base y un valor más laxo que también funcionó bien

    # --- Parámetros Fijos (Basados en tu Configuración "Bastante Buena") ---
    "EXIT_ZSCORE": [0.25, 0.5, 0.75],
    "STOP_LOSS_ZSCORE": [4.0],
    "POSITION_SIZE_PCT": [0.5], # Usando el 0.5 que indicaste como bueno
    "PNL_STOP_LOSS_PCT": [0.1],
    "BETA_ROLL_STD_MAX": [0.25],
    "TRAIL_STOP_PCT": [1.0],
    "SL_COOLDOWN_HOURS": [8]
}
default_params = StrategyParameters()
grid_keys = list(param_grid.keys())
total_combos = math.prod(len(param_grid[key]) for key in grid_keys)

results = []
OUTPUT_FILE = "gridsearch_results.csv"
# If we are starting a fresh run, remove any prior partial file so the header is written cleanly.
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

try:
    for combo in tqdm(itertools.product(*(param_grid[key] for key in grid_keys)), total=total_combos, desc="Grid search"):
        grid_kwargs = dict(zip(grid_keys, combo))
        # Skip invalid combinations: Z‑score window cannot exceed OLS window
        if grid_kwargs["ZSCORE_WINDOW_DAYS"] > grid_kwargs["OLS_WINDOW_DAYS"]:
            continue
        params_dict = asdict(default_params)
        params_dict.update(grid_kwargs)

        # Run walk‑forward backtest
        equity, trades = run_walkforward_backtest(
            df, walk_train_days, walk_test_days, **params_dict
        )
        # Compute performance metrics
        perf = calculate_performance_metrics(
            equity,
            trades,
            initial_capital=params_dict.get("INITIAL_CAPITAL"),
            risk_free_rate_annual=params_dict.get("RISK_FREE_RATE_ANNUAL"),
        )
        # Merge param settings with performance outputs
        result = {**params_dict, **perf}
        results.append(result)

        # Persist this single result immediately.
        pd.DataFrame([result]).to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0,
        )

        print(
            f"✓ {params_dict} -> Final Equity: {perf.get('Final Equity', float('nan')):.2f}, "
            f"Sharpe: {perf.get('Sharpe Ratio (Annualized, Daily Ret)', float('nan')):.2f}, "
            f"Trades: {perf.get('Number of Trades', 0)}"
        )
finally:
    # Guarantee that everything accumulated so far is saved even if the run is interrupted.
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nResultados parciales guardados en {OUTPUT_FILE} (total filas {len(results)}).")