#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint
import time
from tqdm import tqdm # <--- tqdm IMPORTADO DE NUEVO

# --- Configuration ---
LOG_LEVEL = logging.WARNING 

# --- Database and Symbols ---
DB_PATH = "trading_data.db" 
PAIRS_TO_ANALYZE = [
    ("MEXCFTS_PERP_GIGA_USDT", "MEXCFTS_PERP_SPX_USDT"),
]

USE_LOG_PRICES = True
PRICE_DATA_START_DATE = "2024-11-15" 

# Ventanas en DÍAS para RollingOLS y métricas asociadas al spread dinámico
OLS_WINDOW_DAYS_CONFIG = [7] 

# Ventanas en DÍAS para Rolling Cointegration sobre los precios originales
ROLLING_COINT_PRICE_WINDOW_DAYS_CONFIG = [30]

# Ventanas en DÍAS para Correlación Rodante de log-precios
CORR_WINDOW_DAYS_CONFIG = [15]

# Umbral de p-value para considerar una ventana rodante de cointegración como "pasada"
ROLLING_COINT_PVALUE_THRESHOLD = 0.10 

CANDLES_PER_DAY_5MIN = 24 * (60 // 5)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d][%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_db_connection(db_path=DB_PATH):
    if not os.path.exists(db_path):
        logger.critical(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(db_path)

def load_and_prepare_data(conn, symbol_y, symbol_x, start_date_filter):
    logger.info(f"Loading data for pair: {symbol_y} / {symbol_x}")
    try:
        query = "SELECT ts_start, symbol_id, mark_price FROM mark_price_vwap WHERE symbol_id IN (?, ?)"
        df = pd.read_sql(query, conn, params=[symbol_y, symbol_x], parse_dates=["ts_start"])
        
        if df.empty:
            logger.warning(f"No data found for one or both symbols: {symbol_y}, {symbol_x}")
            return pd.DataFrame()

        df_pivot = df.pivot(index="ts_start", columns="symbol_id", values="mark_price")
        df_pivot = df_pivot.rename(columns={symbol_y: "Y", symbol_x: "X"})
        
        if "Y" not in df_pivot.columns or "X" not in df_pivot.columns:
            logger.warning(f"Could not pivot data correctly for {symbol_y}/{symbol_x}. Columns found: {df_pivot.columns.tolist()}")
            return pd.DataFrame()

        df_pivot = df_pivot[["Y", "X"]].dropna() 

        if start_date_filter:
            df_pivot = df_pivot[df_pivot.index >= start_date_filter]
        
        all_windows_days = OLS_WINDOW_DAYS_CONFIG + ROLLING_COINT_PRICE_WINDOW_DAYS_CONFIG + CORR_WINDOW_DAYS_CONFIG
        min_data_needed = 0
        if all_windows_days:
            min_data_needed = max(all_windows_days) * CANDLES_PER_DAY_5MIN
        
        if df_pivot.empty or len(df_pivot) < min_data_needed: 
             logger.warning(f"Not enough common data for {symbol_y}/{symbol_x} for all configured windows (available: {len(df_pivot)}, need at least {min_data_needed}).")
             return pd.DataFrame()

        logger.info(f"Loaded {len(df_pivot)} common data points for {symbol_y}/{symbol_x} from {df_pivot.index.min()} to {df_pivot.index.max()}.")
        
        if USE_LOG_PRICES:
            df_pivot = df_pivot[(df_pivot["Y"] > 0) & (df_pivot["X"] > 0)]
            if df_pivot.empty:
                logger.warning(f"No positive price data for log transformation for {symbol_y}/{symbol_x}.")
                return pd.DataFrame()
            df_pivot = np.log(df_pivot)
            logger.info("Applied log transformation.")
        
        return df_pivot
    except Exception as e:
        logger.error(f"Error loading/preparing data for {symbol_y}/{symbol_x}: {e}")
        return pd.DataFrame()

def test_rolling_engle_granger_on_prices(prices_y: pd.Series, prices_x: pd.Series, window_candles: int, p_value_threshold: float) -> tuple:
    logger.warning(f"    Starting Rolling Cointegration on prices for window {window_candles} candles (this will be slow)...")
    start_time_rc = time.time()
    if len(prices_y) < window_candles:
        logger.warning(f"      Not enough data for rolling cointegration with window {window_candles}.")
        return np.nan, np.nan, np.nan, 0

    p_values_list = []
    min_obs = int(window_candles * 0.9) 

    # Wrap the loop with tqdm
    for i in tqdm(range(window_candles - 1, len(prices_y)), desc=f"RollingCoint W{window_candles//CANDLES_PER_DAY_5MIN}d", unit="candle", leave=False, ncols=100):
        window_y = prices_y.iloc[i - window_candles + 1 : i + 1]
        window_x = prices_x.iloc[i - window_candles + 1 : i + 1]

        if len(window_y.dropna()) < min_obs or len(window_x.dropna()) < min_obs:
            p_values_list.append(np.nan)
            continue
        try:
            _, p_val, _ = coint(window_y, window_x, trend='c', autolag='AIC')
            p_values_list.append(p_val)
        except Exception: 
            p_values_list.append(np.nan)
    
    end_time_rc = time.time()
    logger.warning(f"    Finished Rolling Cointegration for window {window_candles} in {end_time_rc - start_time_rc:.2f}s.")

    if not p_values_list:
        return 0.0, np.nan, np.nan, 0

    p_values_series = pd.Series(p_values_list).dropna()
    if p_values_series.empty:
        return 0.0, np.nan, np.nan, 0
        
    pct_windows_coint = (p_values_series < p_value_threshold).sum() / len(p_values_series)
    return pct_windows_coint, p_values_series.mean(), p_values_series.median(), len(p_values_series)

# --- Main Analysis Loop ---
def main():
    overall_start_time = time.time()
    conn = get_db_connection()

    logger.warning(f"Starting Simplified Pair Validation Script. DB: {DB_PATH}")
    logger.warning(f"Pairs to analyze: {PAIRS_TO_ANALYZE}")
    logger.warning(f"Using Log Prices: {USE_LOG_PRICES}, Start Date Filter: '{PRICE_DATA_START_DATE}'")
    logger.warning(f"OLS Window Config (days): {OLS_WINDOW_DAYS_CONFIG}")
    logger.warning(f"Rolling Cointegration (Prices) Window Config (days): {ROLLING_COINT_PRICE_WINDOW_DAYS_CONFIG}")
    logger.warning(f"Correlation Window Config (days): {CORR_WINDOW_DAYS_CONFIG}")

    for symbol_y, symbol_x in PAIRS_TO_ANALYZE:
        pair_desc = f"{symbol_y} / {symbol_x}"
        logger.warning(f"\n--- Analyzing Pair: {pair_desc} ---") 
        
        df_pair_prices = load_and_prepare_data(conn, symbol_y, symbol_x, PRICE_DATA_START_DATE)

        if df_pair_prices.empty:
            logger.warning(f"Skipping analysis for {pair_desc} due to data loading issues.")
            continue

        static_corr = df_pair_prices['Y'].corr(df_pair_prices['X'])
        print(f"\nPair: {pair_desc}")
        print(f"  Static Correlation ({'Log-' if USE_LOG_PRICES else ''}Prices): {static_corr:.4f}")

        if CORR_WINDOW_DAYS_CONFIG:
            print("  Rolling Correlation of Prices:")
            for days in CORR_WINDOW_DAYS_CONFIG:
                window_candles = days * CANDLES_PER_DAY_5MIN
                if len(df_pair_prices) >= window_candles:
                    rolling_corr = df_pair_prices['Y'].rolling(window=window_candles, min_periods=int(window_candles*0.8)).corr(df_pair_prices['X'])
                    print(f"    {days}d Window: Mean={rolling_corr.mean():.4f}, Median={rolling_corr.median():.4f}, Std={rolling_corr.std():.4f}")
                else:
                    print(f"    {days}d Window: Not enough data.")
        
        print(f"\n  Rolling Cointegration of {'Log-' if USE_LOG_PRICES else ''}Prices (Threshold p < {ROLLING_COINT_PVALUE_THRESHOLD}):")
        rolling_coint_summary_list = []
        for days_coint in ROLLING_COINT_PRICE_WINDOW_DAYS_CONFIG:
            window_candles_coint = days_coint * CANDLES_PER_DAY_5MIN
            pct_pass, mean_pval, median_pval, num_windows = test_rolling_engle_granger_on_prices(
                df_pair_prices["Y"], df_pair_prices["X"], window_candles_coint, ROLLING_COINT_PVALUE_THRESHOLD
            )
            rolling_coint_summary_list.append({
                "Rolling Coint. Window (Days)": days_coint,
                "Pct Passing Windows": pct_pass,
                "Mean PVal": mean_pval,
                "Median PVal": median_pval,
                "Num Valid Windows": num_windows
            })
        if rolling_coint_summary_list:
            rolling_coint_df = pd.DataFrame(rolling_coint_summary_list).set_index("Rolling Coint. Window (Days)")
            print(f"{rolling_coint_df.to_string(float_format='%.4f')}")

        pair_ols_results_summary = []
        print(f"\n  Rolling OLS and Dynamic Spread ADF Analysis:")
        for ols_window_d in OLS_WINDOW_DAYS_CONFIG: 
            logger.warning(f"    --- OLS Window: {ols_window_d} days ---") 
            ols_window_candles = ols_window_d * CANDLES_PER_DAY_5MIN

            if len(df_pair_prices) < ols_window_candles:
                logger.warning(f"      Not enough data for {ols_window_d}d RollingOLS. Skipping.")
                continue

            exog_for_rols = sm.add_constant(df_pair_prices["X"])
            rols_model = RollingOLS(endog=df_pair_prices["Y"], exog=exog_for_rols, window=ols_window_candles, min_nobs=int(ols_window_candles * 0.9))
            rols_results = rols_model.fit()

            beta_t = rols_results.params.get("X", pd.Series(dtype=float)).dropna()
            alpha_t = rols_results.params.get("const", pd.Series(dtype=float)).dropna()
            r_squared_t = rols_results.rsquared.dropna() if hasattr(rols_results, 'rsquared') and rols_results.rsquared is not None else pd.Series(dtype=float)
            
            if beta_t.empty or alpha_t.empty:
                logger.warning(f"      RollingOLS failed for {ols_window_d}d window.")
                continue

            common_params_idx = beta_t.index.intersection(alpha_t.index)
            if common_params_idx.empty:
                logger.warning(f"      No common index for beta_t and alpha_t for {ols_window_d}d window.")
                continue
            
            beta_t_aligned = beta_t.loc[common_params_idx]
            alpha_t_aligned = alpha_t.loc[common_params_idx]
            prices_y_aligned = df_pair_prices.loc[common_params_idx, "Y"]
            prices_x_aligned = df_pair_prices.loc[common_params_idx, "X"]
            dynamic_spread = prices_y_aligned - (alpha_t_aligned + beta_t_aligned * prices_x_aligned)
            dynamic_spread = dynamic_spread.dropna()

            if dynamic_spread.empty:
                logger.warning(f"      Dynamic spread empty for {ols_window_d}d window.")
                pair_ols_results_summary.append({
                    "OLS Window (Days)": ols_window_d, "Beta Mean": np.nan, "R² Mean": np.nan,
                    "ADF Stat (Dyn.Spread)": np.nan, "ADF p-value (Dyn.Spread)": np.nan,
                    "Spread Points": 0
                })
                continue

            beta_mean = beta_t.mean()
            r_squared_mean = r_squared_t.mean() if not r_squared_t.empty else np.nan
            
            adf_stat, adf_pvalue = np.nan, np.nan
            if len(dynamic_spread) >= 20:
                 try:
                    adf_results = adfuller(dynamic_spread, regression="n") 
                    adf_stat, adf_pvalue = adf_results[0], adf_results[1]
                 except Exception as e_adf:
                    logger.error(f"      ADF test failed for {ols_window_d}d spread: {e_adf}")
            
            pair_ols_results_summary.append({
                "OLS Window (Days)": ols_window_d, "Beta Mean": beta_mean, "R² Mean": r_squared_mean,
                "ADF Stat (Dyn.Spread)": adf_stat, "ADF p-value (Dyn.Spread)": adf_pvalue,
                "Spread Points": len(dynamic_spread)
            })

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False) 
            fig.suptitle(f"{pair_desc} - OLS Window: {ols_window_d} days ({ols_window_candles} candles)", fontsize=16)
            if not r_squared_t.empty:
                r_squared_t.plot(ax=axes[0], title="Rolling OLS R²(t)", lw=1.5)
                axes[0].axhline(r_squared_mean, color='r', linestyle='--', lw=1, label=f"Mean R²: {r_squared_mean:.3f}")
            else: axes[0].set_title("Rolling OLS R²(t) - No data")
            axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.7)
            dynamic_spread.plot(ax=axes[1], title="Dynamic Spread ε(t) = Y - (α(t) + β(t) * X)", lw=1.5)
            axes[1].axhline(0, color='k', linestyle='-', lw=1); axes[1].grid(True, linestyle=':', alpha=0.7)
            if not dynamic_spread.empty:
                dynamic_spread.hist(bins=100, ax=axes[2], density=True, alpha=0.7)
                dynamic_spread.plot(kind='kde', ax=axes[2], color='black', lw=1.5, label="KDE")
            axes[2].set_title("Histogram of Dynamic Spread ε(t)"); axes[2].set_xlabel("Spread Value"); axes[2].set_ylabel("Density")
            axes[2].legend(); axes[2].grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plot_filename = f"{symbol_y.replace('/','_')}_{symbol_x.replace('/','_')}_OLS_{ols_window_d}d.png"
            plots_dir = "validation_plots"
            if not os.path.exists(plots_dir): os.makedirs(plots_dir)
            full_plot_path = os.path.join(plots_dir, plot_filename)
            try: plt.savefig(full_plot_path); 
            except Exception as e: logger.error(f"    Could not save plot {full_plot_path}: {e}")
            plt.close(fig) 

        if pair_ols_results_summary:
            summary_df_ols = pd.DataFrame(pair_ols_results_summary).set_index("OLS Window (Days)")
            def p_value_formatter(x):
                if pd.isna(x): return "NaN"
                return f"{x:.4e}" if abs(x) < 0.0001 and x != 0 else f"{x:.4f}" 
            formatters = {
                'Beta Mean': '{:.4f}'.format, 'R² Mean': '{:.4f}'.format,
                'ADF Stat (Dyn.Spread)': '{:.4f}'.format, 'ADF p-value (Dyn.Spread)': p_value_formatter,
                'Spread Points': '{:.0f}'.format
            }
            print(f"\nSummary Table for Dynamic Spread (from RollingOLS) for Pair: {pair_desc}\n{summary_df_ols.to_string(formatters=formatters)}")
        else:
            logger.warning(f"\nNo OLS window results to summarize for {pair_desc}.")

    conn.close()
    overall_end_time = time.time()
    logger.warning(f"\n\nSimplified Pair Validation Script Finished in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
