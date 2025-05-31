import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import logging
import os
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import Optional

# --- Configuración General ---
DATA_FILEPATH = "/Users/elgask0/REPOS/algo_meme/data/spread_data.csv"
LOG_DIR = "/Users/elgask0/REPOS/algo_meme/logs"
PLOT_DIR = "/Users/elgask0/REPOS/algo_meme/plots"
LOG_FILE = os.path.join(LOG_DIR, "backtest_run.log")
FUNDING_FILEPATH = "/Users/elgask0/REPOS/algo_meme/data/funding_rate_history.csv"
TRADES_CSV_FILE = os.path.join(LOG_DIR, "trades_summary.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)


# Dataclass to collect all tunable strategy inputs in one place
@dataclass
class StrategyParameters:
    ASSET_Y_COL: str = "MEXCFTS_PERP_SPX_USDT"
    ASSET_X_COL: str = "MEXCFTS_PERP_GIGA_USDT"
    OLS_WINDOW_DAYS: int = 30
    ZSCORE_WINDOW_DAYS: int = 15
    BETA_P_VALUE_THRESHOLD: float = 0.05
    MIN_OLS_R_SQUARED_THRESHOLD: float = 0.6
    ENTRY_ZSCORE: float = 1.75
    EXIT_ZSCORE: float = 0.25
    STOP_LOSS_ZSCORE: float = 4.0
    POSITION_SIZE_PCT: float = 0.5
    MIN_NOMINAL_PER_LEG: float = 5.0
    TAKER_FEE_PCT: float = 0.0002
    SLIPPAGE_PCT: float = 0.01
    SL_COOLDOWN_HOURS: int = 2
    PNL_STOP_LOSS_PCT: float = 0.1
    INITIAL_CAPITAL: float = 2000
    RISK_FREE_RATE_ANNUAL: float = 0.02
    RESAMPLE_ALL_DATA_TO_MINUTES: Optional[int] = 15
    DEFAULT_DATA_FREQ_MINUTES: int = 5
    BETA_MIN_THRESHOLD: float = 0.3
    BETA_MAX_THRESHOLD: float = 3.0
    BETA_ROLL_STD_MAX: float = 0.25
    TRAIL_STOP_PCT: float = 1


# Expose default strategy parameters as module-level globals for IDE/linting
_default_params = StrategyParameters()
ASSET_Y_COL = _default_params.ASSET_Y_COL
ASSET_X_COL = _default_params.ASSET_X_COL
OLS_WINDOW_DAYS = _default_params.OLS_WINDOW_DAYS
ZSCORE_WINDOW_DAYS = _default_params.ZSCORE_WINDOW_DAYS
BETA_P_VALUE_THRESHOLD = _default_params.BETA_P_VALUE_THRESHOLD
MIN_OLS_R_SQUARED_THRESHOLD = _default_params.MIN_OLS_R_SQUARED_THRESHOLD
ENTRY_ZSCORE = _default_params.ENTRY_ZSCORE
EXIT_ZSCORE = _default_params.EXIT_ZSCORE
STOP_LOSS_ZSCORE = _default_params.STOP_LOSS_ZSCORE
POSITION_SIZE_PCT = _default_params.POSITION_SIZE_PCT
MIN_NOMINAL_PER_LEG = _default_params.MIN_NOMINAL_PER_LEG
TAKER_FEE_PCT = _default_params.TAKER_FEE_PCT
SLIPPAGE_PCT = _default_params.SLIPPAGE_PCT
SL_COOLDOWN_HOURS = _default_params.SL_COOLDOWN_HOURS
PNL_STOP_LOSS_PCT = _default_params.PNL_STOP_LOSS_PCT
INITIAL_CAPITAL = _default_params.INITIAL_CAPITAL
RISK_FREE_RATE_ANNUAL = _default_params.RISK_FREE_RATE_ANNUAL
RESAMPLE_ALL_DATA_TO_MINUTES = _default_params.RESAMPLE_ALL_DATA_TO_MINUTES
DEFAULT_DATA_FREQ_MINUTES = _default_params.DEFAULT_DATA_FREQ_MINUTES
BETA_MIN_THRESHOLD = _default_params.BETA_MIN_THRESHOLD
BETA_MAX_THRESHOLD = _default_params.BETA_MAX_THRESHOLD
BETA_ROLL_STD_MAX = _default_params.BETA_ROLL_STD_MAX
TRAIL_STOP_PCT = _default_params.TRAIL_STOP_PCT


# --- OOP Wrapper for Backtesting ---
class PairTradingBacktester:
    def __init__(self, params: StrategyParameters = None):
        if params is None:
            self.params = StrategyParameters()
        else:
            self.params = params

    def run(self, df_input):
        # Pass all params as keyword arguments to run_backtest
        # Use asdict to extract all fields from the dataclass
        param_dict = asdict(self.params)
        return run_backtest(df_input, **param_dict)


# --- Funciones Auxiliares ---
def load_data(filepath, params: StrategyParameters):
    logging.info(f"Cargando datos desde: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df["ts_start"] = pd.to_datetime(df["ts_start"])
        df.set_index("ts_start", inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=[params.ASSET_Y_COL, params.ASSET_X_COL], inplace=True)
        if df.empty:
            logging.error("No hay datos después de eliminar NaNs iniciales.")
            return None
        logging.info(
            f"Datos cargados. Rango original: {df.index.min()} a {df.index.max()}. Filas: {len(df)}"
        )
        # --- Añadir funding rates ---
        try:
            fr = pd.read_csv(FUNDING_FILEPATH, parse_dates=["ts"])
            fr.set_index("ts", inplace=True)
            fr_y = fr[fr["symbol"] == params.ASSET_Y_COL][["funding_rate"]].rename(
                columns={"funding_rate": "funding_rate_y"}
            )
            fr_x = fr[fr["symbol"] == params.ASSET_X_COL][["funding_rate"]].rename(
                columns={"funding_rate": "funding_rate_x"}
            )
            # Unir, rellenar con 0 si falta dato
            df = (
                df.join(fr_y, how="left")
                .join(fr_x, how="left")
                .fillna({"funding_rate_y": 0.0, "funding_rate_x": 0.0})
            )
        except FileNotFoundError:
            logging.warning("Archivo de funding no encontrado, se ignora.")
        except Exception as e_f:
            logging.warning(f"Error leyendo funding: {e_f}")
        return df
    except FileNotFoundError:
        logging.error(f"Archivo de datos no encontrado: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error cargando datos: {e}")
        return None


def calculate_rolling_ols_and_spread(df, asset_y_col, asset_x_col, window_periods):
    """
    Rolling OLS based on statsmodels.RollingOLS (2.5) with
    rolling in-sample price normalisation (mean 0, std 1 for each window)
    and beta‑quality flags (2.1).
    Adds the columns:
        alpha, beta, beta_p_value, ols_r_squared, spread,
        beta_roll_std_20, beta_ok
    """
    import pandas as pd
    import numpy as np

    # --- Log‑prices ---
    log_y = np.log(df[asset_y_col])
    log_x = np.log(df[asset_x_col])

    # --- Rolling Normalisation (mean‑0, std‑1) per window ---
    roll_mean_y = log_y.rolling(window_periods, min_periods=window_periods).mean()
    roll_std_y = log_y.rolling(window_periods, min_periods=window_periods).std(ddof=0)
    log_y_std = (log_y - roll_mean_y) / roll_std_y

    roll_mean_x = log_x.rolling(window_periods, min_periods=window_periods).mean()
    roll_std_x = log_x.rolling(window_periods, min_periods=window_periods).std(ddof=0)
    log_x_std = (log_x - roll_mean_x) / roll_std_x

    # Exogenous matrix with constant.
    x_name = f"{asset_x_col}_std"
    exog = pd.DataFrame({x_name: log_x_std})
    exog = sm.add_constant(exog, prepend=False)  # columns: x_name, const

    # --- Rolling OLS (2.5) ---
    rols = RollingOLS(log_y_std, exog, window=window_periods)
    rres = rols.fit()

    # Align outputs to the original index
    params = rres.params.copy()
    params.index = df.index

    # pvalues and r_squared from RollingResults (arrays)
    pvalues = pd.DataFrame(rres.pvalues, index=df.index, columns=params.columns)
    r_squared = pd.Series(rres.rsquared, index=df.index)

    df["alpha"] = params["const"]
    df["beta"] = params[x_name]
    df["beta_p_value"] = pvalues[x_name]
    df["ols_r_squared"] = r_squared

    # Spread (residual of the regression on rolling-normalised space)
    df["spread"] = log_y_std - (df["alpha"] + df["beta"] * log_x_std)

    # --- Beta quality quick filters (2.1) ---
    df["beta_roll_std_20"] = df["beta"].rolling(20, min_periods=20).std()
    df["beta_ok"] = df["beta"].abs().between(BETA_MIN_THRESHOLD, BETA_MAX_THRESHOLD) & (
        df["beta_roll_std_20"] < BETA_ROLL_STD_MAX
    )

    return df


def calculate_zscore(series, window_periods):
    if window_periods < 1:
        logging.error(f"Z-score window_periods es {window_periods}, debe ser >= 1.")
        return pd.Series(np.nan, index=series.index)
    mean = series.rolling(window=window_periods, min_periods=window_periods).mean()
    std = series.rolling(window=window_periods, min_periods=window_periods).std()
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def apply_slippage(price, side, slippage_pct):
    if side == "buy":
        return price * (1 + slippage_pct)
    elif side == "sell":
        return price * (1 - slippage_pct)
    return price


# --- Lógica Principal del Backtest ---
def run_backtest(df_input, **params):
    # Allow external scripts to override any constant simply by passing
    # keyword arguments, e.g. run_backtest(df, ENTRY_ZSCORE=2.0).
    # The override is done once per call so a grid‑search can explore many
    # combinations safely.
    globals().update(params)
    logging.info("Iniciando backtest...")
    # --- Log strategy parameters for traceability ---
    try:
        # Gather all fields from StrategyParameters dataclass
        param_keys = list(StrategyParameters.__annotations__.keys())
        current_params = {key: globals().get(key) for key in param_keys}
        logging.info(f"Estrategia parámetros: {current_params}")
    except Exception as e:
        logging.warning(f"No se pudieron registrar los parámetros de estrategia: {e}")
    df = df_input.copy()

    current_data_freq_minutes = DEFAULT_DATA_FREQ_MINUTES
    if RESAMPLE_ALL_DATA_TO_MINUTES is not None and RESAMPLE_ALL_DATA_TO_MINUTES > 0:
        logging.info(
            f"Resampleando todos los datos a velas de {RESAMPLE_ALL_DATA_TO_MINUTES} minutos."
        )
        df = df.resample(f"{RESAMPLE_ALL_DATA_TO_MINUTES}min").last()
        df.dropna(subset=[ASSET_Y_COL, ASSET_X_COL], inplace=True)
        if df.empty:
            logging.error("No hay datos después del resampleo general.")
            return df, None, None
        current_data_freq_minutes = RESAMPLE_ALL_DATA_TO_MINUTES
        logging.info(
            f"Datos resampleados. Nuevo rango: {df.index.min()} a {df.index.max()}. Filas: {len(df)}"
        )

    if current_data_freq_minutes <= 0:
        logging.error(f"Frecuencia de datos ({current_data_freq_minutes}) inválida.")
        return df, None, None

    periods_in_day = (24 * 60) / current_data_freq_minutes
    ols_window_periods = int(OLS_WINDOW_DAYS * periods_in_day)
    zscore_window_periods = int(ZSCORE_WINDOW_DAYS * periods_in_day)

    logging.info(f"Frecuencia de datos para cálculos: {current_data_freq_minutes} min.")
    logging.info(
        f"Ventana OLS: {OLS_WINDOW_DAYS} días = {ols_window_periods} periodos."
    )
    logging.info(
        f"Ventana Z-Score: {ZSCORE_WINDOW_DAYS} días = {zscore_window_periods} periodos."
    )

    df = calculate_rolling_ols_and_spread(
        df, ASSET_Y_COL, ASSET_X_COL, ols_window_periods
    )
    df["z_score"] = calculate_zscore(df["spread"], zscore_window_periods)

    df.dropna(
        subset=["alpha", "beta", "beta_p_value", "ols_r_squared", "spread", "z_score"],
        inplace=True,
    )
    if df.empty:
        logging.error(
            "No hay datos válidos después de calcular OLS (incl. p-value y R² de beta) y Z-score."
        )
        return df, None, None

    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(index=df.index, dtype=float)
    trades_log = []

    current_position = None
    position_entry_market_price_y = None
    position_entry_market_price_x = None
    position_entry_exec_price_y = None
    position_entry_exec_price_x = None
    position_qty_y = 0
    position_qty_x = 0
    position_entry_alpha = None
    position_entry_beta = None
    position_entry_time = None
    position_entry_zscore_value = None
    position_entry_spread_value = None
    is_leg_x_long = None
    position_equity_base_for_pnl_stop = None  # NUEVO: Equity base para P&L Stop
    position_total_entry_fees = None  # NUEVO: Comisiones de entrada del trade actual
    trail_peak_pnl = None  # máx. PnL neto observado para trailing‑stop

    sl_cooldown_until = None
    z_signal_active = False  # Flag to avoid repeated skip logs for the same signal

    for timestamp, row in df.iterrows():
        z_score_val = row["z_score"]
        # --- Funding cash‑flow ------------------------------------------------
        if current_position:
            # Y‑leg
            fr_y = row.get("funding_rate_y", 0.0)
            if fr_y:
                dir_y = 1 if current_position == "long_spread" else -1
                notional_y = position_qty_y * row[ASSET_Y_COL]
                funding_y_amount = fr_y * notional_y * dir_y
                equity -= funding_y_amount
                logging.debug(
                    f"{timestamp}: Funding Y leg: fr_y={fr_y:.6f}, notional_y={notional_y:.2f}, dir={dir_y}, amount={funding_y_amount:.2f}"
                )

            # X‑leg
            fr_x = row.get("funding_rate_x", 0.0)
            if fr_x:
                dir_x = 1 if is_leg_x_long else -1
                notional_x = position_qty_x * row[ASSET_X_COL]
                funding_x_amount = fr_x * notional_x * dir_x
                equity -= funding_x_amount
                logging.debug(
                    f"{timestamp}: Funding X leg: fr_x={fr_x:.6f}, notional_x={notional_x:.2f}, dir={dir_x}, amount={funding_x_amount:.2f}"
                )
        # ---------------------------------------------------------------------

        # --- Mark‑to‑Market valuation (realised + floating PnL) ---
        portfolio_value = equity
        if current_position:
            price_y = row[ASSET_Y_COL]
            price_x = row[ASSET_X_COL]

            # Floating PnL for Y‑leg
            if current_position == "long_spread":
                pnl_y_float = (price_y - position_entry_exec_price_y) * position_qty_y
            else:  # short_spread
                pnl_y_float = (position_entry_exec_price_y - price_y) * position_qty_y

            # Floating PnL for X‑leg
            if is_leg_x_long:
                pnl_x_float = (price_x - position_entry_exec_price_x) * position_qty_x
            else:
                pnl_x_float = (position_entry_exec_price_x - price_x) * position_qty_x

            portfolio_value += pnl_y_float + pnl_x_float

        # Record the marked‑to‑market portfolio value for drawdown stats
        equity_curve[timestamp] = portfolio_value

        if sl_cooldown_until and timestamp < sl_cooldown_until:
            logging.debug(
                f"{timestamp}: Skipping iteration due to cooldown until {sl_cooldown_until}"
            )
            continue
        elif sl_cooldown_until and timestamp >= sl_cooldown_until:
            logging.debug(f"{timestamp}: Cooldown por SL finalizado.")
            sl_cooldown_until = None

        if (
            pd.isna(z_score_val)
            or pd.isna(row["beta"])
            or pd.isna(row["alpha"])
            or pd.isna(row["beta_p_value"])
            or pd.isna(row["ols_r_squared"])
        ):
            continue

        # --- GESTIÓN DE POSICIONES ABIERTAS ---
        if current_position:
            exit_signal = False
            exit_reason = None

            # --- INICIO: NUEVO Stop-Loss por P&L ---
            if (
                PNL_STOP_LOSS_PCT > 0
                and position_equity_base_for_pnl_stop is not None
                and position_total_entry_fees is not None
            ):
                current_market_price_y_float = row[ASSET_Y_COL]
                current_market_price_x_float = row[ASSET_X_COL]
                pnl_y_float = 0
                pnl_x_float = 0

                if pd.isna(current_market_price_y_float) or pd.isna(
                    current_market_price_x_float
                ):
                    logging.warning(
                        f"{timestamp}: Precio de mercado NaN para P&L flotante. Y: {current_market_price_y_float}, X: {current_market_price_x_float}. Saltando P&L Stop check esta barra."
                    )
                else:
                    if current_position == "long_spread":
                        pnl_y_float = (
                            current_market_price_y_float - position_entry_exec_price_y
                        ) * position_qty_y
                    else:  # short_spread
                        pnl_y_float = (
                            position_entry_exec_price_y - current_market_price_y_float
                        ) * position_qty_y

                    if is_leg_x_long:
                        pnl_x_float = (
                            current_market_price_x_float - position_entry_exec_price_x
                        ) * position_qty_x
                    else:  # short_spread_x
                        pnl_x_float = (
                            position_entry_exec_price_x - current_market_price_x_float
                        ) * position_qty_x

                    floating_pnl_gross = pnl_y_float + pnl_x_float
                    floating_pnl_net_after_entry_costs = (
                        floating_pnl_gross - position_total_entry_fees
                    )
                    max_loss_for_trade = (
                        position_equity_base_for_pnl_stop * PNL_STOP_LOSS_PCT
                    )

                    # Log floating PnL details for P&L Stop-Loss
                    logging.debug(
                        f"{timestamp}: Floating PnL gross: {floating_pnl_gross:.2f}, "
                        f"net after entry fees: {floating_pnl_net_after_entry_costs:.2f}, "
                        f"max_loss_allowed: {max_loss_for_trade:.2f}"
                    )

                    if floating_pnl_net_after_entry_costs < -max_loss_for_trade:
                        exit_signal = True
                        exit_reason = f"P&L Stop Loss ({floating_pnl_net_after_entry_costs:.2f} < -{max_loss_for_trade:.2f})"
                        sl_cooldown_until = timestamp + timedelta(
                            hours=SL_COOLDOWN_HOURS
                        )
                        logging.warning(
                            f"{timestamp}: P&L Stop Loss activado para {current_position}. PnL Flotante Neto: {floating_pnl_net_after_entry_costs:.2f}. Cooldown hasta {sl_cooldown_until}"
                        )

                    # --- Trailing‑stop check ---
                    if floating_pnl_net_after_entry_costs > 0:
                        if (
                            trail_peak_pnl is None
                            or floating_pnl_net_after_entry_costs > trail_peak_pnl
                        ):
                            trail_peak_pnl = floating_pnl_net_after_entry_costs
                        elif (
                            trail_peak_pnl - floating_pnl_net_after_entry_costs
                        ) > trail_peak_pnl * TRAIL_STOP_PCT:
                            exit_signal = True
                            exit_reason = (
                                f"Trailing Stop ({floating_pnl_net_after_entry_costs:.2f} < "
                                f"{trail_peak_pnl*(1-TRAIL_STOP_PCT):.2f})"
                            )
                            sl_cooldown_until = timestamp + timedelta(
                                hours=SL_COOLDOWN_HOURS
                            )
                            logging.warning(
                                f"{timestamp}: Trailing‑stop activado. Cooldown hasta {sl_cooldown_until}"
                            )
                    # -----------------------------------------------------------
            # --- FIN: NUEVO Stop-Loss por P&L ---

            if not exit_signal:  # Solo comprobar Z-Score si P&L Stop no se activó
                if (
                    current_position == "long_spread" and z_score_val >= -EXIT_ZSCORE
                ) or (
                    current_position == "short_spread" and z_score_val <= EXIT_ZSCORE
                ):
                    exit_signal = True
                    exit_reason = f"Z-Score Reversion ({z_score_val:.2f})"
                elif (
                    current_position == "long_spread"
                    and z_score_val <= -STOP_LOSS_ZSCORE
                ) or (
                    current_position == "short_spread"
                    and z_score_val >= STOP_LOSS_ZSCORE
                ):
                    exit_signal = True
                    exit_reason = f"Z-Score SL ({z_score_val:.2f})"  # Modificado
                    sl_cooldown_until = timestamp + timedelta(hours=SL_COOLDOWN_HOURS)
                    logging.warning(
                        f"{timestamp}: Z-Score Stop Loss activado. Cooldown hasta {sl_cooldown_until}"
                    )

            if exit_signal:
                market_price_y_exit = row[ASSET_Y_COL]
                market_price_x_exit = row[ASSET_X_COL]
                spread_value_at_exit = row["spread"]

                if current_position == "long_spread":
                    exec_price_y_exit = apply_slippage(
                        market_price_y_exit, "sell", SLIPPAGE_PCT
                    )
                    pnl_y = (
                        exec_price_y_exit - position_entry_exec_price_y
                    ) * position_qty_y
                else:
                    exec_price_y_exit = apply_slippage(
                        market_price_y_exit, "buy", SLIPPAGE_PCT
                    )
                    pnl_y = (
                        position_entry_exec_price_y - exec_price_y_exit
                    ) * position_qty_y

                if is_leg_x_long:
                    exec_price_x_exit = apply_slippage(
                        market_price_x_exit, "sell", SLIPPAGE_PCT
                    )
                    pnl_x = (
                        exec_price_x_exit - position_entry_exec_price_x
                    ) * position_qty_x
                else:
                    exec_price_x_exit = apply_slippage(
                        market_price_x_exit, "buy", SLIPPAGE_PCT
                    )
                    pnl_x = (
                        position_entry_exec_price_x - exec_price_x_exit
                    ) * position_qty_x

                slippage_cost_y_exit = (
                    abs(exec_price_y_exit - market_price_y_exit) * position_qty_y
                )
                slippage_cost_x_exit = (
                    abs(exec_price_x_exit - market_price_x_exit) * position_qty_x
                )

                fee_cost_y_exit = abs(
                    exec_price_y_exit * position_qty_y * TAKER_FEE_PCT
                )
                fee_cost_x_exit = abs(
                    exec_price_x_exit * position_qty_x * TAKER_FEE_PCT
                )
                total_fees_exit = fee_cost_y_exit + fee_cost_x_exit

                pnl_trade_gross = pnl_y + pnl_x
                pnl_trade_net_this_exit_leg = pnl_trade_gross - total_fees_exit

                equity += pnl_trade_net_this_exit_leg
                equity_curve[timestamp] = equity

                final_trade_pnl_net_calculated = 0
                updated_in_log = False
                for i in range(len(trades_log) - 1, -1, -1):
                    if (
                        trades_log[i]["exit_time"] is None
                        and trades_log[i]["position_type"] == current_position
                    ):
                        trades_log[i].update(
                            {
                                "exit_time": timestamp,
                                "exit_zscore": z_score_val,
                                "reason_exit": exit_reason,
                                "market_price_y_exit": market_price_y_exit,
                                "exec_price_y_exit": exec_price_y_exit,
                                "pnl_y": pnl_y,
                                "market_price_x_exit": market_price_x_exit,
                                "exec_price_x_exit": exec_price_x_exit,
                                "pnl_x": pnl_x,
                                "slippage_cost_y_exit": slippage_cost_y_exit,
                                "slippage_cost_x_exit": slippage_cost_x_exit,
                                "fee_cost_y_exit": fee_cost_y_exit,
                                "fee_cost_x_exit": fee_cost_x_exit,
                                "spread_value_at_exit": spread_value_at_exit,
                                "total_pnl_gross": pnl_trade_gross,
                                "total_fees": trades_log[i]["entry_fees_total"]
                                + total_fees_exit,
                                "equity_after_trade": equity,
                            }
                        )
                        trades_log[i]["final_trade_pnl_net"] = (
                            trades_log[i]["total_pnl_gross"]
                            - trades_log[i]["total_fees"]
                        )
                        final_trade_pnl_net_calculated = trades_log[i][
                            "final_trade_pnl_net"
                        ]
                        updated_in_log = True
                        break
                if not updated_in_log:
                    logging.error(
                        f"Error: No se encontró trade abierto {current_position} para actualizar en {timestamp}"
                    )

                logging.info(
                    f"TRADE CLOSED: {current_position} | Reason: {exit_reason} | PnL Net: {final_trade_pnl_net_calculated:.2f} | Equity: {equity:.2f} | "
                    f"PnL Y: {pnl_y:.2f} (Exec Px: {exec_price_y_exit:.4f}), PnL X: {pnl_x:.2f} (Exec Px: {exec_price_x_exit:.4f}) | "
                    f"Fees Exit: {total_fees_exit:.2f}, Slip Exit Y: {slippage_cost_y_exit:.2f}, Slip Exit X: {slippage_cost_x_exit:.2f}"
                )
                # Log net PnL per leg after slippage and fees
                net_pnl_y = pnl_y - slippage_cost_y_exit - fee_cost_y_exit
                net_pnl_x = pnl_x - slippage_cost_x_exit - fee_cost_x_exit
                logging.info(
                    f"{timestamp}: Net PnL per leg: Y={net_pnl_y:.2f}, X={net_pnl_x:.2f}"
                )

                current_position = None
                is_leg_x_long = None
                position_equity_base_for_pnl_stop = None  # Resetear
                position_total_entry_fees = None  # Resetear
                trail_peak_pnl = None

                if (
                    "SL" in exit_reason or "Stop Loss" in exit_reason
                ):  # Modificado para cubrir ambos tipos de stop
                    continue

        # --- LÓGICA DE APERTURA DE POSICIÓN ---
        if not current_position:
            # Skip if still in cooldown
            if sl_cooldown_until and timestamp < sl_cooldown_until:
                continue
            # Check if Z-score triggers an entry signal
            z_signal = (
                z_score_val < -ENTRY_ZSCORE and z_score_val > -STOP_LOSS_ZSCORE
            ) or (z_score_val > ENTRY_ZSCORE and z_score_val < STOP_LOSS_ZSCORE)
            if not z_signal:
                # Reset on no signal
                z_signal_active = False
                continue
            # If this is the first bar of the signal, allow logging; otherwise skip
            if z_signal_active:
                continue
            z_signal_active = True
            # From here, proceed to filter checks and potential trade entry
            if (
                row["beta_p_value"] >= BETA_P_VALUE_THRESHOLD
                or row["ols_r_squared"] < MIN_OLS_R_SQUARED_THRESHOLD
            ):
                logging.debug(
                    f"{timestamp}: Skipping trade: beta_p_value={row['beta_p_value']:.4f} (threshold {BETA_P_VALUE_THRESHOLD}), ols_r_squared={row['ols_r_squared']:.4f} (threshold {MIN_OLS_R_SQUARED_THRESHOLD})"
                )
                continue

            # Skip if beta quality filter fails (2.1)
            if not row.get("beta_ok", False):
                logging.debug(
                    f"{timestamp}: Skipping trade: beta_ok False (beta={row['beta']:.4f} thresholds [{BETA_MIN_THRESHOLD}, {BETA_MAX_THRESHOLD}], beta_roll_std_20={row['beta_roll_std_20']:.4f} threshold {BETA_ROLL_STD_MAX})"
                )
                continue

            # NUEVO: Guardar equity base para P&L stop y tamaño del trade
            current_equity_base_for_trade_and_pnl_stop = equity

            trade_nominal_total = (
                current_equity_base_for_trade_and_pnl_stop * POSITION_SIZE_PCT
            )
            beta_at_trade = row["beta"]
            beta_abs = abs(beta_at_trade)
            if (1 + beta_abs) == 0:
                logging.warning(
                    f"{timestamp}: beta_abs es tal que (1+beta_abs) es cero. Beta: {beta_at_trade}. Saltando trade."
                )
                continue
            nominal_y_alloc = trade_nominal_total / (1 + beta_abs)
            nominal_x_alloc = trade_nominal_total * beta_abs / (1 + beta_abs)
            if (
                nominal_y_alloc < MIN_NOMINAL_PER_LEG
                or nominal_x_alloc < MIN_NOMINAL_PER_LEG
            ):
                logging.debug(
                    f"{timestamp}: Skipping trade: nominal allocations too small: y={nominal_y_alloc:.2f}, x={nominal_x_alloc:.2f}, threshold={MIN_NOMINAL_PER_LEG}"
                )
                continue

            current_market_price_y_entry = row[ASSET_Y_COL]
            current_market_price_x_entry = row[ASSET_X_COL]
            current_spread_value_at_entry = row["spread"]

            if (
                pd.isna(current_market_price_y_entry)
                or pd.isna(current_market_price_x_entry)
                or current_market_price_y_entry <= 0
                or current_market_price_x_entry <= 0
            ):
                logging.warning(
                    f"{timestamp}: Precio inválido para Y o X. Y: {current_market_price_y_entry}, X: {current_market_price_x_entry}. Saltando trade."
                )
                continue

            entry_type = None
            current_is_leg_x_long = None
            if z_score_val < -ENTRY_ZSCORE and z_score_val > -STOP_LOSS_ZSCORE:
                entry_type = "long_spread"
                exec_price_y_open = apply_slippage(
                    current_market_price_y_entry, "buy", SLIPPAGE_PCT
                )
                if beta_at_trade < 0:
                    exec_price_x_open = apply_slippage(
                        current_market_price_x_entry, "buy", SLIPPAGE_PCT
                    )
                    current_is_leg_x_long = True
                else:
                    exec_price_x_open = apply_slippage(
                        current_market_price_x_entry, "sell", SLIPPAGE_PCT
                    )
                    current_is_leg_x_long = False
            elif z_score_val > ENTRY_ZSCORE and z_score_val < STOP_LOSS_ZSCORE:
                entry_type = "short_spread"
                exec_price_y_open = apply_slippage(
                    current_market_price_y_entry, "sell", SLIPPAGE_PCT
                )
                if beta_at_trade < 0:
                    exec_price_x_open = apply_slippage(
                        current_market_price_x_entry, "sell", SLIPPAGE_PCT
                    )
                    current_is_leg_x_long = False
                else:
                    exec_price_x_open = apply_slippage(
                        current_market_price_x_entry, "buy", SLIPPAGE_PCT
                    )
                    current_is_leg_x_long = True

            if entry_type:
                if exec_price_y_open <= 0 or exec_price_x_open <= 0:
                    logging.warning(
                        f"{timestamp}: Precio de ejecución inválido post-slippage. Y_exec: {exec_price_y_open}, X_exec: {exec_price_x_open}. Saltando trade."
                    )
                    continue

                current_qty_y = nominal_y_alloc / exec_price_y_open
                current_qty_x = nominal_x_alloc / exec_price_x_open

                slippage_cost_y_entry = (
                    abs(exec_price_y_open - current_market_price_y_entry)
                    * current_qty_y
                )
                slippage_cost_x_entry = (
                    abs(exec_price_x_open - current_market_price_x_entry)
                    * current_qty_x
                )

                fee_cost_y_entry = abs(
                    exec_price_y_open * current_qty_y * TAKER_FEE_PCT
                )
                fee_cost_x_entry = abs(
                    exec_price_x_open * current_qty_x * TAKER_FEE_PCT
                )
                total_fees_entry = fee_cost_y_entry + fee_cost_x_entry

                equity -= total_fees_entry  # Equity se actualiza DESPUÉS de haber guardado current_equity_base_for_trade_and_pnl_stop
                equity_curve[timestamp] = equity
                trail_peak_pnl = None  # reinicia trailing para el nuevo trade

                current_position = entry_type
                position_entry_market_price_y = current_market_price_y_entry
                position_entry_market_price_x = current_market_price_x_entry
                position_entry_exec_price_y = exec_price_y_open
                position_entry_exec_price_x = exec_price_x_open
                position_qty_y = current_qty_y
                position_qty_x = current_qty_x
                position_entry_alpha = row["alpha"]
                position_entry_beta = beta_at_trade
                position_entry_time = timestamp
                position_entry_zscore_value = z_score_val
                position_entry_spread_value = current_spread_value_at_entry
                is_leg_x_long = current_is_leg_x_long
                # NUEVO: Guardar variables de estado para P&L Stop
                position_equity_base_for_pnl_stop = (
                    current_equity_base_for_trade_and_pnl_stop
                )
                position_total_entry_fees = total_fees_entry

                trades_log.append(
                    {
                        "entry_time": timestamp,
                        "exit_time": None,
                        "position_type": current_position,
                        "entry_zscore": position_entry_zscore_value,
                        "exit_zscore": None,
                        "reason_entry": f"Z-Score Signal ({z_score_val:.2f})",
                        "reason_exit": None,
                        "asset_y": ASSET_Y_COL,
                        "market_price_y_entry": position_entry_market_price_y,
                        "exec_price_y_entry": position_entry_exec_price_y,
                        "market_price_y_exit": None,
                        "exec_price_y_exit": None,
                        "qty_y": position_qty_y,
                        "nominal_y_alloc": nominal_y_alloc,
                        "pnl_y": None,
                        "slippage_cost_y_entry": slippage_cost_y_entry,
                        "fee_cost_y_entry": fee_cost_y_entry,
                        "slippage_cost_y_exit": None,
                        "fee_cost_y_exit": None,
                        "asset_x": ASSET_X_COL,
                        "market_price_x_entry": position_entry_market_price_x,
                        "exec_price_x_entry": position_entry_exec_price_x,
                        "market_price_x_exit": None,
                        "exec_price_x_exit": None,
                        "qty_x": position_qty_x,
                        "nominal_x_alloc": nominal_x_alloc,
                        "pnl_x": None,
                        "slippage_cost_x_entry": slippage_cost_x_entry,
                        "fee_cost_x_entry": fee_cost_x_entry,
                        "slippage_cost_x_exit": None,
                        "fee_cost_x_exit": None,
                        "entry_alpha": position_entry_alpha,
                        "entry_beta": position_entry_beta,
                        "beta_p_value_at_entry": row["beta_p_value"],
                        "ols_r_squared_at_entry": row["ols_r_squared"],
                        "spread_value_at_entry": position_entry_spread_value,
                        "spread_value_at_exit": None,
                        "is_leg_x_long_at_entry": is_leg_x_long,
                        "entry_fees_total": total_fees_entry,
                        "pnl_net_after_entry_costs": -total_fees_entry,
                        "total_fees": None,
                        "total_pnl_gross": None,
                        "final_trade_pnl_net": None,
                        "equity_after_trade": None,
                    }
                )
                logging.info(
                    f"TRADE OPENED: {current_position} | Z: {z_score_val:.2f} | Beta: {beta_at_trade:.4f} "
                    f"(pVal: {row['beta_p_value']:.4f}, R²: {row['ols_r_squared']:.4f}) | "
                    f"Market Y: {current_market_price_y_entry:.4f}, Exec Y: {exec_price_y_open:.4f}, Qty Y: {current_qty_y:.4f}, Slip Y: {slippage_cost_y_entry:.2f}, Fee Y: {fee_cost_y_entry:.2f} | "
                    f"Market X: {current_market_price_x_entry:.4f}, Exec X: {exec_price_x_open:.4f}, Qty X: {current_qty_x:.4f}, Slip X: {slippage_cost_x_entry:.2f}, Fee X: {fee_cost_x_entry:.2f} | "
                    f"Nom Y: {nominal_y_alloc:.2f}, Nom X: {nominal_x_alloc:.2f} | Eq after fees: {equity:.2f}"
                )

    if current_position:  # Si hay posición abierta al final del backtest
        logging.info(
            f"Cerrando posición abierta al final del backtest: {current_position}"
        )
        last_row = df.iloc[-1]
        market_price_y_exit_final = last_row[ASSET_Y_COL]
        market_price_x_exit_final = last_row[ASSET_X_COL]
        spread_value_at_exit_final = last_row["spread"]

        if current_position == "long_spread":
            exec_price_y_close = apply_slippage(
                market_price_y_exit_final, "sell", SLIPPAGE_PCT
            )
            pnl_y = (exec_price_y_close - position_entry_exec_price_y) * position_qty_y
        else:
            exec_price_y_close = apply_slippage(
                market_price_y_exit_final, "buy", SLIPPAGE_PCT
            )
            pnl_y = (position_entry_exec_price_y - exec_price_y_close) * position_qty_y
        if is_leg_x_long:
            exec_price_x_close = apply_slippage(
                market_price_x_exit_final, "sell", SLIPPAGE_PCT
            )
            pnl_x = (exec_price_x_close - position_entry_exec_price_x) * position_qty_x
        else:
            exec_price_x_close = apply_slippage(
                market_price_x_exit_final, "buy", SLIPPAGE_PCT
            )
            pnl_x = (position_entry_exec_price_x - exec_price_x_close) * position_qty_x

        slippage_cost_y_exit_final = (
            abs(exec_price_y_close - market_price_y_exit_final) * position_qty_y
        )
        slippage_cost_x_exit_final = (
            abs(exec_price_x_close - market_price_x_exit_final) * position_qty_x
        )
        fee_cost_y_exit_final = abs(exec_price_y_close * position_qty_y * TAKER_FEE_PCT)
        fee_cost_x_exit_final = abs(exec_price_x_close * position_qty_x * TAKER_FEE_PCT)
        total_fees_close_final = fee_cost_y_exit_final + fee_cost_x_exit_final

        pnl_trade_gross = pnl_y + pnl_x
        pnl_trade_net_exit_leg = pnl_trade_gross - total_fees_close_final
        equity += pnl_trade_net_exit_leg
        equity_curve[df.index[-1]] = equity

        final_trade_pnl_net_calculated_eob = 0
        updated_in_log = False
        for i in range(len(trades_log) - 1, -1, -1):
            if trades_log[i]["exit_time"] is None:
                trades_log[i].update(
                    {
                        "exit_time": df.index[-1],
                        "exit_zscore": last_row["z_score"],
                        "reason_exit": "End of Backtest",
                        "market_price_y_exit": market_price_y_exit_final,
                        "exec_price_y_exit": exec_price_y_close,
                        "pnl_y": pnl_y,
                        "market_price_x_exit": market_price_x_exit_final,
                        "exec_price_x_exit": exec_price_x_close,
                        "pnl_x": pnl_x,
                        "slippage_cost_y_exit": slippage_cost_y_exit_final,
                        "slippage_cost_x_exit": slippage_cost_x_exit_final,
                        "fee_cost_y_exit": fee_cost_y_exit_final,
                        "fee_cost_x_exit": fee_cost_x_exit_final,
                        "spread_value_at_exit": spread_value_at_exit_final,
                        "total_pnl_gross": pnl_trade_gross,
                        "total_fees": trades_log[i]["entry_fees_total"]
                        + total_fees_close_final,
                        "equity_after_trade": equity,
                    }
                )
                trades_log[i]["final_trade_pnl_net"] = (
                    trades_log[i]["total_pnl_gross"] - trades_log[i]["total_fees"]
                )
                final_trade_pnl_net_calculated_eob = trades_log[i][
                    "final_trade_pnl_net"
                ]
                updated_in_log = True
                break
        if not updated_in_log:
            logging.error(
                f"Error: No se encontró trade abierto {current_position} para actualizar al final."
            )
        logging.info(
            f"TRADE CLOSED (End of Data): {current_position} | PnL Net: {final_trade_pnl_net_calculated_eob:.2f} | Equity: {equity:.2f} | "
            f"PnL Y: {pnl_y:.2f} (Exec Px: {exec_price_y_close:.4f}), PnL X: {pnl_x:.2f} (Exec Px: {exec_price_x_close:.4f})"
        )

    logging.info("Backtest finalizado.")
    return df, equity_curve.dropna(), pd.DataFrame(trades_log)


# --- Métricas y Gráficos ---
def calculate_performance_metrics(
    equity_curve, trades_df, initial_capital, risk_free_rate_annual
):
    if equity_curve is None or equity_curve.empty:
        logging.warning("Equity curve vacía. No se pueden calcular métricas.")
        return {}
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity / initial_capital) - 1
    duration_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
    duration_years = duration_seconds / (365.25 * 24 * 60 * 60)
    if duration_years < (1 / 365.25):
        duration_years = (
            1 / 365.25
        )  # Avoid division by zero or extremely small numbers for CAGR
    cagr = (
        ((final_equity / initial_capital) ** (1 / duration_years)) - 1
        if duration_years > 0 and initial_capital > 0
        else 0
    )
    rolling_max = equity_curve.cummax()
    daily_drawdown = equity_curve / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    daily_equity_resampled = equity_curve.resample("D").last()
    daily_returns = daily_equity_resampled.pct_change().dropna()
    sharpe_ratio = 0
    sortino_ratio = 0
    if not daily_returns.empty and daily_returns.std() != 0:
        rf_daily = (1 + risk_free_rate_annual) ** (
            1 / 252
        ) - 1  # Assuming 252 trading days
        sharpe_ratio = (
            (daily_returns.mean() - rf_daily) / daily_returns.std() * np.sqrt(252)
        )
        negative_returns = daily_returns[daily_returns < 0]
        if not negative_returns.empty and negative_returns.std() != 0:
            sortino_ratio = (
                (daily_returns.mean() - rf_daily)
                / negative_returns.std()
                * np.sqrt(252)
            )
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    num_trades = len(trades_df) if trades_df is not None else 0

    winning_trades = (
        trades_df[trades_df["final_trade_pnl_net"] > 0]
        if trades_df is not None
        and not trades_df.empty
        and "final_trade_pnl_net" in trades_df.columns
        else pd.DataFrame()
    )
    losing_trades = (
        trades_df[trades_df["final_trade_pnl_net"] <= 0]
        if trades_df is not None
        and not trades_df.empty
        and "final_trade_pnl_net" in trades_df.columns
        else pd.DataFrame()
    )

    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    avg_win = (
        winning_trades["final_trade_pnl_net"].mean() if not winning_trades.empty else 0
    )
    avg_loss = (
        losing_trades["final_trade_pnl_net"].mean() if not losing_trades.empty else 0
    )  # Avg loss will be negative or zero
    sum_wins = (
        winning_trades["final_trade_pnl_net"].sum() if not winning_trades.empty else 0
    )
    sum_losses = (
        abs(losing_trades["final_trade_pnl_net"].sum())
        if not losing_trades.empty
        else 0
    )  # Sum of absolute losses
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else np.inf
    metrics = {
        "Initial Equity": initial_capital,
        "Final Equity": final_equity,
        "Total Return (%)": total_return * 100,
        "CAGR (%)": cagr * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Sharpe Ratio (Annualized, Daily Ret)": sharpe_ratio,
        "Sortino Ratio (Annualized, Daily Ret)": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate * 100,
        "Average Win ($)": avg_win,
        "Average Loss ($)": avg_loss,
        "Profit Factor": profit_factor,
        "Total Fees Paid ($)": (
            trades_df["total_fees"].sum()
            if trades_df is not None
            and "total_fees" in trades_df.columns
            and not trades_df.empty
            else 0
        ),
    }
    logging.info("--- Resumen de Métricas ---")
    for key, value in metrics.items():
        logging.info(
            f"{key}: {value:.2f}"
            if isinstance(value, (float, np.float64))
            else f"{key}: {value}"
        )
    return metrics


def plot_results(df_with_indicators, equity_curve, trades_df, plot_dir):
    if equity_curve is None or equity_curve.empty:
        logging.warning("Equity curve vacía. No se pueden generar gráficos.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")

    # --- Combined Figure: Equity Curve (top) & Spread Z‑Score (bottom) ---
    fig, (ax_eq, ax) = plt.subplots(2, 1, figsize=(15, 12), sharex=False)

    # Equity curve subplot ---------------------------------------------------
    ax_eq.plot(equity_curve.index, equity_curve, color="blue", label="Equity ($)")
    ax_eq.set_title("Equity Curve")
    ax_eq.set_xlabel("Date")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.legend(loc="best")
    ax_eq.grid(True, linestyle=":", alpha=0.7)

    if df_with_indicators is None or df_with_indicators.empty:
        logging.warning(
            "DataFrame con indicadores vacío. No se pueden generar gráficos de spread/z-score."
        )
        return
    logging.info(f"Guardando gráficos en: {plot_dir}")

    df_plot = df_with_indicators.loc[
        df_with_indicators.index.isin(equity_curve.index)
    ].copy()
    if df_plot.empty and not df_with_indicators.empty:
        df_plot = (
            df_with_indicators.copy()
        )  # Fallback to all data if equity curve has different timestamps
    elif df_plot.empty and df_with_indicators.empty:
        logging.warning(
            "df_plot está vacío, no se pueden generar gráficos de spread/z-score."
        )
        return

    # 2. Spread Z-Score con Trades (now uses ax from combined figure)
    if "z_score" in df_plot.columns:
        ax.plot(
            df_plot.index,
            df_plot["z_score"],
            label="Z-Score",
            color="dodgerblue",
            alpha=0.9,
            linewidth=1.5,
        )
        ax.axhline(
            ENTRY_ZSCORE,
            color="gray",
            linestyle="--",
            label=f"Entry Z={ENTRY_ZSCORE:.2f}",
        )
        ax.axhline(-ENTRY_ZSCORE, color="gray", linestyle="--")
        ax.axhline(
            EXIT_ZSCORE,
            color="silver",
            linestyle=":",
            label=f"Exit Z={EXIT_ZSCORE:.2f}",
        )
        ax.axhline(-EXIT_ZSCORE, color="silver", linestyle=":")
        ax.axhline(
            STOP_LOSS_ZSCORE,
            color="black",
            linestyle="-.",
            label=f"Stop Loss Z={STOP_LOSS_ZSCORE:.1f}",
        )
        ax.axhline(-STOP_LOSS_ZSCORE, color="black", linestyle="-.")

        if trades_df is not None and not trades_df.empty:
            trades_df_copy = trades_df.copy()
            # Ensure entry_time is datetime for plotting
            if (
                "entry_time" in trades_df_copy.columns
                and not pd.api.types.is_datetime64_any_dtype(
                    trades_df_copy["entry_time"]
                )
            ):
                try:
                    trades_df_copy["entry_time"] = pd.to_datetime(
                        trades_df_copy["entry_time"]
                    )
                except:
                    logging.error(
                        "Error convirtiendo entry_time a datetime para plotting."
                    )

            valid_trades = (
                trades_df_copy[trades_df_copy["entry_time"].isin(df_plot.index)]
                if "entry_time" in trades_df_copy.columns
                else pd.DataFrame()
            )

            long_entries = valid_trades[valid_trades["position_type"] == "long_spread"]
            short_entries = valid_trades[
                valid_trades["position_type"] == "short_spread"
            ]

            if not long_entries.empty and "entry_time" in long_entries.columns:
                entry_times_long = pd.to_datetime(
                    long_entries["entry_time"]
                )  # Ensure datetime
                plot_entry_times_long = entry_times_long[
                    entry_times_long.isin(df_plot.index)
                ]
                if not plot_entry_times_long.empty:
                    ax.scatter(
                        plot_entry_times_long,
                        df_plot.loc[plot_entry_times_long, "z_score"],
                        marker="^",
                        color="limegreen",
                        s=120,
                        label="Long Entry",
                        zorder=5,
                        edgecolors="black",
                    )
            if not short_entries.empty and "entry_time" in short_entries.columns:
                entry_times_short = pd.to_datetime(
                    short_entries["entry_time"]
                )  # Ensure datetime
                plot_entry_times_short = entry_times_short[
                    entry_times_short.isin(df_plot.index)
                ]
                if not plot_entry_times_short.empty:
                    ax.scatter(
                        plot_entry_times_short,
                        df_plot.loc[plot_entry_times_short, "z_score"],
                        marker="v",
                        color="red",
                        s=120,
                        label="Short Entry",
                        zorder=5,
                        edgecolors="black",
                    )

            if "exit_time" in valid_trades.columns:
                exits_to_plot = valid_trades.dropna(subset=["exit_time"])
                if not exits_to_plot.empty:
                    if not pd.api.types.is_datetime64_any_dtype(
                        exits_to_plot["exit_time"]
                    ):
                        try:
                            exits_to_plot["exit_time"] = pd.to_datetime(
                                exits_to_plot["exit_time"]
                            )
                        except:
                            logging.error(
                                "Error convirtiendo exit_time a datetime para plotting."
                            )

                    valid_exit_times_in_plot = (
                        exits_to_plot["exit_time"][
                            exits_to_plot["exit_time"].isin(df_plot.index)
                        ]
                        if "exit_time" in exits_to_plot.columns
                        else pd.Series(dtype="datetime64[ns]")
                    )
                    if not valid_exit_times_in_plot.empty:
                        ax.scatter(
                            valid_exit_times_in_plot,
                            df_plot.loc[valid_exit_times_in_plot, "z_score"],
                            marker="o",
                            color="mediumpurple",
                            s=80,
                            label="Exit",
                            zorder=5,
                            edgecolors="black",
                        )
        # --- OLS filter shading ---
        condition_beta_pval = df_plot["beta_p_value"] >= BETA_P_VALUE_THRESHOLD
        condition_r_squared = df_plot["ols_r_squared"] < MIN_OLS_R_SQUARED_THRESHOLD
        filter_active = condition_beta_pval | condition_r_squared
        if filter_active.any():
            y_min, y_max = ax.get_ylim()
            ax.fill_between(
                df_plot.index,
                y_min,
                y_max,
                where=filter_active.astype(bool),
                color="lightcoral",
                alpha=0.3,
                label=(
                    f"Filtro OLS activo "
                    f"(pVal>{BETA_P_VALUE_THRESHOLD:.2f} "
                    f"o R²<{MIN_OLS_R_SQUARED_THRESHOLD:.2f})"
                ),
            )
        ax.set_title("Spread Z-Score con Señales de Trading", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Z-Score")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.7)

    # ---- Save combined figure ----
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "equity_spread_summary.png"))
    plt.close(fig)

    logging.info("Gráfico combinado guardado.")


def run_walkforward_backtest(df_input, train_days: int, test_days: int, **params):
    """
    Walk‑forward back‑test.
    Each cycle uses `train_days` worth of history (only to warm‑up the
    rolling windows) and evaluates trades during the subsequent
    `test_days` out‑of‑sample period.  Equity is stitched at the
    boundaries so results are continuous.

    Returns
    -------
    equity_curve : pd.Series
        Combined equity curve for all test windows.
    trades_df : pd.DataFrame
        Concatenated trade log for every window.
    """
    import pandas as pd

    all_trades = []
    overall_equity = pd.Series(dtype=float)

    t0 = df_input.index.min()
    t_end = df_input.index.max()
    cursor = t0

    while True:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if train_end >= t_end:
            break

        # Slice contains the training look‑back (for rolling stats) plus the test window.
        df_slice = df_input.loc[cursor:test_end]
        if df_slice.empty:
            break

        # Use the equity at the splice point as fresh initial capital.
        slice_initial = (
            overall_equity.iloc[-1]
            if not overall_equity.empty
            else params.get("INITIAL_CAPITAL", INITIAL_CAPITAL)
        )
        params_slice = dict(params)
        params_slice["INITIAL_CAPITAL"] = slice_initial

        _, slice_equity, slice_trades = run_backtest(df_slice, **params_slice)

        if slice_equity is None or slice_equity.empty:
            cursor = train_end  # advance anyway
            continue

        # Keep only the part of the equity that belongs to the *test* window.
        slice_equity = slice_equity.loc[train_end:]
        overall_equity = pd.concat([overall_equity, slice_equity])

        if slice_trades is not None and not slice_trades.empty:
            all_trades.append(slice_trades)

        cursor = train_end  # next cycle starts where this one trained

    trades_df = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )
    return overall_equity, trades_df


# --- Ejecución Principal ---
if __name__ == "__main__":
    logging.info("--- INICIO DEL SCRIPT DE BACKTESTING ---")
    params = StrategyParameters()
    df_original = load_data(DATA_FILEPATH, params)
    if df_original is not None and not df_original.empty:
        # Usar OOP wrapper PairTradingBacktester
        backtester = PairTradingBacktester(params)
        df_processed, equity_curve, trades_summary_df = backtester.run(df_original)
        trades_df_for_metrics = (
            trades_summary_df if trades_summary_df is not None else pd.DataFrame()
        )
        if equity_curve is not None and not equity_curve.empty:
            if (
                trades_summary_df is not None and not trades_summary_df.empty
            ):  # Asegurarse que no está vacío
                # Convertir columnas de fecha/hora a string ANTES de guardar CSV y ANTES de calcular métricas si estas esperan strings
                trades_summary_df_csv = trades_summary_df.copy()
                for col in ["entry_time", "exit_time"]:
                    if (
                        col in trades_summary_df_csv.columns
                        and pd.api.types.is_datetime64_any_dtype(
                            trades_summary_df_csv[col]
                        )
                    ):
                        trades_summary_df_csv[col] = trades_summary_df_csv[
                            col
                        ].dt.strftime("%Y-%m-%d %H:%M:%S")
                trades_summary_df_csv.to_csv(TRADES_CSV_FILE, index=False)
                logging.info(f"Resumen de trades guardado en: {TRADES_CSV_FILE}")
                # Para métricas y plots, usar el DataFrame con datetimes si es necesario, o el copiado (trades_df_for_metrics)
                # calculate_performance_metrics espera trades_df, que aquí es trades_df_for_metrics (la copia original del output del backtest)
                # plot_results también espera trades_df con datetimes para el scatter plot
                metrics = calculate_performance_metrics(
                    equity_curve,
                    trades_df_for_metrics,
                    INITIAL_CAPITAL,
                    RISK_FREE_RATE_ANNUAL,
                )
                plot_results(
                    df_processed, equity_curve, trades_df_for_metrics, PLOT_DIR
                )
            else:
                logging.warning(
                    "No se generaron trades para guardar en CSV o calcular métricas."
                )
                # Aún así, intentar graficar la curva de equity si existe
                plot_results(df_processed, equity_curve, pd.DataFrame(), PLOT_DIR)

        else:
            logging.error("El backtest no produjo una curva de equity válida.")
    else:
        logging.error("No se pudieron cargar los datos. Terminando script.")
    logging.info("--- FIN DEL SCRIPT DE BACKTESTING ---")
