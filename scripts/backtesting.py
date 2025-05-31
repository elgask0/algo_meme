import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import logging
import os
import sqlite3
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import Optional

# --- General Configuration ---
DATA_FILEPATH = "/Users/elgask0/REPOS/algo_meme/trading_data.db"
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


def load_data(filepath, params: StrategyParameters):
    logging.info(f"Loading data from DB: {filepath}")
    try:
        conn = sqlite3.connect(filepath)
        # --- Get mark_price_vwap prices for both symbols ---
        query_mark = f"""
            SELECT ts_start, symbol_id, mark_price
            FROM mark_price_vwap
            WHERE symbol_id IN ('{params.ASSET_Y_COL}', '{params.ASSET_X_COL}')
        """
        df_mark = pd.read_sql_query(query_mark, conn, parse_dates=['ts_start'])
        if df_mark.empty:
            logging.error("No price data available in mark_price_vwap for the assets.")
            conn.close()
            return None
        # Pivot to have separate columns for each symbol
        df_mark_pivot = df_mark.pivot(index='ts_start', columns='symbol_id', values='mark_price')
        # Rename columns to match ASSET_Y_COL and ASSET_X_COL
        df = df_mark_pivot.rename(columns={
            params.ASSET_Y_COL: params.ASSET_Y_COL,
            params.ASSET_X_COL: params.ASSET_X_COL
        })

        # --- Get historical funding rates for both symbols ---
        query_funding = f"""
            SELECT ts AS ts, symbol, funding_rate
            FROM mexc_funding_rate_history
            WHERE symbol IN ('{params.ASSET_Y_COL}', '{params.ASSET_X_COL}')
        """
        df_funding = pd.read_sql_query(query_funding, conn, parse_dates=['ts'])
        conn.close()

        df_funding_y = (
            df_funding[df_funding['symbol'] == params.ASSET_Y_COL]
            .set_index('ts')[['funding_rate']]
            .rename(columns={'funding_rate': 'funding_rate_y'})
        )
        df_funding_x = (
            df_funding[df_funding['symbol'] == params.ASSET_X_COL]
            .set_index('ts')[['funding_rate']]
            .rename(columns={'funding_rate': 'funding_rate_x'})
        )

        # Join funding rates and fill with 0 where data is missing
        df = (
            df.join(df_funding_y, how="left")
              .join(df_funding_x, how="left")
              .fillna({"funding_rate_y": 0.0, "funding_rate_x": 0.0})
        )
        df.sort_index(inplace=True)

        # Remove rows without price data
        df.dropna(subset=[params.ASSET_Y_COL, params.ASSET_X_COL], inplace=True)
        if df.empty:
            logging.error("No data after removing initial NaNs.")
            return None

        logging.info(
            f"Data loaded. Range: {df.index.min()} to {df.index.max()}. Rows: {len(df)}"
        )
        return df
    except sqlite3.Error as e:
        logging.error(f"Error loading data from DB: {e}")
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
        logging.error(f"Z-score window_periods is {window_periods}, must be >= 1.")
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


# --- Main Backtest Logic ---
def run_backtest(df_input, **params):
    # Allow external scripts to override any constant simply by passing
    # keyword arguments, e.g. run_backtest(df, ENTRY_ZSCORE=2.0).
    # The override is done once per call so a grid‑search can explore many
    # combinations safely.
    globals().update(params)
    logging.info("Starting backtest...")
    # --- Log strategy parameters for traceability ---
    try:
        # Gather all fields from StrategyParameters dataclass
        param_keys = list(StrategyParameters.__annotations__.keys())
        current_params = {key: globals().get(key) for key in param_keys}
        logging.info(f"Strategy parameters: {current_params}")
    except Exception as e:
        logging.warning(f"Could not log strategy parameters: {e}")
    df = df_input.copy()

    current_data_freq_minutes = DEFAULT_DATA_FREQ_MINUTES
    if RESAMPLE_ALL_DATA_TO_MINUTES is not None and RESAMPLE_ALL_DATA_TO_MINUTES > 0:
        logging.info(
            f"Resampling all data to {RESAMPLE_ALL_DATA_TO_MINUTES} minute candles."
        )
        df = df.resample(f"{RESAMPLE_ALL_DATA_TO_MINUTES}min").last()
        df.dropna(subset=[ASSET_Y_COL, ASSET_X_COL], inplace=True)
        if df.empty:
            logging.error("No data after general resampling.")
            return df, None, None
        current_data_freq_minutes = RESAMPLE_ALL_DATA_TO_MINUTES
        logging.info(
            f"Data resampled. New range: {df.index.min()} to {df.index.max()}. Rows: {len(df)}"
        )

    if current_data_freq_minutes <= 0:
        logging.error(f"Invalid data frequency ({current_data_freq_minutes}).")
        return df, None, None

    periods_in_day = (24 * 60) / current_data_freq_minutes
    ols_window_periods = int(OLS_WINDOW_DAYS * periods_in_day)
    zscore_window_periods = int(ZSCORE_WINDOW_DAYS * periods_in_day)

    logging.info(f"Data frequency for calculations: {current_data_freq_minutes} min.")
    logging.info(
        f"OLS Window: {OLS_WINDOW_DAYS} days = {ols_window_periods} periods."
    )
    logging.info(
        f"Z-Score Window: {ZSCORE_WINDOW_DAYS} days = {zscore_window_periods} periods."
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
            "No valid data after calculating OLS (including beta p-value and R²) and Z-score."
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
    position_equity_base_for_pnl_stop = None  # NEW: Equity base for P&L Stop
    position_total_entry_fees = None  # NEW: Entry fees for the current trade
    trail_peak_pnl = None  # max net PnL observed for trailing-stop

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
            logging.debug(f"{timestamp}: SL cooldown finished.")
            sl_cooldown_until = None

        if (
            pd.isna(z_score_val)
            or pd.isna(row["beta"])
            or pd.isna(row["alpha"])
            or pd.isna(row["beta_p_value"])
            or pd.isna(row["ols_r_squared"])
        ):
            continue

        # --- OPEN POSITION MANAGEMENT ---
        if current_position:
            exit_signal = False
            exit_reason = None

            # --- START: NEW P&L Stop-Loss ---
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
                        f"{timestamp}: Market price NaN for floating P&L. Y: {current_market_price_y_float}, X: {current_market_price_x_float}. Skipping P&L Stop check this bar."
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
                            f"{timestamp}: P&L Stop Loss activated for {current_position}. Net Floating PnL: {floating_pnl_net_after_entry_costs:.2f}. Cooldown until {sl_cooldown_until}"
                        )

                    # --- Trailing-stop check ---
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
                                f"{timestamp}: Trailing-stop activated. Cooldown until {sl_cooldown_until}"
                            )
                    # -----------------------------------------------------------
            # --- END: NEW P&L Stop-Loss ---

            if not exit_signal:  # Only check Z-Score if P&L Stop was not activated
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
                    exit_reason = f"Z-Score SL ({z_score_val:.2f})"  # Modified
                    sl_cooldown_until = timestamp + timedelta(hours=SL_COOLDOWN_HOURS)
                    logging.warning(
                        f"{timestamp}: Z-Score Stop Loss activated. Cooldown until {sl_cooldown_until}"
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
                        f"Error: No open trade {current_position} found to update at {timestamp}"
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
                position_equity_base_for_pnl_stop = None  # Reset
                position_total_entry_fees = None  # Reset
                trail_peak_pnl = None

                if (
                    "SL" in exit_reason or "Stop Loss" in exit_reason
                ):  # Modified to cover both stop types
                    continue

        # --- POSITION OPENING LOGIC ---
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

            # NEW: Save equity base for P&L stop and trade size
            current_equity_base_for_trade_and_pnl_stop = equity

            trade_nominal_total = (
                current_equity_base_for_trade_and_pnl_stop * POSITION_SIZE_PCT
            )
            beta_at_trade = row["beta"]
            beta_abs = abs(beta_at_trade)
            if (1 + beta_abs) == 0:
                logging.warning(
                    f"{timestamp}: beta_abs is such that (1+beta_abs) is zero. Beta: {beta_at_trade}. Skipping trade."
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
                    f"{timestamp}: Invalid price for Y or X. Y: {current_market_price_y_entry}, X: {current_market_price_x_entry}. Skipping trade."
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
                        f"{timestamp}: Invalid execution price post-slippage. Y_exec: {exec_price_y_open}, X_exec: {exec_price_x_open}. Skipping trade."
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

                equity -= total_fees_entry  # Equity is updated AFTER saving current_equity_base_for_trade_and_pnl_stop
                equity_curve[timestamp] = equity
                trail_peak_pnl = None  # reset trailing for the new trade

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
                # NEW: Save state variables for P&L Stop
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

    if current_position:  # If there is an open position at the end of the backtest
        logging.info(
            f"Closing open position at the end of the backtest: {current_position}"
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
                f"Error: No open trade {current_position} found to update at the end."
            )
        logging.info(
            f"TRADE CLOSED (End of Data): {current_position} | PnL Net: {final_trade_pnl_net_calculated_eob:.2f} | Equity: {equity:.2f} | "
            f"PnL Y: {pnl_y:.2f} (Exec Px: {exec_price_y_close:.4f}), PnL X: {pnl_x:.2f} (Exec Px: {exec_price_x_close:.4f})"
        )

    logging.info("Backtest finished.")
    return df, equity_curve.dropna(), pd.DataFrame(trades_log)


# --- Metrics and Plots ---
def calculate_performance_metrics(
    equity_curve, trades_df, initial_capital, risk_free_rate_annual
):
    if equity_curve is None or equity_curve.empty:
        logging.warning("Empty equity curve. Cannot calculate metrics.")
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
    logging.info("--- Metrics Summary ---")
    for key, value in metrics.items():
        logging.info(
            f"{key}: {value:.2f}"
            if isinstance(value, (float, np.float64))
            else f"{key}: {value}"
        )
    return metrics


def plot_results(df_with_indicators, equity_curve, trades_df, plot_dir):
    if equity_curve is None or equity_curve.empty:
        logging.warning("Empty equity curve. Cannot generate plots.")
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
            "DataFrame with indicators is empty. Cannot generate spread/z-score plots."
        )
        return
    logging.info(f"Saving plots to: {plot_dir}")

    df_plot = df_with_indicators.loc[
        df_with_indicators.index.isin(equity_curve.index)
    ].copy()
    if df_plot.empty and not df_with_indicators.empty:
        df_plot = (
            df_with_indicators.copy()
        )  # Fallback to all data if equity curve has different timestamps
    elif df_plot.empty and df_with_indicators.empty:
        logging.warning(
            "df_plot is empty, cannot generate spread/z_score plots."
        )
        return

    # 2. Spread Z-Score with Trades (now uses ax from combined figure)
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
                        "Error converting entry_time to datetime for plotting."
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
                                "Error converting exit_time to datetime for plotting."
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
                    f"OLS Filter Active "
                    f"(pVal>{BETA_P_VALUE_THRESHOLD:.2f} "
                    f"or R²<{MIN_OLS_R_SQUARED_THRESHOLD:.2f})"
                ),
            )
        ax.set_title("Spread Z-Score with Trading Signals", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Z-Score")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.7)

    # ---- Save combined figure ----
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "equity_spread_summary.png"))
    plt.close(fig)

    logging.info("Combined plot saved.")


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


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- START OF BACKTESTING SCRIPT ---")
    params = StrategyParameters()
    df_original = load_data(DATA_FILEPATH, params)
    if df_original is not None and not df_original.empty:
        # Use OOP wrapper PairTradingBacktester
        backtester = PairTradingBacktester(params)
        df_processed, equity_curve, trades_summary_df = backtester.run(df_original)
        trades_df_for_metrics = (
            trades_summary_df if trades_summary_df is not None else pd.DataFrame()
        )
        if equity_curve is not None and not equity_curve.empty:
            if (
                trades_summary_df is not None and not trades_summary_df.empty
            ):  # Ensure it is not empty
                # Convert datetime columns to string BEFORE saving CSV and BEFORE calculating metrics if they expect strings
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
                logging.info(f"Trades summary saved to: {TRADES_CSV_FILE}")
                # For metrics and plots, use the DataFrame with datetimes if necessary, or the copy (trades_df_for_metrics)
                # calculate_performance_metrics expects trades_df, which here is trades_df_for_metrics (the original copy of the backtest output)
                # plot_results also expects trades_df with datetimes for the scatter plot
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
                    "No trades were generated to save to CSV or calculate metrics."
                )
                # Still, try to plot the equity curve if it exists
                plot_results(df_processed, equity_curve, pd.DataFrame(), PLOT_DIR)

        else:
            logging.error("Backtest did not produce a valid equity curve.")
    else:
        logging.error("Could not load data. Terminating script.")
    logging.info("--- END OF BACKTESTING SCRIPT ---")
