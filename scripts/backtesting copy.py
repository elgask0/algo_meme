"""
Pair Trading Backtesting Framework
==================================
A comprehensive backtesting system for statistical arbitrage pair trading strategies
with rolling OLS regression, z-score signals, and advanced risk management.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import logging
import os
from datetime import timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, Any
from pathlib import Path


# === Configuration ===
@dataclass
class Config:
    """Central configuration for paths and directories"""
    base_dir: Path = Path("/Users/elgask0/REPOS/algo_meme")
    data_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    plot_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.log_dir = self.base_dir / "logs"
        self.plot_dir = self.base_dir / "plots"
        
        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
    
    @property
    def spread_data_path(self) -> Path:
        return self.data_dir / "spread_data.csv"
    
    @property
    def funding_data_path(self) -> Path:
        return self.data_dir / "funding_rate_history.csv"
    
    @property
    def log_file_path(self) -> Path:
        return self.log_dir / "backtest_run.log"
    
    @property
    def trades_csv_path(self) -> Path:
        return self.log_dir / "trades_summary.csv"


@dataclass
class StrategyParameters:
    """All tunable strategy parameters in one place"""
    # Asset configuration
    asset_y: str = "MEXCFTS_PERP_SPX_USDT"
    asset_x: str = "MEXCFTS_PERP_GIGA_USDT"
    
    # Rolling window parameters
    ols_window_days: int = 30
    zscore_window_days: int = 15
    
    # Statistical thresholds
    beta_p_value_threshold: float = 0.05
    min_ols_r_squared: float = 0.6
    beta_min: float = 0.3
    beta_max: float = 3.0
    beta_roll_std_max: float = 0.25
    
    # Trading signals
    entry_zscore: float = 1.75
    exit_zscore: float = 0.25
    stop_loss_zscore: float = 4.0
    
    # Position sizing and risk management
    position_size_pct: float = 0.5
    min_nominal_per_leg: float = 5.0
    pnl_stop_loss_pct: float = 0.1
    trailing_stop_pct: float = 1
    sl_cooldown_hours: int = 24
    
    # Trading costs
    taker_fee_pct: float = 0.0002
    slippage_pct: float = 0.01
    
    # Capital and data
    initial_capital: float = 2000.0
    risk_free_rate_annual: float = 0.02
    resample_minutes: Optional[int] = 15
    default_data_freq_minutes: int = 5


# === Logging Setup ===
def setup_logging(log_file: Path):
    """Configure logging with both file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ],
    )


# === Data Loading and Preparation ===
class DataLoader:
    """Handles all data loading and preprocessing"""
    
    def __init__(self, config: Config, params: StrategyParameters):
        self.config = config
        self.params = params
    
    def load_price_data(self) -> Optional[pd.DataFrame]:
        """Load and prepare price data"""
        logging.info(f"Loading data from: {self.config.spread_data_path}")
        
        try:
            df = pd.read_csv(self.config.spread_data_path)
            df["ts_start"] = pd.to_datetime(df["ts_start"])
            df.set_index("ts_start", inplace=True)
            df.sort_index(inplace=True)
            
            # Drop rows with missing prices
            df.dropna(subset=[self.params.asset_y, self.params.asset_x], inplace=True)
            
            if df.empty:
                logging.error("No data after removing NaNs")
                return None
            
            logging.info(
                f"Data loaded: {df.index.min()} to {df.index.max()}, {len(df)} rows"
            )
            
            # Add funding rates if available
            df = self._add_funding_rates(df)
            
            # Resample if requested
            if self.params.resample_minutes:
                df = self._resample_data(df)
            
            return df
            
        except FileNotFoundError:
            logging.error(f"Data file not found: {self.config.spread_data_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None
    
    def _add_funding_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add funding rates to the dataframe"""
        try:
            funding = pd.read_csv(self.config.funding_data_path, parse_dates=["ts"])
            funding.set_index("ts", inplace=True)
            
            # Extract funding rates for each asset
            fr_y = funding[funding["symbol"] == self.params.asset_y][["funding_rate"]]
            fr_x = funding[funding["symbol"] == self.params.asset_x][["funding_rate"]]
            
            # Rename columns
            fr_y.columns = ["funding_rate_y"]
            fr_x.columns = ["funding_rate_x"]
            
            # Join and fill missing values with 0
            df = df.join(fr_y, how="left").join(fr_x, how="left")
            df.fillna({"funding_rate_y": 0.0, "funding_rate_x": 0.0}, inplace=True)
            
        except Exception as e:
            logging.warning(f"Could not load funding rates: {e}")
            df["funding_rate_y"] = 0.0
            df["funding_rate_x"] = 0.0
        
        return df
    
    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to specified frequency"""
        logging.info(f"Resampling data to {self.params.resample_minutes} minutes")
        
        df = df.resample(f"{self.params.resample_minutes}min").last()
        df.dropna(subset=[self.params.asset_y, self.params.asset_x], inplace=True)
        
        logging.info(
            f"Data resampled: {df.index.min()} to {df.index.max()}, {len(df)} rows"
        )
        
        return df


# === Statistical Calculations ===
class StatisticalEngine:
    """Handles rolling OLS regression and statistical calculations"""
    
    def __init__(self, params: StrategyParameters):
        self.params = params
    
    def calculate_signals(self, df: pd.DataFrame, periods_per_day: float) -> pd.DataFrame:
        """Calculate all statistical signals"""
        # Convert days to periods
        ols_periods = int(self.params.ols_window_days * periods_per_day)
        zscore_periods = int(self.params.zscore_window_days * periods_per_day)
        
        # Calculate rolling OLS and spread
        df = self._calculate_rolling_ols(df, ols_periods)
        
        # Calculate z-score
        df["z_score"] = self._calculate_zscore(df["spread"], zscore_periods)
        
        # Drop rows with incomplete calculations
        df.dropna(subset=["alpha", "beta", "z_score"], inplace=True)
        
        return df
    
    def _calculate_rolling_ols(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling OLS with in-sample normalization"""
        # Log prices
        log_y = np.log(df[self.params.asset_y])
        log_x = np.log(df[self.params.asset_x])
        
        # Rolling normalization (z-score within each window)
        log_y_norm = self._rolling_normalize(log_y, window)
        log_x_norm = self._rolling_normalize(log_x, window)
        
        # Prepare exogenous variables
        exog = pd.DataFrame({f"{self.params.asset_x}_std": log_x_norm})
        exog = sm.add_constant(exog, prepend=False)
        
        # Rolling OLS
        model = RollingOLS(log_y_norm, exog, window=window)
        results = model.fit()
        
        # Extract results
        params_df = pd.DataFrame(results.params, index=df.index)
        pvalues_df = pd.DataFrame(results.pvalues, index=df.index, columns=params_df.columns)
        
        df["alpha"] = params_df["const"]
        df["beta"] = params_df[f"{self.params.asset_x}_std"]
        df["beta_p_value"] = pvalues_df[f"{self.params.asset_x}_std"]
        df["ols_r_squared"] = pd.Series(results.rsquared, index=df.index)
        
        # Calculate spread (residual)
        df["spread"] = log_y_norm - (df["alpha"] + df["beta"] * log_x_norm)
        
        # Beta quality metrics
        df["beta_roll_std_20"] = df["beta"].rolling(20, min_periods=20).std()
        df["beta_ok"] = (
            df["beta"].abs().between(self.params.beta_min, self.params.beta_max) & 
            (df["beta_roll_std_20"] < self.params.beta_roll_std_max)
        )
        
        return df
    
    @staticmethod
    def _rolling_normalize(series: pd.Series, window: int) -> pd.Series:
        """Normalize series to mean=0, std=1 within rolling window"""
        roll_mean = series.rolling(window, min_periods=window).mean()
        roll_std = series.rolling(window, min_periods=window).std(ddof=0)
        return (series - roll_mean) / roll_std
    
    @staticmethod
    def _calculate_zscore(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std()
        return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


# === Position Management ===
@dataclass
class Position:
    """Represents an open trading position"""
    type: str  # "long_spread" or "short_spread"
    entry_time: pd.Timestamp
    entry_zscore: float
    entry_spread: float
    entry_alpha: float
    entry_beta: float
    
    # Quantities and prices
    qty_y: float
    qty_x: float
    entry_price_y: float
    entry_price_x: float
    exec_price_y: float
    exec_price_x: float
    
    # Risk management
    is_x_long: bool
    equity_base: float
    entry_fees: float
    peak_pnl: float = 0.0
    
    def calculate_pnl(self, current_price_y: float, current_price_x: float) -> Tuple[float, float, float]:
        """Calculate current P&L (gross and per leg)"""
        # Y leg P&L
        if self.type == "long_spread":
            pnl_y = (current_price_y - self.exec_price_y) * self.qty_y
        else:
            pnl_y = (self.exec_price_y - current_price_y) * self.qty_y
        
        # X leg P&L
        if self.is_x_long:
            pnl_x = (current_price_x - self.exec_price_x) * self.qty_x
        else:
            pnl_x = (self.exec_price_x - current_price_x) * self.qty_x
        
        return pnl_y + pnl_x, pnl_y, pnl_x


# === Trading Engine ===
class TradingEngine:
    """Core trading logic and position management"""
    
    def __init__(self, params: StrategyParameters):
        self.params = params
        self.position: Optional[Position] = None
        self.equity = params.initial_capital
        self.trades = []
        self.equity_curve = pd.Series(dtype=float)
        self.sl_cooldown_until: Optional[pd.Timestamp] = None
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Execute the backtest"""
        logging.info("Starting backtest...")
        
        # Calculate data frequency
        freq_minutes = self._get_data_frequency(df)
        periods_per_day = (24 * 60) / freq_minutes
        
        # Calculate signals
        stats_engine = StatisticalEngine(self.params)
        df = stats_engine.calculate_signals(df, periods_per_day)
        
        if df.empty:
            logging.error("No valid data after calculating signals")
            return self.equity_curve, pd.DataFrame()
        
        # Main trading loop
        for timestamp, row in df.iterrows():
            self._process_bar(timestamp, row)
        
        # Close any remaining position
        if self.position:
            self._close_position(df.iloc[-1], "End of Backtest")
        
        logging.info("Backtest completed")
        return self.equity_curve.dropna(), pd.DataFrame(self.trades)
    
    def _process_bar(self, timestamp: pd.Timestamp, row: pd.Series):
        """Process a single bar of data"""
        # Update equity curve with current portfolio value
        portfolio_value = self._calculate_portfolio_value(row)
        self.equity_curve[timestamp] = portfolio_value
        
        # Apply funding costs if position is open
        if self.position:
            self._apply_funding_costs(row)
        
        # Check if in cooldown
        if self.sl_cooldown_until and timestamp < self.sl_cooldown_until:
            return
        elif self.sl_cooldown_until and timestamp >= self.sl_cooldown_until:
            logging.debug(f"{timestamp}: Cooldown ended")
            self.sl_cooldown_until = None
        
        # Skip if data is invalid
        if self._is_data_invalid(row):
            return
        
        # Manage existing position or look for new entry
        if self.position:
            self._manage_position(timestamp, row)
        else:
            self._check_entry_signal(timestamp, row)
    
    def _calculate_portfolio_value(self, row: pd.Series) -> float:
        """Calculate current portfolio value including unrealized P&L"""
        value = self.equity
        
        if self.position:
            gross_pnl, _, _ = self.position.calculate_pnl(
                row[self.params.asset_y], 
                row[self.params.asset_x]
            )
            value += gross_pnl
        
        return value
    
    def _apply_funding_costs(self, row: pd.Series):
        """Apply funding rate costs to open position"""
        # Y leg funding
        fr_y = row.get("funding_rate_y", 0.0)
        if fr_y:
            dir_y = 1 if self.position.type == "long_spread" else -1
            notional_y = self.position.qty_y * row[self.params.asset_y]
            funding_y = fr_y * notional_y * dir_y
            self.equity -= funding_y
        
        # X leg funding
        fr_x = row.get("funding_rate_x", 0.0)
        if fr_x:
            dir_x = 1 if self.position.is_x_long else -1
            notional_x = self.position.qty_x * row[self.params.asset_x]
            funding_x = fr_x * notional_x * dir_x
            self.equity -= funding_x
    
    def _is_data_invalid(self, row: pd.Series) -> bool:
        """Check if current bar data is valid"""
        return any(pd.isna(row[col]) for col in ["z_score", "beta", "alpha", "beta_p_value", "ols_r_squared"])
    
    def _manage_position(self, timestamp: pd.Timestamp, row: pd.Series):
        """Manage existing position - check exits"""
        exit_signal, exit_reason = self._check_exit_conditions(row)
        
        if exit_signal:
            self._close_position(row, exit_reason)
            
            # Set cooldown if stop loss
            if "Stop Loss" in exit_reason:
                self.sl_cooldown_until = timestamp + timedelta(hours=self.params.sl_cooldown_hours)
                logging.warning(f"{timestamp}: Stop loss activated, cooldown until {self.sl_cooldown_until}")
    
    def _check_exit_conditions(self, row: pd.Series) -> Tuple[bool, str]:
        """Check all exit conditions"""
        z_score = row["z_score"]
        
        # P&L stop loss
        if self.params.pnl_stop_loss_pct > 0:
            gross_pnl, _, _ = self.position.calculate_pnl(
                row[self.params.asset_y], 
                row[self.params.asset_x]
            )
            net_pnl = gross_pnl - self.position.entry_fees
            max_loss = self.position.equity_base * self.params.pnl_stop_loss_pct
            
            if net_pnl < -max_loss:
                return True, f"P&L Stop Loss ({net_pnl:.2f} < -{max_loss:.2f})"
            
            # Trailing stop
            if net_pnl > 0:
                if net_pnl > self.position.peak_pnl:
                    self.position.peak_pnl = net_pnl
                elif (self.position.peak_pnl - net_pnl) > self.position.peak_pnl * self.params.trailing_stop_pct:
                    return True, f"Trailing Stop ({net_pnl:.2f} < {self.position.peak_pnl * (1 - self.params.trailing_stop_pct):.2f})"
        
        # Z-score exit
        if self.position.type == "long_spread":
            if z_score >= -self.params.exit_zscore:
                return True, f"Z-Score Reversion ({z_score:.2f})"
            elif z_score <= -self.params.stop_loss_zscore:
                return True, f"Z-Score Stop Loss ({z_score:.2f})"
        else:  # short_spread
            if z_score <= self.params.exit_zscore:
                return True, f"Z-Score Reversion ({z_score:.2f})"
            elif z_score >= self.params.stop_loss_zscore:
                return True, f"Z-Score Stop Loss ({z_score:.2f})"
        
        return False, ""
    
    def _check_entry_signal(self, timestamp: pd.Timestamp, row: pd.Series):
        """Check for entry signals"""
        z_score = row["z_score"]
        
        # Check z-score thresholds
        long_signal = -self.params.stop_loss_zscore < z_score < -self.params.entry_zscore
        short_signal = self.params.entry_zscore < z_score < self.params.stop_loss_zscore
        
        if not (long_signal or short_signal):
            return
        
        # Check statistical quality
        if not self._is_signal_quality_ok(row):
            return
        
        # Calculate position size
        position_info = self._calculate_position_size(row)
        if not position_info:
            return
        
        # Open position
        position_type = "long_spread" if long_signal else "short_spread"
        self._open_position(timestamp, row, position_type, position_info)
    
    def _is_signal_quality_ok(self, row: pd.Series) -> bool:
        """Check if signal meets quality criteria"""
        if row["beta_p_value"] >= self.params.beta_p_value_threshold:
            logging.debug(f"Skipping: beta p-value {row['beta_p_value']:.4f} >= {self.params.beta_p_value_threshold}")
            return False
        
        if row["ols_r_squared"] < self.params.min_ols_r_squared:
            logging.debug(f"Skipping: R² {row['ols_r_squared']:.4f} < {self.params.min_ols_r_squared}")
            return False
        
        if not row.get("beta_ok", False):
            logging.debug(f"Skipping: beta quality check failed")
            return False
        
        return True
    
    def _calculate_position_size(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate position sizes for both legs"""
        trade_nominal = self.equity * self.params.position_size_pct
        beta = abs(row["beta"])
        
        if (1 + beta) == 0:
            logging.warning("Invalid beta for position sizing")
            return None
        
        nominal_y = trade_nominal / (1 + beta)
        nominal_x = trade_nominal * beta / (1 + beta)
        
        if nominal_y < self.params.min_nominal_per_leg or nominal_x < self.params.min_nominal_per_leg:
            logging.debug(f"Position too small: Y={nominal_y:.2f}, X={nominal_x:.2f}")
            return None
        
        return {
            "nominal_y": nominal_y,
            "nominal_x": nominal_x,
            "beta": row["beta"]
        }
    
    def _open_position(self, timestamp: pd.Timestamp, row: pd.Series, 
                      position_type: str, position_info: Dict[str, float]):
        """Open a new position"""
        # Determine trade direction
        is_x_long = self._determine_x_direction(position_type, position_info["beta"])
        
        # Calculate execution prices with slippage
        price_y = row[self.params.asset_y]
        price_x = row[self.params.asset_x]
        
        if position_type == "long_spread":
            exec_price_y = self._apply_slippage(price_y, "buy")
            exec_price_x = self._apply_slippage(price_x, "buy" if is_x_long else "sell")
        else:
            exec_price_y = self._apply_slippage(price_y, "sell")
            exec_price_x = self._apply_slippage(price_x, "sell" if not is_x_long else "buy")
        
        # Calculate quantities
        qty_y = position_info["nominal_y"] / exec_price_y
        qty_x = position_info["nominal_x"] / exec_price_x
        
        # Calculate costs
        fee_y = abs(exec_price_y * qty_y * self.params.taker_fee_pct)
        fee_x = abs(exec_price_x * qty_x * self.params.taker_fee_pct)
        total_fees = fee_y + fee_x
        
        # Update equity
        self.equity -= total_fees
        
        # Create position
        self.position = Position(
            type=position_type,
            entry_time=timestamp,
            entry_zscore=row["z_score"],
            entry_spread=row["spread"],
            entry_alpha=row["alpha"],
            entry_beta=position_info["beta"],
            qty_y=qty_y,
            qty_x=qty_x,
            entry_price_y=price_y,
            entry_price_x=price_x,
            exec_price_y=exec_price_y,
            exec_price_x=exec_price_x,
            is_x_long=is_x_long,
            equity_base=self.equity + total_fees,  # Equity before fees
            entry_fees=total_fees
        )
        
        # Log trade entry
        self._log_trade_entry(timestamp, row, total_fees)
        
        logging.info(
            f"TRADE OPENED: {position_type} | Z: {row['z_score']:.2f} | "
            f"Beta: {position_info['beta']:.4f} | Equity after fees: {self.equity:.2f}"
        )
    
    def _close_position(self, row: pd.Series, exit_reason: str):
        """Close current position"""
        # Calculate execution prices
        price_y = row[self.params.asset_y]
        price_x = row[self.params.asset_x]
        
        if self.position.type == "long_spread":
            exec_price_y = self._apply_slippage(price_y, "sell")
        else:
            exec_price_y = self._apply_slippage(price_y, "buy")
        
        if self.position.is_x_long:
            exec_price_x = self._apply_slippage(price_x, "sell")
        else:
            exec_price_x = self._apply_slippage(price_x, "buy")
        
        # Calculate P&L
        gross_pnl, pnl_y, pnl_x = self.position.calculate_pnl(price_y, price_x)
        
        # Calculate exit fees
        fee_y = abs(exec_price_y * self.position.qty_y * self.params.taker_fee_pct)
        fee_x = abs(exec_price_x * self.position.qty_x * self.params.taker_fee_pct)
        total_fees = fee_y + fee_x
        
        # Update equity
        self.equity += gross_pnl - total_fees
        net_pnl = gross_pnl - self.position.entry_fees - total_fees
        
        # Update trade log
        self._update_trade_exit(row, exit_reason, exec_price_y, exec_price_x, 
                               pnl_y, pnl_x, total_fees, net_pnl)
        
        logging.info(
            f"TRADE CLOSED: {self.position.type} | Reason: {exit_reason} | "
            f"Net P&L: {net_pnl:.2f} | Equity: {self.equity:.2f}"
        )
        
        # Clear position
        self.position = None
    
    def _log_trade_entry(self, timestamp: pd.Timestamp, row: pd.Series, entry_fees: float):
        """Log trade entry details"""
        self.trades.append({
            "entry_time": timestamp,
            "exit_time": None,
            "position_type": self.position.type,
            "entry_zscore": self.position.entry_zscore,
            "exit_zscore": None,
            "reason_exit": None,
            "asset_y": self.params.asset_y,
            "asset_x": self.params.asset_x,
            "qty_y": self.position.qty_y,
            "qty_x": self.position.qty_x,
            "entry_price_y": self.position.entry_price_y,
            "entry_price_x": self.position.entry_price_x,
            "exec_price_y_entry": self.position.exec_price_y,
            "exec_price_x_entry": self.position.exec_price_x,
            "entry_beta": self.position.entry_beta,
            "entry_fees": entry_fees,
            "is_x_long": self.position.is_x_long,
        })
    
    def _update_trade_exit(self, row: pd.Series, exit_reason: str, 
                          exec_price_y: float, exec_price_x: float,
                          pnl_y: float, pnl_x: float, exit_fees: float, net_pnl: float):
        """Update trade log with exit details"""
        # Find the open trade
        for trade in reversed(self.trades):
            if trade["exit_time"] is None:
                trade.update({
                    "exit_time": row.name,
                    "exit_zscore": row["z_score"],
                    "reason_exit": exit_reason,
                    "exit_price_y": row[self.params.asset_y],
                    "exit_price_x": row[self.params.asset_x],
                    "exec_price_y_exit": exec_price_y,
                    "exec_price_x_exit": exec_price_x,
                    "pnl_y": pnl_y,
                    "pnl_x": pnl_x,
                    "exit_fees": exit_fees,
                    "total_fees": trade["entry_fees"] + exit_fees,
                    "gross_pnl": pnl_y + pnl_x,
                    "net_pnl": net_pnl,
                    "final_equity": self.equity
                })
                break
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price"""
        if side == "buy":
            return price * (1 + self.params.slippage_pct)
        else:
            return price * (1 - self.params.slippage_pct)
    
    @staticmethod
    def _determine_x_direction(position_type: str, beta: float) -> bool:
        """Determine if X leg should be long"""
        if position_type == "long_spread":
            return beta < 0
        else:
            return beta > 0
    
    @staticmethod
    def _get_data_frequency(df: pd.DataFrame) -> float:
        """Estimate data frequency in minutes"""
        if len(df) < 2:
            return 5.0  # Default
        
        # Calculate median time difference
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        return median_diff.total_seconds() / 60


# === Performance Analytics ===
class PerformanceAnalyzer:
    """Calculate performance metrics and generate reports"""
    
    def __init__(self, params: StrategyParameters):
        self.params = params
    
    def calculate_metrics(self, equity_curve: pd.Series, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if equity_curve.empty:
            logging.warning("Empty equity curve, cannot calculate metrics")
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics["initial_equity"] = self.params.initial_capital
        metrics["final_equity"] = equity_curve.iloc[-1]
        metrics["total_return_pct"] = ((metrics["final_equity"] / metrics["initial_equity"]) - 1) * 100
        
        # Time-based metrics
        duration_years = self._calculate_duration_years(equity_curve)
        metrics["cagr_pct"] = (((metrics["final_equity"] / metrics["initial_equity"]) ** (1 / duration_years)) - 1) * 100 if duration_years > 0 else 0
        
        # Risk metrics
        drawdown_series = self._calculate_drawdown(equity_curve)
        metrics["max_drawdown_pct"] = drawdown_series.min() * 100
        
        # Risk-adjusted returns
        daily_returns = equity_curve.resample("D").last().pct_change().dropna()
        if not daily_returns.empty:
            metrics["sharpe_ratio"] = self._calculate_sharpe(daily_returns)
            metrics["sortino_ratio"] = self._calculate_sortino(daily_returns)
            metrics["calmar_ratio"] = metrics["cagr_pct"] / abs(metrics["max_drawdown_pct"]) if metrics["max_drawdown_pct"] != 0 else 0
        
        # Trade statistics
        if not trades_df.empty and "net_pnl" in trades_df.columns:
            metrics.update(self._calculate_trade_stats(trades_df))
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def _calculate_duration_years(self, equity_curve: pd.Series) -> float:
        """Calculate strategy duration in years"""
        duration = equity_curve.index[-1] - equity_curve.index[0]
        years = duration.total_seconds() / (365.25 * 24 * 60 * 60)
        return max(years, 1/365.25)  # Minimum 1 day
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = equity_curve.cummax()
        return equity_curve / rolling_max - 1.0
    
    def _calculate_sharpe(self, daily_returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if daily_returns.std() == 0:
            return 0
        
        rf_daily = (1 + self.params.risk_free_rate_annual) ** (1/252) - 1
        excess_returns = daily_returns - rf_daily
        return excess_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    def _calculate_sortino(self, daily_returns: pd.Series) -> float:
        """Calculate annualized Sortino ratio"""
        negative_returns = daily_returns[daily_returns < 0]
        if negative_returns.empty or negative_returns.std() == 0:
            return 0
        
        rf_daily = (1 + self.params.risk_free_rate_annual) ** (1/252) - 1
        excess_returns = daily_returns - rf_daily
        return excess_returns.mean() / negative_returns.std() * np.sqrt(252)
    
    def _calculate_trade_stats(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-specific statistics"""
        stats = {}
        
        # Filter completed trades
        completed_trades = trades_df[trades_df["exit_time"].notna()].copy()
        if completed_trades.empty:
            return stats
        
        stats["num_trades"] = len(completed_trades)
        
        # Win/loss analysis
        winning_trades = completed_trades[completed_trades["net_pnl"] > 0]
        losing_trades = completed_trades[completed_trades["net_pnl"] <= 0]
        
        stats["win_rate_pct"] = (len(winning_trades) / len(completed_trades)) * 100 if len(completed_trades) > 0 else 0
        stats["avg_win"] = winning_trades["net_pnl"].mean() if not winning_trades.empty else 0
        stats["avg_loss"] = losing_trades["net_pnl"].mean() if not losing_trades.empty else 0
        
        # Profit factor
        gross_wins = winning_trades["net_pnl"].sum() if not winning_trades.empty else 0
        gross_losses = abs(losing_trades["net_pnl"].sum()) if not losing_trades.empty else 0
        stats["profit_factor"] = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Total fees
        stats["total_fees"] = completed_trades["total_fees"].sum() if "total_fees" in completed_trades else 0
        
        return stats
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        logging.info("=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                if "pct" in key or "ratio" in key:
                    logging.info(f"{key}: {value:.2f}")
                else:
                    logging.info(f"{key}: ${value:,.2f}")
            else:
                logging.info(f"{key}: {value}")


# === Visualization ===
class Visualizer:
    """Generate charts and visualizations"""
    
    def __init__(self, params: StrategyParameters, plot_dir: Path):
        self.params = params
        self.plot_dir = plot_dir
        plt.style.use("seaborn-v0_8-darkgrid")
    
    def create_charts(self, df: pd.DataFrame, equity_curve: pd.Series, trades_df: pd.DataFrame):
        """Create all visualization charts"""
        if equity_curve.empty:
            logging.warning("Empty equity curve, skipping charts")
            return
        
        # Create combined figure
        fig, (ax_equity, ax_zscore) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # Plot equity curve
        self._plot_equity_curve(ax_equity, equity_curve)
        
        # Plot z-score and trades
        if not df.empty and "z_score" in df.columns:
            self._plot_zscore_and_trades(ax_zscore, df, trades_df)
        
        # Save figure
        plt.tight_layout()
        output_path = self.plot_dir / "backtest_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Charts saved to {output_path}")
    
    def _plot_equity_curve(self, ax: plt.Axes, equity_curve: pd.Series):
        """Plot equity curve"""
        ax.plot(equity_curve.index, equity_curve.values, color="blue", linewidth=2, label="Equity")
        
        # Add drawdown shading
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        
        ax.fill_between(
            equity_curve.index,
            equity_curve.values,
            rolling_max.values,
            where=(equity_curve < rolling_max),
            color="red",
            alpha=0.2,
            label="Drawdown"
        )
        
        ax.set_title("Equity Curve", fontsize=16, fontweight="bold")
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Add performance text
        initial = equity_curve.iloc[0]
        final = equity_curve.iloc[-1]
        total_return = (final / initial - 1) * 100
        max_dd = drawdown.min()
        
        text = f"Return: {total_return:.1f}%\nMax DD: {max_dd:.1f}%"
        ax.text(0.02, 0.95, text, transform=ax.transAxes, 
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    def _plot_zscore_and_trades(self, ax: plt.Axes, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Plot z-score with trade markers"""
        # Plot z-score
        ax.plot(df.index, df["z_score"], color="dodgerblue", linewidth=1.5, alpha=0.9, label="Z-Score")
        
        # Add threshold lines
        ax.axhline(self.params.entry_zscore, color="green", linestyle="--", alpha=0.7, label=f"Entry ±{self.params.entry_zscore}")
        ax.axhline(-self.params.entry_zscore, color="green", linestyle="--", alpha=0.7)
        ax.axhline(self.params.exit_zscore, color="orange", linestyle=":", alpha=0.7, label=f"Exit ±{self.params.exit_zscore}")
        ax.axhline(-self.params.exit_zscore, color="orange", linestyle=":", alpha=0.7)
        ax.axhline(self.params.stop_loss_zscore, color="red", linestyle="-.", alpha=0.7, label=f"Stop ±{self.params.stop_loss_zscore}")
        ax.axhline(-self.params.stop_loss_zscore, color="red", linestyle="-.", alpha=0.7)
        
        # Plot trade markers
        if not trades_df.empty:
            self._add_trade_markers(ax, df, trades_df)
        
        # Add filter regions
        self._add_filter_regions(ax, df)
        
        ax.set_title("Z-Score and Trading Signals", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Z-Score")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    def _add_trade_markers(self, ax: plt.Axes, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Add trade entry/exit markers"""
        # Entry markers
        for _, trade in trades_df.iterrows():
            if pd.notna(trade["entry_time"]) and trade["entry_time"] in df.index:
                z_val = df.loc[trade["entry_time"], "z_score"]
                color = "green" if trade["position_type"] == "long_spread" else "red"
                marker = "^" if trade["position_type"] == "long_spread" else "v"
                ax.scatter(trade["entry_time"], z_val, color=color, marker=marker, 
                          s=100, edgecolors="black", linewidth=1, zorder=5)
        
        # Exit markers
        completed_trades = trades_df[trades_df["exit_time"].notna()]
        for _, trade in completed_trades.iterrows():
            if trade["exit_time"] in df.index:
                z_val = df.loc[trade["exit_time"], "z_score"]
                ax.scatter(trade["exit_time"], z_val, color="purple", marker="o", 
                          s=80, edgecolors="black", linewidth=1, zorder=5)
    
    def _add_filter_regions(self, ax: plt.Axes, df: pd.DataFrame):
        """Add shading for filtered regions"""
        # OLS filter conditions
        filter_active = (
            (df["beta_p_value"] >= self.params.beta_p_value_threshold) |
            (df["ols_r_squared"] < self.params.min_ols_r_squared) |
            (~df["beta_ok"])
        )
        
        if filter_active.any():
            y_min, y_max = ax.get_ylim()
            ax.fill_between(
                df.index,
                y_min,
                y_max,
                where=filter_active,
                color="lightcoral",
                alpha=0.2,
                label="Filter Active"
            )


# === Main Backtester Class ===
class PairTradingBacktester:
    """Main interface for running backtests"""
    
    def __init__(self, params: StrategyParameters = None):
        self.params = params or StrategyParameters()
        self.config = Config()
        setup_logging(self.config.log_file_path)
    
    def run(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Run complete backtest workflow"""
        # Load data if not provided
        if df is None:
            loader = DataLoader(self.config, self.params)
            df = loader.load_price_data()
            if df is None:
                logging.error("Failed to load data")
                return {}
        
        # Run backtest
        engine = TradingEngine(self.params)
        equity_curve, trades_df = engine.run_backtest(df)
        
        # Calculate metrics
        analyzer = PerformanceAnalyzer(self.params)
        metrics = analyzer.calculate_metrics(equity_curve, trades_df)
        
        # Generate visualizations
        visualizer = Visualizer(self.params, self.config.plot_dir)
        visualizer.create_charts(df, equity_curve, trades_df)
        
        # Save trade log
        if not trades_df.empty:
            self._save_trades(trades_df)
        
        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades_df
        }
    
    def _save_trades(self, trades_df: pd.DataFrame):
        """Save trades to CSV"""
        # Convert timestamps to strings for CSV
        for col in ["entry_time", "exit_time"]:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        trades_df.to_csv(self.config.trades_csv_path, index=False)
        logging.info(f"Trades saved to {self.config.trades_csv_path}")


# === Walk-Forward Analysis ===
class WalkForwardAnalyzer:
    """Perform walk-forward analysis"""
    
    def __init__(self, params: StrategyParameters):
        self.params = params
    
    def run_analysis(self, df: pd.DataFrame, train_days: int, test_days: int) -> Tuple[pd.Series, pd.DataFrame]:
        """Run walk-forward backtest"""
        all_trades = []
        equity_segments = []
        
        start_date = df.index.min()
        end_date = df.index.max()
        
        current_date = start_date
        current_equity = self.params.initial_capital
        
        while current_date < end_date:
            # Define window
            train_end = current_date + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            if train_end >= end_date:
                break
            
            # Get data slice
            window_data = df[current_date:test_end]
            if window_data.empty:
                current_date = train_end
                continue
            
            # Run backtest on this window
            params_copy = StrategyParameters(**asdict(self.params))
            params_copy.initial_capital = current_equity
            
            engine = TradingEngine(params_copy)
            equity_curve, trades = engine.run_backtest(window_data)
            
            # Extract test period results
            test_equity = equity_curve[train_end:]
            if not test_equity.empty:
                equity_segments.append(test_equity)
                current_equity = test_equity.iloc[-1]
            
            if not trades.empty:
                all_trades.append(trades)
            
            current_date = train_end
        
        # Combine results
        combined_equity = pd.concat(equity_segments) if equity_segments else pd.Series()
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        return combined_equity, combined_trades


# === Entry Point ===
def main():
    """Main entry point"""
    logging.info("=== Pair Trading Backtest Started ===")
    
    # Initialize backtester with default parameters
    backtester = PairTradingBacktester()
    
    # Run backtest
    results = backtester.run()
    
    if results:
        logging.info("=== Backtest Completed Successfully ===")
    else:
        logging.error("=== Backtest Failed ===")


if __name__ == "__main__":
    main()