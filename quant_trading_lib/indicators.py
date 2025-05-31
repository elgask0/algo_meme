import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any

# Attempt to import from local data_management module
try:
    # This assumes data_management.py is in the same directory or package
    from .data_management import load_cleaned_ohlcv_data
except ImportError:
    logger = logging.getLogger(__name__) # Define logger here if import fails for standalone use
    logger.warning("Could not perform relative import of load_cleaned_ohlcv_data from .data_management. Using MOCK.")
    def load_cleaned_ohlcv_data(db_path: str, symbol: str,
                                start_date_iso: Optional[str] = None,
                                end_date_iso: Optional[str] = None) -> pd.DataFrame:
        logger.warning("Using MOCK load_cleaned_ohlcv_data. Ensure data_management module is correctly installed/imported.")
        # Return an empty DataFrame with expected columns for the mock
        return pd.DataFrame(columns=['time_period_start', 'price_close'])


# --- Logging Configuration ---
# Ensure logger is defined globally for the module if not done in except block
if 'logger' not in globals():
    logger = logging.getLogger(__name__)

# --- VWAP and Mark Price Calculation Functions ---

def calculate_snapshot_vwap_and_depth(df_orderbook_snapshots: pd.DataFrame, symbol_id_context: str) -> pd.DataFrame:
    """
    Calculates Volume Weighted Average Price (VWAP) and total depth for each order book snapshot
    up to 3 levels.
    Based on calculate_vwap_and_depth from scripts/compute_mark_price_vwap.py.

    Args:
        df_orderbook_snapshots (pd.DataFrame): DataFrame with order book snapshots.
                                               Expected columns: 'ts', 'bid1_px', 'bid1_sz', ..., 'ask3_sz'.
        symbol_id_context (str): Symbol ID for logging context.

    Returns:
        pd.DataFrame: DataFrame with columns: 'ts', 'vwap', 'depth_sum_sz'.
                      Returns an empty DataFrame with these columns if input is empty or invalid.
    """
    logger.debug(f"[{symbol_id_context}] Calculating VWAP and depth for {len(df_orderbook_snapshots)} snapshots.")
    if df_orderbook_snapshots.empty:
        logger.warning(f"[{symbol_id_context}] Input DataFrame is empty. Cannot calculate VWAP/depth.")
        return pd.DataFrame(columns=["ts", "vwap", "depth_sum_sz"])

    df_calc = df_orderbook_snapshots.copy()

    if 'ts' not in df_calc.columns:
        logger.error(f"[{symbol_id_context}] 'ts' column missing in input DataFrame.")
        return pd.DataFrame(columns=["ts", "vwap", "depth_sum_sz"])

    bid_px_cols = [f"bid{i}_px" for i in range(1, 4)]
    bid_sz_cols = [f"bid{i}_sz" for i in range(1, 4)]
    ask_px_cols = [f"ask{i}_px" for i in range(1, 4)]
    ask_sz_cols = [f"ask{i}_sz" for i in range(1, 4)]
    all_level_cols = bid_px_cols + bid_sz_cols + ask_px_cols + ask_sz_cols

    for col in all_level_cols:
        if col not in df_calc.columns:
            df_calc[col] = 0.0
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)

    total_bid_value = np.sum([df_calc[bp] * df_calc[bs] for bp, bs in zip(bid_px_cols, bid_sz_cols)], axis=0)
    total_ask_value = np.sum([df_calc[ap] * df_calc[as_] for ap, as_ in zip(ask_px_cols, ask_sz_cols)], axis=0)

    total_bid_size = df_calc[bid_sz_cols].sum(axis=1)
    total_ask_size = df_calc[ask_sz_cols].sum(axis=1)

    df_calc["total_value"] = total_bid_value + total_ask_value
    df_calc["total_size"] = total_bid_size + total_ask_size

    df_calc["vwap"] = np.where(df_calc["total_size"].abs() > 1e-9, df_calc["total_value"] / df_calc["total_size"], np.nan)
    df_calc["depth_sum_sz"] = df_calc["total_size"]

    num_valid_vwap = df_calc['vwap'].notna().sum()
    if num_valid_vwap == 0 and not df_calc.empty:
        logger.warning(f"[{symbol_id_context}] VWAP calculation resulted in all NaN values.")
    else:
        logger.debug(f"[{symbol_id_context}] VWAP and depth calculated. {num_valid_vwap} valid VWAPs.")

    return df_calc[["ts", "vwap", "depth_sum_sz"]]


def aggregate_mark_price(df_vwap_snapshots: pd.DataFrame, symbol_id: str, freq: str) -> pd.DataFrame:
    """
    Aggregates snapshot VWAP data to a given frequency to compute a mark price.
    Based on aggregate_vwap from scripts/compute_mark_price_vwap.py.

    Args:
        df_vwap_snapshots (pd.DataFrame): DataFrame from calculate_snapshot_vwap_and_depth.
                                          Expected columns: 'ts', 'vwap', 'depth_sum_sz'.
        symbol_id (str): Symbol ID for the output DataFrame.
        freq (str): Frequency string for aggregation (e.g., '5min', '1H').

    Returns:
        pd.DataFrame: DataFrame with columns: 'symbol_id', 'ts_start', 'ts_end',
                      'mark_price', 'depth_sum_sz', 'n_snapshots'.
                      Returns an empty DataFrame if input is unsuitable.
    """
    logger.info(f"[{symbol_id}] Aggregating VWAP snapshots to frequency {freq}...")
    if df_vwap_snapshots.empty or 'vwap' not in df_vwap_snapshots.columns:
        logger.warning(f"[{symbol_id}] VWAP snapshots DataFrame is empty or missing 'vwap' column. Cannot aggregate.")
        return pd.DataFrame(columns=["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"])

    if 'ts' not in df_vwap_snapshots.columns:
        logger.error(f"[{symbol_id}] 'ts' column missing in VWAP snapshots DataFrame.")
        return pd.DataFrame(columns=["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"])

    df_with_ts_index = df_vwap_snapshots.copy()
    # Ensure 'ts' is datetime and UTC aware for proper floor/ceil operations
    df_with_ts_index['ts'] = pd.to_datetime(df_with_ts_index['ts'], errors='coerce', utc=True)
    df_with_ts_index.dropna(subset=['ts'], inplace=True)
    if df_with_ts_index.empty:
        logger.warning(f"[{symbol_id}] No valid timestamps in VWAP snapshots after coercion. Cannot aggregate.")
        return pd.DataFrame(columns=["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"])

    df_with_ts_index = df_with_ts_index.set_index("ts")

    min_ts = df_with_ts_index.index.min()
    max_ts = df_with_ts_index.index.max()
    expected_index = None

    if pd.isna(min_ts) or pd.isna(max_ts):
        logger.warning(f"[{symbol_id}] Could not determine time range (min_ts or max_ts is NaT). Aggregating based on available data points only.")
        df_with_ts_index["ts_start_agg"] = df_with_ts_index.index.floor(freq)
    else:
        logger.info(f"[{symbol_id}] Data range for aggregation: {min_ts} to {max_ts}")
        df_with_ts_index["ts_start_agg"] = df_with_ts_index.index.floor(freq)
        try:
            expected_index = pd.date_range(start=min_ts.floor(freq), end=max_ts.ceil(freq), freq=freq, name="ts_start")
            if expected_index.tz is None and min_ts.tz is not None: # Ensure expected_index is tz-aware if source is
                expected_index = expected_index.tz_localize(min_ts.tz)
        except Exception as e:
            logger.error(f"[{symbol_id}] Error creating date range for frequency '{freq}': {e}. Proceeding without full range fill.")
            expected_index = None # Fallback

    aggregated = (
        df_with_ts_index.groupby("ts_start_agg")
          .agg(
              mark_price_median = ("vwap", "median"),
              depth_sum_sz_median = ("depth_sum_sz", "median"),
              n_snapshots = ("vwap", "count")
          )
    ).rename_axis("ts_start")

    if expected_index is not None and not expected_index.empty:
        aggregated = aggregated.reindex(expected_index)
        logger.info(f"[{symbol_id}] Reindexed aggregates to full time range of {len(expected_index)} intervals for frequency {freq}.")
    elif expected_index is not None and expected_index.empty and not aggregated.empty:
         logger.info(f"[{symbol_id}] Data spans less than one '{freq}' interval. Aggregated {len(aggregated)} row(s).")
    elif aggregated.empty: # This can happen if groupby results in nothing (e.g. no data points after floor)
        logger.warning(f"[{symbol_id}] Aggregation resulted in an empty DataFrame before processing NaNs.")
        if expected_index is not None and not expected_index.empty: # If we have a range, create empty shell
             aggregated = pd.DataFrame(index=expected_index)
        else: # Truly no data to form even an empty shell with an index
            return pd.DataFrame(columns=["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"])


    aggregated['mark_price'] = aggregated.get('mark_price_median', pd.Series(dtype=float)).ffill().bfill()
    aggregated['depth_sum_sz'] = aggregated.get('depth_sum_sz_median', pd.Series(dtype=float)).fillna(0)
    aggregated['n_snapshots'] = aggregated.get('n_snapshots', pd.Series(dtype=int)).fillna(0).astype(int)

    aggregated["ts_end"] = aggregated.index + pd.Timedelta(freq) - pd.Timedelta(seconds=1)
    aggregated["symbol_id"] = symbol_id
    aggregated = aggregated.reset_index() # ts_start becomes a column

    final_cols = ["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"]
    for col in final_cols:
        if col not in aggregated.columns:
            if col == "mark_price": aggregated[col] = np.nan
            elif col == "depth_sum_sz" or col == "n_snapshots": aggregated[col] = 0
            elif col == "ts_start" and "index" in aggregated.columns: aggregated.rename(columns={"index":"ts_start"},inplace=True) # if reset_index() was used on empty
            # symbol_id, ts_end should be fine

    aggregated = aggregated[final_cols]
    logger.info(f"[{symbol_id}] Aggregated {len(aggregated)} mark price intervals (frequency {freq}).")
    return aggregated

# --- Mark Price Cross-Check Functions ---

def _get_ohlcv_symbol_for_mark_price(mark_price_symbol_id: str) -> str:
    """
    Determines the corresponding OHLCV symbol name for a given mark price symbol ID.
    Example: 'MEXCFTS_PERP_SPX_USDT' -> 'MEXC_SPOT_SPX_USDT'
    """
    logger.debug(f"Getting OHLCV symbol for mark price ID: {mark_price_symbol_id}")
    if mark_price_symbol_id.startswith("MEXCFTS_PERP_"):
        return mark_price_symbol_id.replace("MEXCFTS_PERP_", "MEXC_SPOT_", 1)
    # Add more general or specific rules if needed for other exchanges/symbol formats
    # E.g., COINAPI_SPOT_OKX_BTC_USDT_SWAP (a perp) -> COINAPI_SPOT_OKX_BTC_USDT (spot for OHLCV)
    if "_PERP" in mark_price_symbol_id: # A common suffix for perpetuals
        return mark_price_symbol_id.replace("_PERP", "")
    if "_SWAP" in mark_price_symbol_id: # Another common suffix for perpetuals
         return mark_price_symbol_id.replace("_SWAP", "")

    logger.warning(f"Uncertain OHLCV mapping for '{mark_price_symbol_id}'. Assuming it's a spot or directly usable ID.")
    return mark_price_symbol_id


def cross_check_mark_price_with_ohlcv(
    df_mark_price: pd.DataFrame,
    db_path: str,
    mark_price_symbol_id: str,
    threshold: float,
    ohlcv_symbol_override: Optional[str] = None
) -> pd.DataFrame:
    """
    Cross-checks mark prices against corresponding OHLCV close prices from the clean table.
    Adds 'ohlcv_price_close', 'relative_difference', and 'flag_anomaly' columns.
    """
    if df_mark_price.empty:
        logger.warning(f"[{mark_price_symbol_id}] Mark price DataFrame is empty. Skipping cross-check.")
        return df_mark_price.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    ohlcv_symbol_to_load = ohlcv_symbol_override if ohlcv_symbol_override else _get_ohlcv_symbol_for_mark_price(mark_price_symbol_id)
    logger.info(f"[{mark_price_symbol_id}] Cross-checking with OHLCV symbol '{ohlcv_symbol_to_load}'. Threshold: {threshold:.2%}")

    # Ensure ts_start is datetime for min/max
    if 'ts_start' not in df_mark_price.columns:
        logger.error(f"[{mark_price_symbol_id}] 'ts_start' column missing from mark price data.")
        return df_mark_price.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    df_mark_price['ts_start'] = pd.to_datetime(df_mark_price['ts_start'], errors='coerce', utc=True)
    df_mark_price.dropna(subset=['ts_start'], inplace=True)
    if df_mark_price.empty:
        logger.warning(f"[{mark_price_symbol_id}] Mark price DataFrame empty after 'ts_start' NaT drop. Skipping cross-check.")
        return df_mark_price.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    min_ts_start = df_mark_price["ts_start"].min()
    max_ts_start = df_mark_price["ts_start"].max() # Max ts_start, so range should include its full interval

    if pd.isna(min_ts_start) or pd.isna(max_ts_start):
        logger.warning(f"[{mark_price_symbol_id}] Invalid date range from mark price data for OHLCV query. Skipping cross-check.")
        return df_mark_price.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    # Fetch OHLCV data for the relevant period. Add a small buffer to end_date if max_ts_start is just the start of interval.
    # load_cleaned_ohlcv_data expects ISO strings.
    df_ohlcv = load_cleaned_ohlcv_data(
        db_path,
        symbol=ohlcv_symbol_to_load,
        start_date_iso=min_ts_start.isoformat(),
        end_date_iso=(max_ts_start + pd.Timedelta(days=1)).isoformat() # Ensure we get data for the last day
    )

    if df_ohlcv.empty:
        logger.warning(f"[{mark_price_symbol_id}] No cleaned OHLCV data found for '{ohlcv_symbol_to_load}' in the required range.")
        return df_mark_price.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    # Prepare for merge: OHLCV data has 'time_period_start', Mark Price has 'ts_start'
    df_ohlcv.rename(columns={'time_period_start': 'ts_start', 'price_close': 'ohlcv_price_close'}, inplace=True)

    # Ensure 'ts_start' in df_ohlcv is also datetime and UTC (should be from load_cleaned_ohlcv_data)
    df_ohlcv['ts_start'] = pd.to_datetime(df_ohlcv['ts_start'], errors='coerce', utc=True)
    df_ohlcv.dropna(subset=['ts_start'], inplace=True)

    # Merge mark prices with OHLCV close prices
    df_merged = pd.merge(df_mark_price, df_ohlcv[['ts_start', 'ohlcv_price_close']], on="ts_start", how="left")

    # Calculate relative difference
    # Ensure mark_price is numeric
    df_merged["mark_price"] = pd.to_numeric(df_merged["mark_price"], errors='coerce')

    valid_prices_mask = df_merged["mark_price"].notna() & df_merged["ohlcv_price_close"].notna() & (df_merged["ohlcv_price_close"].abs() > 1e-9)
    df_merged["relative_difference"] = np.nan
    df_merged.loc[valid_prices_mask, "relative_difference"] = \
        (df_merged.loc[valid_prices_mask, "mark_price"] - df_merged.loc[valid_prices_mask, "ohlcv_price_close"]).abs() / df_merged.loc[valid_prices_mask, "ohlcv_price_close"]

    df_merged["flag_anomaly"] = (df_merged["relative_difference"] > threshold) & df_merged["mark_price"].notna()

    num_anomalies = df_merged['flag_anomaly'].sum()
    logger.info(f"[{mark_price_symbol_id}] Cross-check completed: {num_anomalies} anomalies found (threshold {threshold:.2%}).")

    return df_merged
