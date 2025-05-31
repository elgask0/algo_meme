import sqlite3
import os
import requests
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from typing import Optional, List, Dict, Any, Union

# New imports for cleaning and validation
import logging
import numpy as np
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.errors import SchemaErrors # For explicit error handling
from scipy.stats import median_abs_deviation as scipy_mad


# --- Logging Configuration (Basic) ---
logger = logging.getLogger(__name__)


# --- Pandera Schemas ---
OHLCV_SCHEMA = DataFrameSchema({
    "time_period_start": Column(pa.DateTime(tz='UTC'), nullable=False),
    "time_period_end": Column(pa.DateTime(tz='UTC'), nullable=False),
    "time_open": Column(pa.DateTime(tz='UTC'), nullable=True),
    "time_close": Column(pa.DateTime(tz='UTC'), nullable=True),
    "price_open": Column(float, Check.gt(0), nullable=True, coerce=True),
    "price_high": Column(float, Check.gt(0), nullable=True, coerce=True),
    "price_low": Column(float, Check.gt(0), nullable=True, coerce=True),
    "price_close": Column(float, Check.gt(0), nullable=True, coerce=True),
    "volume_traded": Column(float, Check.ge(0), nullable=False, coerce=True),
    "trades_count": Column(pa.Int64, Check.ge(0), nullable=False, coerce=True),
    "flag_bad_structure": Column(bool, nullable=True, required=False),
    "flag_outlier_mad": Column(bool, nullable=True, required=False),
    "flag_jump": Column(bool, nullable=True, required=False),
}, index=pa.Index(pd.DatetimeIndex, name="time_period_start", unique=True), strict=False, ordered=True)

_ORDERBOOK_SIZE_CAP_THRESHOLD = 1e14

ORDERBOOK_SCHEMA = DataFrameSchema({
    "ts": Column(pa.DateTime(tz='UTC'), nullable=False),
    "date": Column(pa.DateTime(tz='UTC'), nullable=False),
    "bid1_px": Column(float, Check.gt(0), nullable=True, coerce=True),
    "bid1_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True),
    "ask1_px": Column(float, Check.gt(0), nullable=True, coerce=True),
    "ask1_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True),
    "bid2_px": Column(float, Check.gt(0), nullable=True, coerce=True, required=False),
    "bid2_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True, required=False),
    "ask2_px": Column(float, Check.gt(0), nullable=True, coerce=True, required=False),
    "ask2_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True, required=False),
    "bid3_px": Column(float, Check.gt(0), nullable=True, coerce=True, required=False),
    "bid3_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True, required=False),
    "ask3_px": Column(float, Check.gt(0), nullable=True, coerce=True, required=False),
    "ask3_sz": Column(float, Check.ge(0), Check.lt(_ORDERBOOK_SIZE_CAP_THRESHOLD), nullable=True, coerce=True, required=False),
    "flag_ob_bad_structure": Column(bool, nullable=True, required=False),
    "flag_spread_mad": Column(bool, nullable=True, required=False),
    "flag_mid_mad": Column(bool, nullable=True, required=False),
}, index=pa.Index(pd.DatetimeIndex, name="ts", unique=True), strict=False, ordered=True)


# --- Database Initialization ---
def initialize_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS symbol_info (symbol_id TEXT PRIMARY KEY, ticker TEXT, exchange_id TEXT, symbol_type TEXT, asset_id_base TEXT, asset_id_quote TEXT, data_start TEXT, data_end TEXT);")
    cur.execute("CREATE TABLE IF NOT EXISTS coinapi_ohlcv (symbol TEXT, time_period_start TEXT, time_period_end TEXT, time_open TEXT, time_close TEXT, price_open REAL, price_high REAL, price_low REAL, price_close REAL, volume_traded REAL, trades_count INTEGER, PRIMARY KEY(symbol, time_period_start));")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_ohlcv_clean (
            symbol TEXT, time_period_start TEXT, time_period_end TEXT,
            time_open TEXT, time_close TEXT,
            price_open REAL, price_high REAL, price_low REAL, price_close REAL,
            volume_traded REAL, trades_count INTEGER,
            flag_bad_structure INTEGER, flag_outlier_fixed INTEGER,
            flag_outlier_mad INTEGER, flag_jump INTEGER,
            PRIMARY KEY(symbol, time_period_start)
        );""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_orderbook_clean (
            symbol_id TEXT, ts TEXT, date TEXT,
            bid1_px REAL, bid1_sz REAL, bid2_px REAL, bid2_sz REAL, bid3_px REAL, bid3_sz REAL,
            ask1_px REAL, ask1_sz REAL, ask2_px REAL, ask2_sz REAL, ask3_px REAL, ask3_sz REAL,
            flag_ob_bad_structure INTEGER, flag_spread_mad INTEGER, flag_mid_mad INTEGER,
            PRIMARY KEY(symbol_id, ts)
        );""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_clean_symbol_date ON coinapi_orderbook_clean(symbol_id, date);")
    cur.execute("CREATE TABLE IF NOT EXISTS mexc_funding_rate_history (symbol TEXT, ts TEXT, funding_rate REAL, collect_cycle INTEGER, PRIMARY KEY(symbol, ts));")
    cur.execute("CREATE TABLE IF NOT EXISTS mark_price_vwap (symbol_id TEXT, ts_start TEXT, ts_end TEXT, mark_price REAL, depth_sum_sz REAL, n_snapshots INTEGER, PRIMARY KEY(symbol_id, ts_start));")
    cur.execute("CREATE TABLE IF NOT EXISTS perp_synthetic (symbol_id TEXT, ts_start TEXT, ts_end TEXT, perp_price REAL, funding_cum REAL, spot_price REAL, PRIMARY KEY(symbol_id, ts_start));")
    conn.commit()
    conn.close()

# --- Symbol Mapping Functions ---
def fetch_candidates(api_key: str, generic: str, symbol_type_filter: str = "PERP") -> list:
    url = "https://rest.coinapi.io/v1/symbols"
    headers = {'Accept': 'application/json', 'X-CoinAPI-Key': api_key}
    params = {'filter_asset_id': generic.upper()}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"CoinAPI request failed for {generic}: {e}")
        return []
    all_symbols = resp.json()
    if symbol_type_filter:
        return [s for s in all_symbols if s.get('symbol_type') == symbol_type_filter.upper()]
    return all_symbols

def parse_start(item: dict) -> datetime:
    ds = item.get('data_trade_start') or item.get('data_start')
    if not ds: return datetime.min
    if 'T' not in ds: ds += 'T00:00:00'
    ds = ds.rstrip('Z')
    common_formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]
    parsed_date = None
    for fmt in common_formats:
        try:
            if fmt == "%Y-%m-%d" and "T" in ds:
                 parsed_date = datetime.strptime(ds.split("T")[0], fmt)
            else: parsed_date = datetime.strptime(ds, fmt)
            break
        except ValueError: continue
    if parsed_date is None:
        try: parsed_date = datetime.fromisoformat(ds)
        except ValueError:
            logger.warning(f"Could not parse date string '{ds}'")
            return datetime.min
    return parsed_date

def map_generic_to_symbol_id(db_path: str, tickers_file_path: str, api_key: str):
    try:
        with open(tickers_file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Tickers file not found: {tickers_file_path}")
        return
    if not tickers: logger.info(f"No tickers in {tickers_file_path}.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for ticker in tickers:
        type_map = {'SPOT': 'SPOT', 'PERP': 'PERP'}
        for stype_label, stype_api_filter in type_map.items():
            db_symbol_type_value = 'PERPETUAL' if stype_api_filter == 'PERP' else stype_api_filter
            cur.execute("SELECT symbol_id FROM symbol_info WHERE ticker = ? AND symbol_type = ?", (ticker, db_symbol_type_value))
            existing = cur.fetchone()
            if existing:
                logger.info(f"'{ticker}' {stype_label} already mapped to '{existing[0]}', skipping.")
                continue
            candidates = fetch_candidates(api_key, ticker, symbol_type_filter=stype_api_filter)
            if not candidates:
                logger.info(f"No candidates for '{ticker}' {stype_label} (API filter: {stype_api_filter}), skipping.")
                continue
            candidates.sort(key=parse_start, reverse=True)
            print(f"\nGenérico '{ticker}' {stype_label} ({stype_api_filter}) → {len(candidates)} candidatos:")
            for i, item in enumerate(candidates):
                start_str = parse_start(item).strftime('%Y-%m-%d') if parse_start(item) != datetime.min else 'N/A'
                print(f" [{i}] {item['symbol_id']} ({item.get('exchange_id', 'N/A')}, {item.get('symbol_type', 'N/A')}), start: {start_str}")
            sel = input(f"Select index for {stype_label} of '{ticker}' (enter to skip): ")
            if not sel.strip():
                logger.info(f"Skipped '{ticker}' {stype_label}.")
                continue
            try:
                chosen = candidates[int(sel)]
                ds_str = parse_start(chosen).strftime('%Y-%m-%dT%H:%M:%S') if parse_start(chosen) != datetime.min else None
                cur.execute(
                    "INSERT OR REPLACE INTO symbol_info (symbol_id, ticker, symbol_type, exchange_id, asset_id_base, asset_id_quote, data_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chosen['symbol_id'], ticker, db_symbol_type_value, chosen.get('exchange_id'), chosen.get('asset_id_base'), chosen.get('asset_id_quote'), ds_str)
                )
                conn.commit()
                logger.info(f"→ '{ticker}' {stype_label} → '{chosen['symbol_id']}' saved.")
            except (ValueError, IndexError): logger.warning("Invalid selection, skipping.")
    conn.close()
    logger.info("Mapping completed.")

# --- CoinAPI Data Ingestion Functions ---
_COINAPI_RATE_LIMIT_DELAY_SECONDS = 0.5
def _coinapi_safe_get(url: str, api_key: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, initial_delay: float = 0.2) -> requests.Response:
    current_delay = initial_delay
    with requests.Session() as session:
        session.headers.update({"X-CoinAPI-Key": api_key, "Accept": "application/json"})
        for attempt in range(retries):
            if attempt > 0: time.sleep(current_delay)
            try:
                response = session.get(url, params=params)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                err_text = e.response.text if e.response else 'No response text'
                log_prefix = f"[CoinAPI] Attempt {attempt+1}/{retries} for {url}: "
                if status_code == 429:
                    wait_time = int(e.response.headers.get("Retry-After", _COINAPI_RATE_LIMIT_DELAY_SECONDS))
                    logger.warning(f"{log_prefix}Rate limit (429). Retrying after {wait_time}s.")
                    time.sleep(wait_time)
                    current_delay = wait_time
                elif status_code in [500, 502, 503, 504]:
                    logger.warning(f"{log_prefix}Server error ({status_code}). Retrying in {current_delay}s.")
                    current_delay *= 2
                else:
                    logger.error(f"{log_prefix}HTTP error {status_code}: {err_text}")
                    raise
                if attempt == retries -1:
                    logger.error(f"Final attempt failed for {url} after HTTP error.")
                    raise
            except requests.exceptions.RequestException as e:
                logger.warning(f"[CoinAPI] Attempt {attempt+1}/{retries} for {url}: Request failed: {e}. Retrying in {current_delay}s.")
                current_delay *= 2
                if attempt == retries -1:
                    logger.error(f"Final attempt failed for {url} after RequestException.")
                    raise
        raise requests.exceptions.RequestException(f"Exhausted retries for {url}")

def fetch_coinapi_symbol_metadata(symbol_id: str, api_key: str) -> Dict[str, Any]:
    url = "https://rest.coinapi.io/v1/symbols"
    params = {"filter_symbol_id": symbol_id}
    try:
        resp = _coinapi_safe_get(url, api_key, params=params)
        data = resp.json()
        return data[0] if isinstance(data, list) and data else {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch CoinAPI metadata for {symbol_id}: {e}")
        return {}

def ingest_coinapi_symbol_metadata(db_path: str, symbol_id: str, api_key: str) -> List[str]:
    logs: List[str] = []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT data_end FROM symbol_info WHERE symbol_id = ?", (symbol_id,))
        meta_row = cur.fetchone()
        if meta_row and meta_row[0]:
            logs.append(f"[DB] Metadata (data_end: {meta_row[0]}) already exists for {symbol_id}. Skipping API call.")
            logger.info(logs[-1])
            return logs
        info = fetch_coinapi_symbol_metadata(symbol_id, api_key)
        if not info or 'symbol_id' not in info:
            logs.append(f"[API] No metadata found or invalid format for {symbol_id} from CoinAPI.")
            logger.warning(logs[-1])
            return logs
        data_start = info.get('data_trade_start') or info.get('data_quote_start') or info.get('data_orderbook_start') or info.get('data_start')
        data_end = info.get('data_trade_end') or info.get('data_quote_end') or info.get('data_orderbook_end') or info.get('data_end')
        cur.execute(
            "UPDATE symbol_info SET exchange_id=?,symbol_type=?,asset_id_base=?,asset_id_quote=?,data_start=?,data_end=? WHERE symbol_id=?",
            (info.get('exchange_id'), info.get('symbol_type'), info.get('asset_id_base'), info.get('asset_id_quote'), data_start, data_end, symbol_id)
        )
        if cur.rowcount == 0: logs.append(f"[DB] No row found for {symbol_id} in symbol_info to update.")
        else:
            conn.commit()
            logs.append(f"[DB] Updated symbol_info metadata for {symbol_id} (data_end: {data_end}).")
        logger.info(logs[-1] if logs else f"No action for {symbol_id} metadata.")
    except sqlite3.Error as e:
        err_msg = f"[DB] SQLite error for {symbol_id} metadata: {e}"
        logs.append(err_msg); logger.error(err_msg)
    except Exception as e:
        err_msg = f"[ERROR] Unexpected error for {symbol_id} metadata: {e}"
        logs.append(err_msg); logger.exception(err_msg)
    finally: conn.close()
    return logs

def fetch_coinapi_ohlcv(symbol_id: str, time_start_iso: str, time_end_iso: str, api_key: str,
                        period_id: str = "5MIN", limit: int = 1000, include_empty_items: bool = True) -> pd.DataFrame:
    if isinstance(time_start_iso, datetime): time_start_iso = time_start_iso.isoformat()
    if isinstance(time_end_iso, datetime): time_end_iso = time_end_iso.isoformat()
    url = (
        f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"
        f"?period_id={period_id.upper()}&time_start={time_start_iso}&time_end={time_end_iso}"
        f"&limit={limit}&include_empty_items={'true' if include_empty_items else 'false'}"
    )
    try:
        resp = _coinapi_safe_get(url, api_key)
        data = resp.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ["time_period_start", "time_period_end", "time_open", "time_close"]:
            df[col] = _robust_to_datetime(df[col], series_name=f"{symbol_id}_{col}")
        return df.set_index("time_period_start").sort_index()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch CoinAPI OHLCV for {symbol_id} ({time_start_iso} to {time_end_iso}): {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing CoinAPI OHLCV for {symbol_id}: {e}")
        return pd.DataFrame()

def ingest_coinapi_ohlcv_for_date(db_path: str, symbol_id: str, date_iso: str,
                                  api_key: str, period_id: str = "5MIN") -> List[str]:
    logs: List[str] = []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM coinapi_ohlcv WHERE symbol = ? AND date(time_period_start) = ? AND trades_count > 0 LIMIT 1", (symbol_id, date_iso))
        if cur.fetchone():
            logs.append(f"[DB] Actual OHLCV data already for {symbol_id} on {date_iso}. Skipping."); logger.info(logs[-1])
            return logs
        start_dt = datetime.fromisoformat(date_iso + "T00:00:00").replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1)
        df = fetch_coinapi_ohlcv(symbol_id, start_dt.isoformat(), end_dt.isoformat(), api_key, period_id)
        cur.execute("DELETE FROM coinapi_ohlcv WHERE symbol = ? AND date(time_period_start) = ? AND trades_count = 0", (symbol_id, date_iso))
        if df.empty or df['trades_count'].sum() == 0:
            marker = (symbol_id, start_dt.isoformat(), end_dt.isoformat(), start_dt.isoformat(), end_dt.isoformat(),0,0,0,0,0,0)
            cur.execute("INSERT OR REPLACE INTO coinapi_ohlcv VALUES (?,?,?,?,?,?,?,?,?,?,?)", marker)
            conn.commit()
            logs.append(f"[DB] {'No data from API' if df.empty else '0 trades count'}. Inserted marker for {symbol_id} on {date_iso}."); logger.info(logs[-1])
            return logs
        rows = []
        for ts, r in df.iterrows():
            if r.get('trades_count', 0) > 0:
                rows.append((symbol_id, ts.isoformat(), r["time_period_end"].isoformat(), r["time_open"].isoformat(), r["time_close"].isoformat(),
                             r["price_open"],r["price_high"],r["price_low"],r["price_close"],r["volume_traded"],r["trades_count"]))
        if rows:
            cur.executemany("INSERT OR REPLACE INTO coinapi_ohlcv VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            logs.append(f"[DB] Inserted {len(rows)} OHLCV records for {symbol_id} on {date_iso} ({period_id})."); logger.info(logs[-1])
    except sqlite3.Error as e:
        err_msg = f"[DB] SQLite error for {symbol_id} OHLCV on {date_iso}: {e}"
        logs.append(err_msg); logger.error(err_msg)
    except Exception as e:
        err_msg = f"[ERROR] Unexpected error for {symbol_id} OHLCV on {date_iso}: {e}"
        logs.append(err_msg); logger.exception(err_msg)
    finally: conn.close()
    return logs

def fetch_coinapi_orderbook_snapshots(symbol_id: str, date_iso: str, api_key: str, limit_levels: int = 3) -> pd.DataFrame:
    url = f"https://rest.coinapi.io/v1/orderbooks/{symbol_id}/history"
    params = {"date": date_iso, "limit": 100}
    try:
        resp = _coinapi_safe_get(url, api_key, params=params)
        snaps_data = resp.json()
        if not snaps_data: return pd.DataFrame()
        records = []
        for snap in snaps_data:
            rec = {"ts": pd.to_datetime(snap["time_exchange"]), "symbol_id": symbol_id, "date": date_iso}
            for i in range(limit_levels):
                for side, key_px, key_sz in [("bids", f"bid{i+1}_px", f"bid{i+1}_sz"), ("asks", f"ask{i+1}_px", f"ask{i+1}_sz")]:
                    item = snap.get(side, [])
                    rec[key_px] = item[i]["price"] if i < len(item) else None
                    rec[key_sz] = item[i]["size"]  if i < len(item) else None
            records.append(rec)
        if not records: return pd.DataFrame()
        return pd.DataFrame(records).set_index("ts").sort_index().resample("5min").first().dropna(how="all")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed CoinAPI orderbook fetch for {symbol_id} on {date_iso}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing CoinAPI orderbook for {symbol_id} on {date_iso}: {e}")
        return pd.DataFrame()

def ingest_coinapi_orderbook_for_date(db_path: str, symbol_id: str, date_iso: str,
                                      api_key: str, limit_levels: int = 3) -> List[str]:
    logs: List[str] = []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM coinapi_orderbook WHERE symbol_id = ? AND date = ? AND bid1_px IS NOT NULL LIMIT 1",(symbol_id, date_iso))
        if cur.fetchone():
            logs.append(f"[DB] Actual order book data already for {symbol_id} on {date_iso}. Skipping."); logger.info(logs[-1])
            return logs
        df_snaps = fetch_coinapi_orderbook_snapshots(symbol_id, date_iso, api_key, limit_levels)
        cur.execute("DELETE FROM coinapi_orderbook WHERE symbol_id = ? AND date = ? AND bid1_px IS NULL", (symbol_id, date_iso))
        cols_base = ["symbol_id", "ts", "date"]
        level_cols = []
        for i in range(1, limit_levels + 1):
            level_cols.extend([f"bid{i}_px", f"bid{i}_sz", f"ask{i}_px", f"ask{i}_sz"])
        cols = cols_base + level_cols

        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO coinapi_orderbook ({', '.join(cols)}) VALUES ({placeholders});"
        if df_snaps.empty:
            ts_marker = datetime.fromisoformat(date_iso + "T00:00:00").replace(tzinfo=timezone.utc).isoformat()
            marker_vals = tuple([symbol_id, ts_marker, date_iso] + [None]*(limit_levels*4))
            cur.execute(sql, marker_vals)
            conn.commit()
            logs.append(f"[DB] No orderbook snapshots from API for {symbol_id} on {date_iso}. Inserted marker."); logger.info(logs[-1])
            return logs
        rows = []
        for ts, r_data in df_snaps.iterrows():
            if r_data.isnull().all(): continue
            row_tuple_list = [symbol_id, ts.isoformat(), date_iso]
            for i in range(1, limit_levels+1):
                row_tuple_list.append(r_data.get(f"bid{i}_px", None))
                row_tuple_list.append(r_data.get(f"bid{i}_sz", None))
                row_tuple_list.append(r_data.get(f"ask{i}_px", None))
                row_tuple_list.append(r_data.get(f"ask{i}_sz", None))
            rows.append(tuple(row_tuple_list))
        if rows:
            cur.executemany(sql, rows)
            conn.commit()
            logs.append(f"[DB] Inserted {len(rows)} orderbook records for {symbol_id} on {date_iso}."); logger.info(logs[-1])
        else:
            ts_marker = datetime.fromisoformat(date_iso + "T00:00:00").replace(tzinfo=timezone.utc).isoformat()
            marker_vals = tuple([symbol_id, ts_marker, date_iso] + [None]*(limit_levels*4))
            cur.execute(sql, marker_vals)
            conn.commit()
            logs.append(f"[DB] Orderbook data for {symbol_id} on {date_iso} had no valid rows after resample. Inserted marker."); logger.info(logs[-1])
    except sqlite3.Error as e:
        err_msg = f"[DB] SQLite error for {symbol_id} orderbook on {date_iso}: {e}"
        logs.append(err_msg); logger.error(err_msg)
    except Exception as e:
        err_msg = f"[ERROR] Unexpected error for {symbol_id} orderbook on {date_iso}: {e}"
        logs.append(err_msg); logger.exception(err_msg)
    finally: conn.close()
    return logs

# --- MEXC Specific Data Ingestion ---
_MEXC_PERP_PREFIX = "MEXCFTS_PERP_"
def _mexc_format_api_symbol(symbol_id: str) -> str:
    if symbol_id.startswith(_MEXC_PERP_PREFIX): return symbol_id[len(_MEXC_PERP_PREFIX):]
    return symbol_id

def fetch_mexc_funding_rate_history(api_symbol: str, page_size: int = 100) -> List[Dict[str, Any]]:
    url = "https://contract.mexc.com/api/v1/contract/funding_rate/history"
    all_records: List[Dict[str, Any]] = []
    current_page = 1
    while True:
        params = {"symbol": api_symbol, "page_num": current_page, "page_size": page_size}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise requests.exceptions.HTTPError(f"MEXC API error: {payload.get('message', 'Unknown error')}")
            data = payload.get("data", {})
            records_on_page = data.get("resultList", [])
            all_records.extend(records_on_page)
            if not records_on_page or current_page >= data.get("totalPage", 0): break
            current_page += 1
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MEXC funding for {api_symbol} page {current_page}: {e}")
            break
    return all_records

def ingest_mexc_funding_history(db_path: str, symbol_id: str) -> List[str]:
    logs: List[str] = []
    api_symbol = _mexc_format_api_symbol(symbol_id)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT MAX(ts) FROM mexc_funding_rate_history WHERE symbol = ?", (symbol_id,))
        last_ts_row = cur.fetchone()
        last_stored_ms: Optional[int] = None
        if last_ts_row and last_ts_row[0]:
            last_dt = datetime.fromisoformat(last_ts_row[0])
            last_stored_ms = int(last_dt.timestamp() * 1000)
            if last_dt >= (datetime.now(timezone.utc) - timedelta(hours=12)):
                 logs.append(f"[DB] Funding for {symbol_id} up-to-date (last: {last_dt.isoformat()}). Skipping API."); logger.info(logs[-1])
                 return logs
        all_api_records = fetch_mexc_funding_rate_history(api_symbol)
        if not all_api_records:
            logs.append(f"[API] No funding history from MEXC for {api_symbol}."); logger.info(logs[-1])
            return logs
        new_records = []
        for rec in all_api_records:
            if (st := rec.get('settleTime')) and (last_stored_ms is None or st > last_stored_ms):
                new_records.append((symbol_id, pd.to_datetime(st,unit='ms',utc=True).isoformat(), rec.get('fundingRate'), rec.get('collectCycle')))
        if new_records:
            cur.executemany("INSERT OR REPLACE INTO mexc_funding_rate_history VALUES (?, ?, ?, ?)", new_records)
            conn.commit()
            logs.append(f"[DB] Inserted {len(new_records)} new MEXC funding records for {symbol_id}."); logger.info(logs[-1])
        else:
            logs.append(f"[DB] No new MEXC funding records for {symbol_id} (last_stored_ts: {last_ts_row[0] if last_ts_row else 'None'})."); logger.info(logs[-1])
    except sqlite3.Error as e:
        err_msg = f"[DB] SQLite error for {symbol_id} MEXC funding: {e}"
        logs.append(err_msg); logger.error(err_msg)
    except Exception as e:
        err_msg = f"[ERROR] Unexpected error for {symbol_id} MEXC funding: {e}"
        logs.append(err_msg); logger.exception(err_msg)
    finally: conn.close()
    return logs

# --- OHLCV Data Loading and Initial Cleaning ---
def _robust_to_datetime(series: pd.Series, series_name: str = "") -> pd.Series:
    converted_series = pd.to_datetime(series, errors='coerce', utc=True)
    num_failed = converted_series.isnull().sum()
    if num_failed > 0 and series[series.notna()].shape[0] > converted_series.notna().shape[0]:
        logger.warning(f"Robust to_datetime for '{series_name}': {num_failed}/{len(series)} values failed conversion and became NaT.")
    return converted_series

def _apply_min_start_date_filter(df: pd.DataFrame, date_col: Union[str, pd.DatetimeIndex], min_start_date_str: Optional[str], symbol_context: str) -> pd.DataFrame:
    if not min_start_date_str: return df
    try: min_start_date = pd.to_datetime(min_start_date_str, utc=True)
    except ValueError:
        logger.error(f"[{symbol_context}] Invalid min_start_date format: '{min_start_date_str}'. Skipping filter.")
        return df

    date_series_to_filter = df[date_col] if isinstance(date_col, str) else date_col
    if not isinstance(date_series_to_filter, pd.Series) and not isinstance(date_series_to_filter, pd.DatetimeIndex):
        logger.error(f"[{symbol_context}] date_col for filtering must be a column name or DatetimeIndex.")
        return df

    original_len = len(df)
    df_filtered = df[date_series_to_filter >= min_start_date].copy()
    if (rows_dropped := original_len - len(df_filtered)) > 0:
        logger.info(f"[{symbol_context}] Filtered {rows_dropped} rows before min_start_date {min_start_date_str}. New length: {len(df_filtered)}.")
    return df_filtered

def _apply_min_initial_price_filter(df: pd.DataFrame, price_ref_series: pd.Series,
                                   min_initial_price: Optional[float], symbol_context: str,
                                   time_col_for_logging: Union[str, pd.DatetimeIndex]) -> pd.DataFrame:
    if not min_initial_price or min_initial_price <= 0: return df
    first_valid_idx = price_ref_series.first_valid_index()
    if first_valid_idx is None:
        logger.warning(f"[{symbol_context}] No valid prices in price_ref_series. Cannot apply min_initial_price filter.")
        return df
    cutoff_idx = None
    for idx, price in price_ref_series.loc[first_valid_idx:].items():
        if pd.notna(price) and price >= min_initial_price: cutoff_idx = idx; break
    if cutoff_idx is None:
        logger.warning(f"[{symbol_context}] All valid prices below min_initial_price ({min_initial_price}). Filtering all data.")
        return df.iloc[0:0]
    original_len = len(df)
    df_filtered = df.loc[cutoff_idx:].copy()

    time_value_for_log = df_filtered.iloc[0][time_col_for_logging] if isinstance(time_col_for_logging, str) else df_filtered.iloc[0].name
    if (rows_dropped := original_len - len(df_filtered)) > 0:
        logger.info(f"[{symbol_context}] Filtered {rows_dropped} rows with initial prices < {min_initial_price}. First kept: {time_value_for_log} (price: {price_ref_series[cutoff_idx]:.2f}).")
    return df_filtered

def load_raw_ohlcv_data(db_path: str, symbol: str,
                        min_start_date: Optional[str] = None,
                        min_initial_price: Optional[float] = None) -> pd.DataFrame:
    logger.info(f"[{symbol}] Loading raw OHLCV data...")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM coinapi_ohlcv WHERE symbol = ? ORDER BY time_period_start ASC", conn, params=(symbol,))
    except Exception as e: logger.error(f"[{symbol}] DB load failed: {e}"); return pd.DataFrame()
    finally: conn.close()
    if df.empty: logger.warning(f"[{symbol}] No data in coinapi_ohlcv."); return pd.DataFrame()
    logger.debug(f"[{symbol}] Loaded {len(df)} raw rows.")
    for col in ["time_period_start", "time_period_end"]:
        if col not in df.columns or df[col].isnull().all():
            logger.error(f"[{symbol}] Critical col '{col}' missing or all NaN. Empty df returned."); return pd.DataFrame()
    for col_time in ["time_period_start", "time_period_end", "time_open", "time_close"]:
        df[col_time] = _robust_to_datetime(df[col_time], f"{symbol}_{col_time}")
    df.dropna(subset=["time_period_start"], inplace=True)
    if df.empty: logger.warning(f"[{symbol}] Empty after NaT time_period_start drop."); return pd.DataFrame()

    df = df.set_index(pd.DatetimeIndex(df["time_period_start"]), drop=False)
    df.index.name = "time_period_start"
    df.rename(columns={'time_period_start': 'time_period_start_col'}, inplace=True)

    df = _apply_min_start_date_filter(df, df.index, min_start_date, symbol)
    if df.empty: logger.warning(f"[{symbol}] Empty after min_start_date filter."); return df
    df = _apply_min_initial_price_filter(df, df["price_close"], min_initial_price, symbol, df.index)
    if df.empty: logger.warning(f"[{symbol}] Empty after min_initial_price filter."); return df

    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    for col_price in price_cols:
        df[col_price] = pd.to_numeric(df[col_price], errors='coerce')
    df["volume_traded"] = pd.to_numeric(df["volume_traded"], errors='coerce')
    df["trades_count"] = pd.to_numeric(df["trades_count"], errors='coerce').astype('Int64')

    for col_p in price_cols:
        if df[col_p].isnull().any(): logger.debug(f"[{symbol}] Column '{col_p}' has NaNs after initial load/filters.")
    logger.info(f"[{symbol}] Finished loading raw OHLCV. Shape: {df.shape}")
    return df

# --- OHLCV Cleaning & Feature Engineering ---
def reindex_ohlcv_time(df_ohlcv: pd.DataFrame, freq: str, symbol_context: str) -> pd.DataFrame:
    if df_ohlcv.empty:
        logger.warning(f"[{symbol_context}] Input DataFrame for reindexing is empty.")
        return df_ohlcv
    if not isinstance(df_ohlcv.index, pd.DatetimeIndex) or df_ohlcv.index.name != 'time_period_start':
        col_to_set_idx = None
        if 'time_period_start_col' in df_ohlcv.columns and pd.api.types.is_datetime64_any_dtype(df_ohlcv['time_period_start_col']):
            col_to_set_idx = 'time_period_start_col'
        elif 'time_period_start' in df_ohlcv.columns and pd.api.types.is_datetime64_any_dtype(df_ohlcv['time_period_start']):
             col_to_set_idx = 'time_period_start'
        if col_to_set_idx:
            logger.info(f"[{symbol_context}] Setting index to '{col_to_set_idx}' for reindexing.")
            df_ohlcv = df_ohlcv.set_index(col_to_set_idx, drop=False).sort_index()
            df_ohlcv.index.name = 'time_period_start'
            if col_to_set_idx != 'time_period_start_col' and 'time_period_start_col' not in df_ohlcv.columns and col_to_set_idx == 'time_period_start':
                 df_ohlcv.rename(columns={col_to_set_idx: 'time_period_start_col'}, inplace=True)
        else:
            logger.error(f"[{symbol_context}] DataFrame must have a DatetimeIndex named 'time_period_start' or a valid time column.")
            raise ValueError("DataFrame must have a DatetimeIndex named 'time_period_start' or a valid time column for reindex_ohlcv_time.")

    logger.info(f"[{symbol_context}] Reindexing to '{freq}' frequency...")
    min_ts, max_ts = df_ohlcv.index.min(), df_ohlcv.index.max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        logger.error(f"[{symbol_context}] Min or Max timestamp is NaT. Cannot create date range for reindexing.")
        return df_ohlcv

    full_range = pd.date_range(start=min_ts.floor(freq), end=max_ts.ceil(freq), freq=freq, name="time_period_start")
    df_reindexed = df_ohlcv.reindex(full_range)

    if 'symbol' in df_reindexed.columns: df_reindexed['symbol'] = df_reindexed['symbol'].ffill()
    else: df_reindexed['symbol'] = symbol_context

    df_reindexed['time_period_end'] = df_reindexed.index + pd.Timedelta(freq) - pd.Timedelta(seconds=1)
    df_reindexed['time_open'] = df_reindexed.index
    df_reindexed['time_close'] = df_reindexed.index + pd.Timedelta(freq) - pd.Timedelta(seconds=1)

    fill_zeros = ['volume_traded', 'trades_count']
    for col in fill_zeros:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].fillna(0)
            if col == 'trades_count': df_reindexed[col] = df_reindexed[col].astype(pa.Int64)

    flag_cols = [col for col in df_reindexed.columns if col.startswith('flag_')]
    for flag_col in flag_cols: df_reindexed[flag_col] = df_reindexed[flag_col].fillna(False).astype(bool)

    # Ensure 'time_period_start' column (expected by schema) has the index values
    df_reindexed['time_period_start'] = df_reindexed.index

    logger.info(f"[{symbol_context}] Reindexing complete. Original length: {len(df_ohlcv)}, new length: {len(df_reindexed)}.")
    return df_reindexed

def detect_ohlcv_structure_issues(df_ohlcv: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    df = df_ohlcv.copy()
    logger.info(f"[{symbol_context}] Detecting OHLCV structure issues...")
    df['flag_bad_structure'] = False
    price_cols = ['price_open', 'price_high', 'price_low', 'price_close']
    for col in price_cols: df.loc[df[col].fillna(1) <= 0, 'flag_bad_structure'] = True # fillna to catch actual <=0
    df.loc[df['volume_traded'].fillna(0) < 0, 'flag_bad_structure'] = True
    df.loc[df['trades_count'].fillna(0) < 0, 'flag_bad_structure'] = True
    df.loc[df['price_high'] < df['price_low'], 'flag_bad_structure'] = True
    for col in ['price_open', 'price_close']:
        condition = (df['price_high'] < df[col]) & df[['price_high', col]].notna().all(axis=1)
        df.loc[condition, 'flag_bad_structure'] = True
        condition = (df['price_low'] > df[col]) & df[['price_low', col]].notna().all(axis=1)
        df.loc[condition, 'flag_bad_structure'] = True
    if (num_flagged := df['flag_bad_structure'].sum()) > 0: logger.warning(f"[{symbol_context}] Found {num_flagged} rows with OHLCV structure issues.")
    else: logger.info(f"[{symbol_context}] No OHLCV structure issues detected.")
    return df

def detect_ohlcv_outliers(df_ohlcv: pd.DataFrame, symbol_context: str,
                          mad_threshold: float = 5.0,
                          jump_window_candles: int = 20, jump_threshold_std: float = 5.0) -> pd.DataFrame:
    df = df_ohlcv.copy()
    logger.info(f"[{symbol_context}] Detecting OHLCV outliers...")
    df['flag_outlier_mad'] = False; df['flag_jump'] = False
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        price_cols = ['price_open', 'price_high', 'price_low', 'price_close']
        daily_median = df[price_cols].resample('D').median()
        daily_mad_val = df[price_cols].resample('D').apply(lambda x: scipy_mad(x.dropna(), nan_policy='omit', scale=1/1.4826) if x.dropna().shape[0] > 1 else np.nan)
        daily_median_upsampled = daily_median.reindex(df.index, method='ffill')
        daily_mad_val_upsampled = daily_mad_val.reindex(df.index, method='ffill')
        for col in price_cols:
            valid_mad = daily_mad_val_upsampled[col].notna() & (daily_mad_val_upsampled[col] > 1e-9)
            deviation = (df[col] - daily_median_upsampled[col]).abs()
            is_outlier = valid_mad & (deviation > (daily_mad_val_upsampled[col] * mad_threshold))
            df.loc[is_outlier, 'flag_outlier_mad'] = True
        if (num_mad := df['flag_outlier_mad'].sum()) > 0: logger.info(f"[{symbol_context}] Flagged {num_mad} MAD outliers.")
        if 'price_close' in df.columns:
            log_returns = np.log(df['price_close'] / df['price_close'].shift(1))
            rolling_std = log_returns.rolling(window=jump_window_candles, min_periods=max(1, jump_window_candles//2)).std()
            valid_std = rolling_std.notna() & (rolling_std > 1e-9)
            df.loc[valid_std & (log_returns.abs() > (rolling_std * jump_threshold_std)), 'flag_jump'] = True
            if (num_jump := df['flag_jump'].sum()) > 0: logger.info(f"[{symbol_context}] Flagged {num_jump} price jumps.")
    return df

def impute_ohlcv_data(df_ohlcv: pd.DataFrame, max_gap_candles: int, symbol_context: str) -> pd.DataFrame:
    df = df_ohlcv.copy()
    logger.info(f"[{symbol_context}] Imputing OHLCV data (max_gap_candles: {max_gap_candles})...")
    price_cols = ['price_open', 'price_high', 'price_low', 'price_close']
    if 'flag_outlier_mad' in df.columns: # Ensure flag exists
        outlier_mask = df['flag_outlier_mad'] == True
        for col in price_cols: df.loc[outlier_mask, col] = np.nan
        logger.info(f"[{symbol_context}] Converted {outlier_mask.sum()} MAD outlier prices to NaN.")
    for col in price_cols: df[col] = df[col].interpolate(method='time', limit=max_gap_candles, limit_direction='both')
    df['volume_traded'] = df['volume_traded'].fillna(0)
    df['trades_count'] = df['trades_count'].fillna(0).astype(pa.Int64)
    for col in price_cols: df[col] = df[col].ffill().bfill()
    if not df[price_cols].isnull().any().any():
        df['price_high'] = df[price_cols].max(axis=1)
        df['price_low'] = df[price_cols].min(axis=1)
        logger.info(f"[{symbol_context}] Re-adjusted high/low based on imputed prices.")
    else: logger.warning(f"[{symbol_context}] NaNs in price columns after imputation. High/low not re-adjusted.")
    for flag_col in [col for col in df.columns if col.startswith('flag_')]: df[flag_col] = df[flag_col].fillna(False).astype(bool)
    logger.info(f"[{symbol_context}] Imputation complete.")
    return df

def validate_ohlcv_schema(df_ohlcv: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Validating OHLCV schema...")
    try:
        if df_ohlcv.index.name != "time_period_start": df_ohlcv.index.name = "time_period_start"
        df_val = df_ohlcv.copy()
        if 'time_period_start' not in df_val.columns and isinstance(df_val.index, pd.DatetimeIndex):
             df_val['time_period_start'] = df_val.index
        for flag in ["flag_bad_structure", "flag_outlier_mad", "flag_jump"]:
            if flag not in df_val.columns: df_val[flag] = False
        validated_df = OHLCV_SCHEMA.validate(df_val, lazy=True)
        logger.info(f"[{symbol_context}] OHLCV schema validation successful.")
        return validated_df
    except SchemaErrors as err:
        logger.error(f"[{symbol_context}] OHLCV schema validation failed. Errors:\n{err.failure_cases.to_string()}")
        raise err
    except Exception as e:
        logger.error(f"[{symbol_context}] Unexpected error during OHLCV schema validation: {e}")
        raise

# --- Order Book Data Loading and Cleaning ---
def load_raw_orderbook_data(db_path: str, symbol_id: str,
                            min_start_date: Optional[str] = None,
                            min_initial_price: Optional[float] = None) -> pd.DataFrame:
    logger.info(f"[{symbol_id}] Loading raw order book data...")
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT * FROM coinapi_orderbook WHERE symbol_id = ? ORDER BY ts ASC"
        df = pd.read_sql_query(query, conn, params=(symbol_id,))
    except Exception as e:
        logger.error(f"[{symbol_id}] Failed to load order book data from DB: {e}")
        return pd.DataFrame()
    finally: conn.close()
    if df.empty: logger.warning(f"[{symbol_id}] No data in coinapi_orderbook table."); return pd.DataFrame()
    logger.debug(f"[{symbol_id}] Loaded {len(df)} raw order book rows.")
    if 'ts' not in df.columns or df['ts'].isnull().all():
        logger.error(f"[{symbol_id}] Critical col 'ts' missing or all NaN. Empty df returned."); return pd.DataFrame()
    df['ts'] = _robust_to_datetime(df['ts'], f"{symbol_id}_ts")
    df.dropna(subset=['ts'], inplace=True)
    if df.empty: logger.warning(f"[{symbol_id}] Order book data empty after NaT 'ts' drop."); return pd.DataFrame()

    df = df.set_index(pd.DatetimeIndex(df["ts"]), drop=False)
    df.index.name = "ts"
    df.rename(columns={'ts': 'ts_col'}, inplace=True)

    df = _apply_min_start_date_filter(df, df.index, min_start_date, symbol_id)
    if df.empty: logger.warning(f"[{symbol_id}] Order book data empty after min_start_date filter."); return df
    if 'bid1_px' in df.columns and 'ask1_px' in df.columns:
        df['mid_price_temp'] = (pd.to_numeric(df['bid1_px'], errors='coerce') + pd.to_numeric(df['ask1_px'], errors='coerce')) / 2
        df = _apply_min_initial_price_filter(df, df['mid_price_temp'], min_initial_price, symbol_id, df.index)
        df.drop(columns=['mid_price_temp'], inplace=True, errors='ignore')
        if df.empty: logger.warning(f"[{symbol_id}] Order book data empty after min_initial_price filter."); return df
    else: logger.warning(f"[{symbol_id}] bid1_px or ask1_px missing for min_initial_price filter.")
    for i in range(1, 4):
        for prefix in ['bid', 'ask']:
            for suffix in ['px', 'sz']:
                col_name = f"{prefix}{i}{suffix}"
                if col_name in df.columns:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    if suffix == '_sz': df.loc[df[col_name] > _ORDERBOOK_SIZE_CAP_THRESHOLD, col_name] = _ORDERBOOK_SIZE_CAP_THRESHOLD
                else: df[col_name] = np.nan
    logger.info(f"[{symbol_id}] Finished loading raw order book data. Shape: {df.shape}")
    return df

def detect_orderbook_structure_issues(df_ob: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    df = df_ob.copy()
    logger.info(f"[{symbol_context}] Detecting order book structure issues...")
    df['flag_ob_bad_structure'] = False
    for i in range(1, 4):
        for prefix in ['bid', 'ask']:
            px_col, sz_col = f"{prefix}{i}_px", f"{prefix}{i}_sz"
            if px_col in df.columns: df.loc[df[px_col].fillna(1) <= 0, 'flag_ob_bad_structure'] = True
            if sz_col in df.columns: df.loc[df[sz_col].fillna(0) < 0, 'flag_ob_bad_structure'] = True
    for i in range(1, 3):
        if f"bid{i}_px" in df.columns and f"bid{i+1}_px" in df.columns and df[[f"bid{i}_px", f"bid{i+1}_px"]].notna().all(axis=1).any():
            df.loc[df[f"bid{i}_px"] < df[f"bid{i+1}_px"], 'flag_ob_bad_structure'] = True
        if f"ask{i}_px" in df.columns and f"ask{i+1}_px" in df.columns and df[[f"ask{i}_px", f"ask{i+1}_px"]].notna().all(axis=1).any():
            df.loc[df[f"ask{i}_px"] > df[f"ask{i+1}_px"], 'flag_ob_bad_structure'] = True
    if "bid1_px" in df.columns and "ask1_px" in df.columns and df[["bid1_px", "ask1_px"]].notna().all(axis=1).any():
         df.loc[df["bid1_px"] >= df["ask1_px"], 'flag_ob_bad_structure'] = True
    if (num_flagged := df['flag_ob_bad_structure'].sum()) > 0: logger.warning(f"[{symbol_context}] Found {num_flagged} rows with order book structure issues.")
    else: logger.info(f"[{symbol_context}] No order book structure issues detected.")
    return df

def detect_orderbook_outliers(df_ob: pd.DataFrame, symbol_context: str, mad_threshold: float = 5.0) -> pd.DataFrame:
    df = df_ob.copy()
    logger.info(f"[{symbol_context}] Detecting order book outliers (mid-price MAD)...")
    df['flag_mid_mad'] = False
    if 'bid1_px' not in df.columns or 'ask1_px' not in df.columns:
        logger.warning(f"[{symbol_context}] bid1_px or ask1_px missing. Cannot calc mid-price for MAD.")
        return df
    mid_price = (pd.to_numeric(df['bid1_px'], errors='coerce') + pd.to_numeric(df['ask1_px'], errors='coerce')) / 2
    if mid_price.isnull().all(): logger.warning(f"[{symbol_context}] Mid-price is all NaN. Cannot calc MAD."); return df
    if not isinstance(df.index, pd.DatetimeIndex): logger.error(f"[{symbol_context}] DataFrame must have DatetimeIndex for MAD calc."); return df
    daily_mid_median = mid_price.resample('D').median()
    daily_mid_mad_val = mid_price.resample('D').apply(lambda x: scipy_mad(x.dropna(), nan_policy='omit', scale=1/1.4826) if x.dropna().shape[0] > 1 else np.nan)
    daily_mid_median_upsampled = daily_mid_median.reindex(df.index, method='ffill')
    daily_mid_mad_val_upsampled = daily_mid_mad_val.reindex(df.index, method='ffill')
    valid_mad = daily_mid_mad_val_upsampled.notna() & (daily_mid_mad_val_upsampled > 1e-9)
    deviation = (mid_price - daily_mid_median_upsampled).abs()
    df.loc[valid_mad & (deviation > (daily_mid_mad_val_upsampled * mad_threshold)), 'flag_mid_mad'] = True
    if (num_mad := df['flag_mid_mad'].sum()) > 0: logger.info(f"[{symbol_context}] Flagged {num_mad} mid-price MAD outliers for order book.")
    else: logger.info(f"[{symbol_context}] No mid-price MAD outliers for order book.")
    return df

# --- Order Book Cleaning & Feature Engineering (Continued) ---
def impute_orderbook_data(df_ob: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    df = df_ob.copy()
    logger.info(f"[{symbol_context}] Imputing order book data...")
    price_cols = [col for col in df.columns if '_px' in col]
    size_cols = [col for col in df.columns if '_sz' in col]
    condition_bad_structure = df.get('flag_ob_bad_structure', pd.Series(False, index=df.index))
    condition_mid_mad = df.get('flag_mid_mad', pd.Series(False, index=df.index))
    combined_condition = condition_bad_structure | condition_mid_mad
    for col in price_cols + size_cols:
        if col in df.columns: df.loc[combined_condition, col] = np.nan
    if combined_condition.sum() > 0: logger.info(f"[{symbol_context}] Set {combined_condition.sum()} rows to NaN based on flags.")
    for sz_col in size_cols:
        if sz_col in df.columns: df.loc[df[sz_col] > _ORDERBOOK_SIZE_CAP_THRESHOLD, sz_col] = np.nan

    current_ts_col_name = 'ts_col' if 'ts_col' in df.columns else 'ts'
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.name != 'ts':
        if current_ts_col_name in df.columns and pd.api.types.is_datetime64_any_dtype(df[current_ts_col_name]):
            df = df.set_index(current_ts_col_name, drop=False).sort_index(); df.index.name = 'ts'
            logger.info(f"[{symbol_context}] Set index to '{current_ts_col_name}' for imputation.")
        else:
            logger.warning(f"[{symbol_context}] Cannot set valid DatetimeIndex named 'ts'. Using simple ffill/bfill.")
            for col in price_cols: df[col] = df[col].ffill().bfill()
            for col in size_cols: df[col] = df[col].ffill().fillna(0)
            return df

    for col in price_cols: df[col] = df[col].ffill().bfill()
    for col in size_cols: df[col] = df[col].ffill().fillna(0)
    for flag_col in [col for col in df.columns if col.startswith('flag_')]: df[flag_col] = df[flag_col].fillna(False).astype(bool)
    logger.info(f"[{symbol_context}] Order book imputation complete.")
    return df

def reindex_orderbook_time(df_ob: pd.DataFrame, freq: str, symbol_context: str) -> pd.DataFrame:
    if df_ob.empty: logger.warning(f"[{symbol_context}] Input DataFrame for order book reindexing is empty."); return df_ob
    current_ts_col_name = 'ts_col' if 'ts_col' in df_ob.columns else 'ts'
    if not (isinstance(df_ob.index, pd.DatetimeIndex) and df_ob.index.name == 'ts'):
        if current_ts_col_name in df_ob.columns and pd.api.types.is_datetime64_any_dtype(df_ob[current_ts_col_name]):
            df_ob = df_ob.set_index(current_ts_col_name, drop=False).sort_index(); df_ob.index.name = 'ts'
            logger.info(f"[{symbol_context}] Set index to '{current_ts_col_name}' for order book reindexing.")
        else: raise ValueError("DataFrame must have DatetimeIndex 'ts' or valid 'ts_col'/'ts' for reindex_orderbook_time.")
    logger.info(f"[{symbol_context}] Reindexing order book to '{freq}' frequency...")
    min_ts, max_ts = df_ob.index.min(), df_ob.index.max()
    if pd.isna(min_ts) or pd.isna(max_ts): logger.error(f"[{symbol_context}] Min/Max timestamp NaT. Cannot reindex."); return df_ob
    full_range = pd.date_range(start=min_ts.floor(freq), end=max_ts.ceil(freq), freq=freq, name="ts")
    df_reindexed = df_ob.reindex(full_range, method='ffill')
    if 'symbol_id' in df_reindexed.columns: df_reindexed['symbol_id'] = df_reindexed['symbol_id'].ffill()
    else: df_reindexed['symbol_id'] = symbol_context

    # Ensure 'date' column is robustly converted/created
    df_reindexed['date'] = _robust_to_datetime(df_reindexed.index.to_series().dt.normalize(), f"{symbol_context}_reindexed_date_ob")

    for flag_col in [col for col in df_reindexed.columns if col.startswith('flag_')]: df_reindexed[flag_col] = df_reindexed[flag_col].fillna(False).astype(bool)

    # Ensure the 'ts' column (expected by schema) has the index values
    df_reindexed['ts'] = df_reindexed.index

    logger.info(f"[{symbol_context}] Order book reindexing complete. Original: {len(df_ob)}, new: {len(df_reindexed)}.")
    return df_reindexed

def validate_orderbook_schema(df_ob: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
   logger.info(f"[{symbol_context}] Validating Order Book schema...")
   try:
       if df_ob.index.name != "ts": df_ob.index.name = "ts"
       df_val = df_ob.copy()
       if 'ts' not in df_val.columns and isinstance(df_val.index, pd.DatetimeIndex):
            df_val['ts'] = _robust_to_datetime(df_val.index.to_series(), f"{symbol_context}_ts_validation_from_index")
       elif 'ts' in df_val.columns:
            df_val['ts'] = _robust_to_datetime(df_val['ts'], f"{symbol_context}_ts_validation_from_col")

       if 'date' in df_val.columns:
           df_val['date'] = _robust_to_datetime(df_val['date'], f"{symbol_context}_date_validation")
       elif isinstance(df_val.index, pd.DatetimeIndex): # Create date from index if missing
           df_val['date'] = _robust_to_datetime(df_val.index.to_series().dt.normalize(), f"{symbol_context}_date_creation_from_index")
       else: # If cannot create date, schema might fail or need adjustment
            logger.warning(f"[{symbol_context}] 'date' column missing and cannot be derived from index for OB validation.")


       for flag in ["flag_ob_bad_structure", "flag_spread_mad", "flag_mid_mad"]:
           if flag not in df_val.columns: df_val[flag] = False
       validated_df = ORDERBOOK_SCHEMA.validate(df_val, lazy=True)
       logger.info(f"[{symbol_context}] Order Book schema validation successful.")
       return validated_df
   except SchemaErrors as err:
       logger.error(f"[{symbol_context}] Order Book schema validation failed. Errors:\n{err.failure_cases.to_string()}")
       raise err
   except Exception as e:
       logger.error(f"[{symbol_context}] Unexpected error during Order Book schema validation: {e}")
       raise

def aggregate_orderbook_to_ohlcv(df_ob: pd.DataFrame, freq: str, symbol_context: str) -> Optional[pd.DataFrame]:
    logger.info(f"[{symbol_context}] Aggregating order book data to {freq} OHLCV...")
    if df_ob.empty or 'bid1_px' not in df_ob.columns or 'ask1_px' not in df_ob.columns:
        logger.warning(f"[{symbol_context}] OB data empty or missing L1 prices for aggregation.")
        return None
    df = df_ob.copy()
    current_ts_col_name = 'ts_col' if 'ts_col' in df.columns else 'ts' # If 'ts' was renamed from index
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.name != 'ts':
        if current_ts_col_name in df.columns and pd.api.types.is_datetime64_any_dtype(df[current_ts_col_name]):
            df = df.set_index(current_ts_col_name).sort_index(); df.index.name = 'ts'
        else: logger.error(f"[{symbol_context}] Valid DatetimeIndex ('ts') required for aggregation."); return None

    df['mid_price'] = (df['bid1_px'] + df['ask1_px']) / 2.0
    df.dropna(subset=['mid_price'], inplace=True)
    if df.empty: logger.warning(f"[{symbol_context}] No valid mid-prices to aggregate."); return None

    ohlc = df['mid_price'].resample(freq).ohlc(); ohlc.columns = ['ob_open', 'ob_high', 'ob_low', 'ob_close']

    # Ensure L1 size columns exist before attempting to use them
    bid1_sz_col, ask1_sz_col = 'bid1_sz', 'ask1_sz'
    if bid1_sz_col not in df.columns: df[bid1_sz_col] = 0
    if ask1_sz_col not in df.columns: df[ask1_sz_col] = 0

    df['l1_depth_bid_val'] = df['bid1_px'] * df[bid1_sz_col]
    df['l1_depth_ask_val'] = df['ask1_px'] * df[ask1_sz_col]
    df['total_l1_depth_value'] = df['l1_depth_bid_val'] + df['l1_depth_ask_val']
    df['depth_delta_abs'] = df['total_l1_depth_value'].diff().abs()

    volume_proxy = df['depth_delta_abs'].resample(freq).sum(); volume_proxy.name = 'ob_volume_proxy'
    trades_proxy = df['mid_price'].resample(freq).count(); trades_proxy.name = 'ob_trades_proxy'

    df_agg = ohlc.join(volume_proxy, how='left').join(trades_proxy, how='left')
    df_agg['ob_volume_proxy'] = df_agg['ob_volume_proxy'].fillna(0)
    df_agg['ob_trades_proxy'] = df_agg['ob_trades_proxy'].fillna(0).astype(int)
    for col in ['ob_open', 'ob_high', 'ob_low', 'ob_close']: df_agg[col] = df_agg[col].ffill().bfill()
    df_agg.index.name = 'time_period_start' # Align index name with OHLCV data
    logger.info(f"[{symbol_context}] OB aggregation to {freq} complete. Shape: {df_agg.shape}")
    return df_agg

# --- Data Filling and Persistence ---
def fill_missing_ohlcv_from_orderbook(df_ohlcv: pd.DataFrame, df_aggregated_ob: Optional[pd.DataFrame],
                                      symbol_context: str) -> pd.DataFrame:
    if df_aggregated_ob is None or df_aggregated_ob.empty:
        logger.info(f"[{symbol_context}] No aggregated order book data to fill OHLCV.")
        return df_ohlcv
    logger.info(f"[{symbol_context}] Filling missing OHLCV from aggregated order book...")
    df_ohlcv_indexed = df_ohlcv.copy()
    # Ensure df_ohlcv has 'time_period_start' index
    if not (isinstance(df_ohlcv_indexed.index, pd.DatetimeIndex) and df_ohlcv_indexed.index.name == 'time_period_start'):
        # Try to set from 'time_period_start' or 'time_period_start_col'
        time_col = 'time_period_start' if 'time_period_start' in df_ohlcv_indexed.columns else 'time_period_start_col'
        if time_col in df_ohlcv_indexed.columns and pd.api.types.is_datetime64_any_dtype(df_ohlcv_indexed[time_col]):
            df_ohlcv_indexed = df_ohlcv_indexed.set_index(time_col, drop=False).sort_index()
            df_ohlcv_indexed.index.name = 'time_period_start'
        else:
            logger.error(f"[{symbol_context}] OHLCV DataFrame has no suitable time index for merging.")
            return df_ohlcv

    df_agg_ob_indexed = df_aggregated_ob.copy()
    if df_agg_ob_indexed.index.name != 'time_period_start': # From aggregate_orderbook_to_ohlcv
        df_agg_ob_indexed.index.name = 'time_period_start'

    df_merged = pd.merge(df_ohlcv_indexed, df_agg_ob_indexed, on='time_period_start', how='left', suffixes=('', '_ob_fill'))
    fill_map = {
        'price_open': 'ob_open', 'price_high': 'ob_high', 'price_low': 'ob_low', 'price_close': 'ob_close',
        'volume_traded': 'ob_volume_proxy', 'trades_count': 'ob_trades_proxy'
    }
    for ohlcv_col, ob_col in fill_map.items():
        if ob_col in df_merged.columns:
            nan_before = df_merged[ohlcv_col].isnull().sum()
            # Only fill OHLCV NaNs with non-NaN OB data
            df_merged[ohlcv_col] = np.where(df_merged[ohlcv_col].isnull() & df_merged[ob_col].notnull(), df_merged[ob_col], df_merged[ohlcv_col])
            filled_count = nan_before - df_merged[ohlcv_col].isnull().sum()
            if filled_count > 0: logger.info(f"[{symbol_context}] Filled {filled_count} NaNs in '{ohlcv_col}' using '{ob_col}'.")
    cols_to_drop = [col for col in df_merged.columns if '_ob_fill' in col or col in fill_map.values()]
    df_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df_merged

def _df_to_sqlite_records(df: pd.DataFrame, columns: List[str]) -> List[tuple]:
    records = []
    for _, row in df.iterrows():
        record = []
        for col in columns:
            val = row.get(col) # Use .get() for safety if a column is unexpectedly missing
            if pd.isna(val): record.append(None)
            elif isinstance(val, (datetime, pd.Timestamp)): record.append(val.isoformat())
            elif isinstance(val, (bool, np.bool_)): record.append(int(val))
            elif isinstance(val, (np.integer)): record.append(int(val)) # Explicit int conversion
            elif isinstance(val, (np.floating)): record.append(float(val)) # Explicit float conversion
            else: record.append(val)
        records.append(tuple(record))
    return records

def persist_cleaned_ohlcv(df_cleaned_ohlcv: pd.DataFrame, db_path: str, symbol_context: str):
    logger.info(f"[{symbol_context}] Persisting {len(df_cleaned_ohlcv)} cleaned OHLCV rows...")
    target_cols = [
        "symbol", "time_period_start", "time_period_end", "time_open", "time_close",
        "price_open", "price_high", "price_low", "price_close", "volume_traded", "trades_count",
        "flag_bad_structure", "flag_outlier_fixed", "flag_outlier_mad", "flag_jump"
    ]
    df_to_persist = df_cleaned_ohlcv.copy()
    if 'flag_outlier_fixed' not in df_to_persist.columns: df_to_persist['flag_outlier_fixed'] = False
    if 'symbol' not in df_to_persist.columns: df_to_persist['symbol'] = symbol_context

    # Ensure 'time_period_start' (PK) is present as a column from index if needed
    if 'time_period_start' not in df_to_persist.columns and df_to_persist.index.name == 'time_period_start':
        df_to_persist['time_period_start'] = df_to_persist.index

    cols_to_persist = [col for col in target_cols if col in df_to_persist.columns]
    records = _df_to_sqlite_records(df_to_persist, cols_to_persist)
    if not records: logger.info(f"[{symbol_context}] No records to persist for cleaned OHLCV."); return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        placeholders = ", ".join(["?"] * len(cols_to_persist))
        sql = f"INSERT OR REPLACE INTO coinapi_ohlcv_clean ({', '.join(cols_to_persist)}) VALUES ({placeholders})"
        cur.executemany(sql, records)
        conn.commit()
        logger.info(f"[{symbol_context}] Inserted/replaced {len(records)} rows into coinapi_ohlcv_clean.")
    except sqlite3.Error as e: logger.error(f"[{symbol_context}] SQLite error persisting cleaned OHLCV: {e}")
    finally: conn.close()

def persist_cleaned_orderbook(df_cleaned_ob: pd.DataFrame, db_path: str, symbol_context: str):
    logger.info(f"[{symbol_context}] Persisting {len(df_cleaned_ob)} cleaned order book rows...")
    target_cols = [
        "symbol_id", "ts", "date",
        "bid1_px", "bid1_sz", "bid2_px", "bid2_sz", "bid3_px", "bid3_sz",
        "ask1_px", "ask1_sz", "ask2_px", "ask2_sz", "ask3_px", "ask3_sz",
        "flag_ob_bad_structure", "flag_spread_mad", "flag_mid_mad"
    ]
    df_to_persist = df_cleaned_ob.copy()
    if 'symbol_id' not in df_to_persist.columns:
        if 'symbol' in df_to_persist.columns: df_to_persist.rename(columns={'symbol': 'symbol_id'}, inplace=True)
        else: df_to_persist['symbol_id'] = symbol_context
    if 'ts' not in df_to_persist.columns and df_to_persist.index.name == 'ts': df_to_persist['ts'] = df_to_persist.index
    if 'date' in df_to_persist.columns:
        if pd.api.types.is_datetime64_any_dtype(df_to_persist['date']): df_to_persist['date'] = df_to_persist['date'].dt.strftime('%Y-%m-%d')
    elif 'ts' in df_to_persist.columns: df_to_persist['date'] = pd.to_datetime(df_to_persist['ts']).dt.strftime('%Y-%m-%d')

    cols_to_persist = [col for col in target_cols if col in df_to_persist.columns]
    records = _df_to_sqlite_records(df_to_persist, cols_to_persist)
    if not records: logger.info(f"[{symbol_context}] No records to persist for cleaned order book."); return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        placeholders = ", ".join(["?"] * len(cols_to_persist))
        sql = f"INSERT OR REPLACE INTO coinapi_orderbook_clean ({', '.join(cols_to_persist)}) VALUES ({placeholders})"
        cur.executemany(sql, records)
        conn.commit()
        logger.info(f"[{symbol_context}] Inserted/replaced {len(records)} rows into coinapi_orderbook_clean.")
    except sqlite3.Error as e: logger.error(f"[{symbol_context}] SQLite error persisting cleaned order book: {e}")
    finally: conn.close()

def setup_clean_db_tables(db_path: str, clear_existing_data: bool = True):
    logger.info(f"Setting up clean DB tables at {db_path}. Clear existing: {clear_existing_data}")
    initialize_database(db_path)
    if clear_existing_data:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM coinapi_ohlcv_clean;"); logger.info("Cleared coinapi_ohlcv_clean.")
            cur.execute("DELETE FROM coinapi_orderbook_clean;"); logger.info("Cleared coinapi_orderbook_clean.")
            conn.commit()
        except sqlite3.Error as e: logger.error(f"SQLite error clearing clean tables: {e}")
        finally: conn.close()
    logger.info("Clean DB tables setup complete.")

# --- New function for prerequisite ---
def load_cleaned_ohlcv_data(db_path: str, symbol: str,
                            start_date_iso: Optional[str] = None,
                            end_date_iso: Optional[str] = None) -> pd.DataFrame:
    """
    Loads cleaned OHLCV data (time_period_start, price_close) for a specific symbol
    from the coinapi_ohlcv_clean table, with optional date filtering.
    """
    logger.info(f"[{symbol}] Loading cleaned OHLCV data (time_period_start, price_close)...")
    conn = sqlite3.connect(db_path)

    params: List[Any] = [symbol]
    query = "SELECT time_period_start, price_close FROM coinapi_ohlcv_clean WHERE symbol = ?"

    if start_date_iso:
        query += " AND time_period_start >= ?"
        params.append(start_date_iso)
    if end_date_iso:
        # To include the whole end_date_iso, the query should be < next day's start or include time part
        # Assuming end_date_iso is just a date string YYYY-MM-DD, adjust to include the full day.
        # For simplicity, if time part is not included, it might just compare against YYYY-MM-DD 00:00:00.
        # A more robust way is to ensure end_date_iso includes time or add one day and use '<'.
        # For now, using it as is.
        query += " AND time_period_start <= ?"
        params.append(end_date_iso)

    query += " ORDER BY time_period_start ASC"

    try:
        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            logger.warning(f"[{symbol}] No cleaned OHLCV data found in coinapi_ohlcv_clean" +
                           (f" between {start_date_iso} and {end_date_iso}" if start_date_iso or end_date_iso else "") + ".")
            return pd.DataFrame(columns=['time_period_start', 'price_close'])

        df["time_period_start"] = _robust_to_datetime(df["time_period_start"], f"{symbol}_cleaned_time_period_start")
        df.dropna(subset=['time_period_start'], inplace=True) # Should not happen if DB stores valid datetimes

        # Price_close should already be numeric from DB
        df["price_close"] = pd.to_numeric(df["price_close"], errors='coerce')

        logger.info(f"[{symbol}] Loaded {len(df)} rows of cleaned OHLCV data.")
    except Exception as e:
        logger.error(f"[{symbol}] Failed to load cleaned OHLCV data from DB: {e}")
        return pd.DataFrame(columns=['time_period_start', 'price_close'])
    finally:
        conn.close()
    return df
