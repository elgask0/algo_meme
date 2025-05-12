import sqlite3
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import numpy as np
from scipy.stats import median_abs_deviation
import os
import argparse
from dotenv import load_dotenv
import logging
import traceback

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Carga de Variables de Entorno y Constantes ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
DEFAULT_FREQ_CANDLE = "5min"
DEFAULT_FREQ_ORDERBOOK = "5min"
DEFAULT_MAX_GAP_CANDLES = 6
ORDERBOOK_SIZE_CAP_THRESHOLD = 1e14

# --- Esquemas de Validación con Pandera ---
OHLCV_SCHEMA = DataFrameSchema({
    "symbol": Column(str, nullable=False),
    "time_period_start": Column(pa.DateTime, nullable=False, coerce=True),
    "time_period_end": Column(pa.DateTime, nullable=False, coerce=True),
    "time_open": Column(pa.DateTime, nullable=True, coerce=True),
    "time_close": Column(pa.DateTime, nullable=True, coerce=True),
    "price_open": Column(float, Check.ge(0), nullable=False, coerce=True),
    "price_high": Column(float, Check.ge(0), nullable=False, coerce=True),
    "price_low": Column(float, Check.ge(0), nullable=False, coerce=True),
    "price_close": Column(float, Check.ge(0), nullable=False, coerce=True),
    "volume_traded": Column(float, Check.ge(0), nullable=False, coerce=True),
    "trades_count": Column(int, Check.ge(0), nullable=False, coerce=True),
    "flag_bad_structure": Column(bool, coerce=True),
    "flag_outlier_mad": Column(bool, coerce=True),
    "flag_jump": Column(bool, coerce=True),
}, strict=False, ordered=False, name="OHLCVSchema")

ORDERBOOK_SCHEMA = DataFrameSchema({
    "symbol_id": Column(str, nullable=False),
    "ts": Column(pa.DateTime, nullable=False, coerce=True),
    "date": Column(pa.DateTime, nullable=True, coerce=True),
    "bid1_px": Column(float, Check.ge(0), nullable=False, coerce=True), "bid1_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "bid2_px": Column(float, Check.ge(0), nullable=False, coerce=True), "bid2_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "bid3_px": Column(float, Check.ge(0), nullable=False, coerce=True), "bid3_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "ask1_px": Column(float, Check.ge(0), nullable=False, coerce=True), "ask1_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "ask2_px": Column(float, Check.ge(0), nullable=False, coerce=True), "ask2_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "ask3_px": Column(float, Check.ge(0), nullable=False, coerce=True), "ask3_sz": Column(float, Check.gt(0), nullable=False, coerce=True),
    "flag_ob_bad_structure": Column(bool, coerce=True),
    "flag_mid_mad": Column(bool, coerce=True),
}, strict=False, ordered=False, name="OrderbookSchema")


# --- Funciones del Pipeline ---

def robust_to_datetime(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series): return pd.Series([], dtype='datetime64[ns]', name=getattr(series, 'name', 'unknown'))
    original_series = series.copy(); series_name = original_series.name if original_series.name else "Unnamed"
    if pd.api.types.is_datetime64_any_dtype(original_series):
        return original_series.dt.tz_convert(None) if getattr(original_series.dt, 'tz', None) is not None else original_series
    if original_series.empty: return pd.Series([], dtype='datetime64[ns]', name=series_name)
    converted = pd.to_datetime(original_series, errors='coerce', utc=True)
    if (nans1 := converted.isna().sum()) > 0 and nans1 < len(original_series):
        failed_mask = converted.isna(); numeric_failed = pd.to_numeric(original_series[failed_mask], errors='coerce')
        if numeric_failed.notna().any():
            for unit in ['s', 'ms']:
                current_failed_mask = converted.isna()
                if not current_failed_mask.any(): break
                numeric_still_failed = pd.to_numeric(original_series[current_failed_mask], errors='coerce')
                if not numeric_still_failed.notna().any(): continue
                try:
                    temp_converted = pd.to_datetime(numeric_still_failed, unit=unit, errors='coerce', utc=True)
                    converted.loc[current_failed_mask] = converted.loc[current_failed_mask].fillna(temp_converted)
                except (FloatingPointError, OverflowError) as e_of: logger.warning(f"'{series_name}': Overflow/FP error con unit='{unit}': {e_of}.")
                except Exception as e: logger.error(f"'{series_name}': Error inesperado con unit='{unit}': {e}", exc_info=False)
    current_nans_final = converted.isna()
    if current_nans_final.any() and current_nans_final.sum() < len(original_series):
        try:
            converted_t = pd.to_datetime(original_series[current_nans_final].astype(str).str.replace(' ', 'T', regex=False), errors='coerce', utc=True)
            converted.loc[current_nans_final] = converted.loc[current_nans_final].fillna(converted_t)
        except Exception: pass 
    if pd.api.types.is_datetime64_any_dtype(converted) and getattr(converted.dt, 'tz', None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted

def load_data(db_path: str, symbol: str) -> pd.DataFrame:
    logger.info(f"Cargando OHLCV para {symbol}")
    conn = sqlite3.connect(db_path); query = "SELECT * FROM coinapi_ohlcv WHERE symbol = ?"
    try:
        df = pd.read_sql(query, conn, params=[symbol]); rows_read = len(df)
        if df.empty: logger.warning(f"No datos OHLCV para {symbol}"); return pd.DataFrame()
        crit_cols = ["time_period_start", "time_period_end"]; nan_mask = pd.Series(False, index=df.index)
        for col in crit_cols:
            if col not in df.columns: logger.error(f"Falta col crítica '{col}'."); return pd.DataFrame()
            nan_mask |= df[col].isnull() | (df[col].astype(str).str.strip() == '' if pd.api.types.is_string_dtype(df[col]) else False)
        if (num_nulls := nan_mask.sum()) > 0:
            df = df[~nan_mask]; logger.warning(f"Eliminados {num_nulls} OHLCV con NULL/vacío en {crit_cols}.")
            if df.empty: return pd.DataFrame()
        time_cols = ["time_period_start", "time_period_end", "time_open", "time_close"]
        for col in time_cols:
            if col in df.columns: df[col] = robust_to_datetime(df[col])
            elif col in crit_cols: logger.error(f"Col crítica '{col}' desapareció."); return pd.DataFrame()
            else: df[col] = pd.NaT
        if (failed_conv := df[crit_cols].isna().any(axis=1).sum()) > 0:
            df = df.dropna(subset=crit_cols); logger.warning(f"Eliminados {failed_conv} OHLCV con NaT en {crit_cols} post-conversión.")
            if df.empty: return pd.DataFrame()
        if 'time_open' in df.columns: df['time_open'] = df['time_open'].fillna(df.get('time_period_start'))
        if 'time_close' in df.columns: df['time_close'] = df['time_close'].fillna(df.get('time_period_end'))
        price_cols = ["price_open", "price_high", "price_low", "price_close"]
        if existing_pc := [c for c in price_cols if c in df.columns]:
            for col in existing_pc: df[col] = pd.to_numeric(df[col], errors='coerce')
            if (dropped_ohlc := len(df) - len(df.dropna(subset=existing_pc))) > 0:
                df = df.dropna(subset=existing_pc); logger.info(f"Eliminadas {dropped_ohlc} filas OHLCV sin precios.")
        logger.info(f"Filas OHLCV válidas post-carga: {len(df)}")
    except Exception as e: logger.error(f"Error carga OHLCV {symbol}: {e}", exc_info=True); return pd.DataFrame()
    finally:
        if conn: conn.close()
    return df.copy()

def validate_schema(df: pd.DataFrame, schema, context_symbol: str) -> pd.DataFrame:
    s_name = schema.name if hasattr(schema, 'name') else 'Schema'
    logger.info(f"Validando {len(df)} filas de {context_symbol} contra {s_name}...")
    try: return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        logger.error(f"Errores validación {s_name} para {context_symbol}:\n{err.failure_cases.head(1)}")
        if err.data is not None: logger.error(f"DF que falló ({len(err.data)} filas):\n{err.data.head(1)}")
        raise
    except Exception as e: logger.error(f"Error inesperado validación {s_name} para {context_symbol}: {e}", exc_info=True); raise

def reindex_time(df: pd.DataFrame, freq_candle: str) -> pd.DataFrame:
    if df.empty or 'time_period_start' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['time_period_start']): return df.copy()
    df = df.dropna(subset=['time_period_start'])
    if df.empty or (len(df) < 2 and not (len(df) == 1 and df['time_period_start'].iloc[0] == df['time_period_start'].iloc[-1])): return df.copy()
    if not df['time_period_start'].is_unique: df = df.sort_values('time_period_start').drop_duplicates(subset=['time_period_start'], keep='last')
    df = df.set_index("time_period_start").sort_index()
    min_ts, max_ts = df.index.min(), df.index.max()
    if pd.isna(min_ts) or pd.isna(max_ts) or min_ts > max_ts: return df.reset_index().copy()
    try:
        full_index = pd.date_range(start=min_ts, end=max_ts, freq=freq_candle)
        if full_index.empty and min_ts == max_ts: full_index = pd.DatetimeIndex([min_ts])
        elif full_index.empty: full_index = df.index
    except ValueError: return df.reset_index().copy()
    out = df.reindex(full_index).reset_index().rename(columns={"index": "time_period_start"})
    if "symbol" in out.columns: out["symbol"] = out["symbol"].fillna(df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else "UNKNOWN")
    try: out["time_period_end"] = out["time_period_start"] + pd.Timedelta(freq_candle)
    except ValueError: out["time_period_end"] = pd.NaT
    out["time_open"] = out.get("time_open", pd.Series(dtype='datetime64[ns]')).fillna(out["time_period_start"])
    out["time_close"] = out.get("time_close", pd.Series(dtype='datetime64[ns]')).fillna(out["time_period_end"])
    out["volume_traded"] = out.get("volume_traded", pd.Series(dtype=float)).fillna(0)
    out["trades_count"] = pd.to_numeric(out.get("trades_count"), errors='coerce').fillna(0).astype(int)
    for flag_col_name in ["flag_bad_structure", "flag_outlier_mad", "flag_jump"]:
        if flag_col_name in out.columns: 
            out[flag_col_name] = out[flag_col_name].fillna(False).astype(bool)
    logger.info(f"Reindexado OHLCV: {len(df)} -> {len(out)} filas.")
    return out.copy()

def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()
    df_c["flag_bad_structure"] = False 
    if df_c.empty: return df_c
    price_cols = ["price_open", "price_high", "price_low", "price_close"]; vol_cols = ["volume_traded", "trades_count"]
    for col in price_cols + vol_cols:
        if col not in df_c.columns: df_c[col] = np.nan
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
    conds = (df_c.price_high >= df_c.price_open) & (df_c.price_high >= df_c.price_low) & (df_c.price_high >= df_c.price_close) & \
            (df_c.price_low <= df_c.price_open) & (df_c.price_low <= df_c.price_high) & (df_c.price_low <= df_c.price_close) & \
            (df_c.volume_traded >= 0) & (df_c.trades_count >= 0) & df_c[price_cols].notna().all(axis=1)
    df_c["flag_bad_structure"] = ~conds
    if (num_bad := df_c["flag_bad_structure"].sum()) > 0: logger.info(f"{num_bad} filas OHLCV con estructura inválida.")
    return df_c

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()
    df_c["flag_outlier_mad"] = False; df_c["flag_jump"] = False 
    if df_c.empty: return df_c
    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    existing_pc = [c for c in price_cols if c in df_c.columns]
    if not existing_pc: return df_c
    for col in existing_pc: df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
    def flag_mad_daily(sub):
        if "flag_outlier_mad" not in sub.columns: sub = sub.assign(flag_outlier_mad=False)
        else: sub["flag_outlier_mad"] = False 
        if sub.empty or sub[existing_pc].isna().all().all(): return sub
        mask = pd.Series(False, index=sub.index)
        for col in existing_pc:
            s = sub[col].dropna();
            if len(s) < 5: continue
            mad_val = median_abs_deviation(s, scale='normal', nan_policy='omit')
            if pd.isna(mad_val): continue
            mad = mad_val if mad_val > 1e-9 else 1e-9
            mask |= (~sub[col].between(s.median() - 5*mad, s.median() + 5*mad)).fillna(False)
        sub["flag_outlier_mad"] = mask; return sub
    if 'time_period_start' in df_c.columns and pd.api.types.is_datetime64_any_dtype(df_c['time_period_start']):
        if not df_c.dropna(subset=['time_period_start']).empty:
             df_c = df_c.groupby(df_c.time_period_start.dt.date, group_keys=False).apply(flag_mad_daily)
    if 'price_open' in df_c.columns and not df_c['price_open'].isna().all():
        df_c = df_c.sort_values('time_period_start')
        delta = df_c.price_open.diff().abs()
        sigma = delta.rolling(window=30, min_periods=15).std().replace(0,np.nan).ffill().bfill()
        df_c["flag_jump"] = ((delta > (10 * sigma)) & sigma.notna()).fillna(False)
    logger.info(f"Outliers OHLCV MAD: {df_c['flag_outlier_mad'].sum()}, Jumps: {df_c['flag_jump'].sum()}")
    return df_c

def impute_data(df: pd.DataFrame, max_gap_candles: int) -> pd.DataFrame:
    if df.empty: return df.copy()
    df_c = df.copy(); price_cols = ["price_open", "price_high", "price_low", "price_close"]
    existing_pc = [c for c in price_cols if c in df_c.columns]
    if "flag_outlier_mad" in df_c.columns and existing_pc and pd.api.types.is_bool_dtype(df_c["flag_outlier_mad"]):
        if (num_nan := df_c["flag_outlier_mad"].sum()) > 0:
            df_c.loc[df_c["flag_outlier_mad"], existing_pc] = np.nan
    if 'time_period_start' not in df_c.columns or not pd.api.types.is_datetime64_any_dtype(df_c['time_period_start']):
        if existing_pc: df_c[existing_pc] = df_c[existing_pc].ffill().bfill()
        for col in ["volume_traded", "trades_count"]:
            if col in df_c.columns: df_c[col] = df_c[col].fillna(0)
            if col == 'trades_count' and col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(0).astype(int)
        return df_c
    if not df_c['time_period_start'].is_unique: df_c = df_c.sort_values('time_period_start').drop_duplicates(subset=['time_period_start'], keep='last')
    df_c = df_c.set_index("time_period_start").sort_index()
    if existing_pc: df_c[existing_pc] = df_c[existing_pc].interpolate(method="time", limit=max_gap_candles, limit_area="inside")
    if "volume_traded" in df_c.columns: df_c["volume_traded"] = df_c["volume_traded"].fillna(0)
    if "trades_count" in df_c.columns: df_c["trades_count"] = pd.to_numeric(df_c["trades_count"], errors='coerce').fillna(0).astype(int)
    if existing_pc: df_c[existing_pc] = df_c[existing_pc].ffill().bfill() 
    if all(c in df_c.columns for c in price_cols):
        df_c["price_high"] = df_c[price_cols].max(axis=1)
        df_c["price_low"] = df_c[price_cols].min(axis=1)
    return df_c.reset_index()

def reduce_columns(df: pd.DataFrame, final_cols: list) -> pd.DataFrame:
    existing = [c for c in final_cols if c in df.columns]
    if missing := [c for c in final_cols if c not in existing]: 
        logger.warning(f"Faltan columnas finales para reducir: {missing}.")
    return df[existing].copy()

def final_data_check(df: pd.DataFrame, source: str = "Data") -> None:
    tag = f"[{source}][FinalCheck]"
    if not isinstance(df, pd.DataFrame) or df.empty: logger.warning(f"{tag} DataFrame vacío."); return
    logger.info(f"{tag} Filas: {len(df)}, Columnas: {len(df.columns)}")
    if (nulls := df.isna().sum())[lambda x: x > 0].empty: logger.info(f"{tag} No NaNs en DataFrame final.")
    else: logger.warning(f"{tag} ¡ALERTA! NaNs en DataFrame final:\n{nulls[nulls > 0]}")

# --- Funciones Pipeline Orderbook ---
def load_orderbook(db_path: str, symbol: str) -> pd.DataFrame:
    logger.info(f"Cargando OB para {symbol}")
    conn = sqlite3.connect(db_path); query = "SELECT * FROM coinapi_orderbook WHERE symbol_id = ?"
    try:
        df = pd.read_sql(query, conn, params=[symbol]); rows_read = len(df)
        if df.empty: return pd.DataFrame()
        nan_mask = df['ts'].isnull() | (df['ts'].astype(str).str.strip() == '' if pd.api.types.is_string_dtype(df['ts']) else False)
        if (num_nulls := nan_mask.sum()) > 0: df = df[~nan_mask]; logger.warning(f"Eliminados {num_nulls} OB con 'ts' NULL/vacío.")
        if df.empty: return pd.DataFrame()
        df['ts'] = robust_to_datetime(df['ts'])
        if (failed_conv := df['ts'].isna().sum()) > 0: df = df.dropna(subset=['ts']); logger.warning(f"Eliminados {failed_conv} OB con 'ts' NaT.")
        if df.empty: return pd.DataFrame()
        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        req_cols = [f"{s}{l}_{t}" for s in ["bid","ask"] for l in [1,2,3] for t in ["px","sz"]]
        if missing := [c for c in req_cols if c not in df.columns]: logger.error(f"Faltan cols OB: {missing}."); return pd.DataFrame()
        for col in req_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        if (dropped := len(df) - len(df.dropna(subset=req_cols))) > 0:
            df = df.dropna(subset=req_cols); logger.warning(f"Eliminados {dropped} OB con NaNs en px/sz.")
        if df.empty: return pd.DataFrame()
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep='last')
        logger.info(f"Carga OB completada. Filas válidas: {len(df)}")
    except Exception as e: logger.error(f"Error carga OB {symbol}: {e}", exc_info=True); return pd.DataFrame()
    finally:
        if conn: conn.close()
    return df.reset_index(drop=True).copy()

def detect_ob_structure(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()
    df_c["flag_ob_bad_structure"] = False 
    if df_c.empty: return df_c
    px_cols = [f"{s}{l}_px" for s in ["bid","ask"] for l in [1,2,3]]; sz_cols = [c for c in df_c.columns if c.endswith("_sz")]
    if any(c not in df_c.columns for c in px_cols): df_c["flag_ob_bad_structure"]=True; return df_c
    for col in px_cols + sz_cols:
        if col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
    cond = (df_c.bid1_px >= df_c.bid2_px) & (df_c.bid2_px >= df_c.bid3_px) & \
           (df_c.ask1_px <= df_c.ask2_px) & (df_c.ask2_px <= df_c.ask3_px) & \
           (df_c.bid1_px < df_c.ask1_px)
    if sz_cols: cond &= (df_c[sz_cols] > 1e-9).all(axis=1)
    cond &= df_c[[c for c in px_cols + sz_cols if c in df_c.columns]].notna().all(axis=1)
    df_c["flag_ob_bad_structure"] = ~cond
    if (nbad := df_c.flag_ob_bad_structure.sum()) > 0: logger.info(f"{nbad} filas OB con estructura inválida.")
    return df_c

def detect_ob_outliers(df: pd.DataFrame) -> pd.DataFrame: 
    df_c = df.copy() 
    df_c["flag_mid_mad"] = False 
    req_cols = ["bid1_px", "ask1_px", "ts"]
    if df_c.empty or not all(c in df_c.columns for c in req_cols) or not pd.api.types.is_datetime64_any_dtype(df_c['ts']):
        return df_c 
    df_c["mid_px"] = (df_c.bid1_px + df_c.ask1_px) / 2
    def flag_day(sub):
        if "flag_mid_mad" not in sub.columns: sub = sub.assign(flag_mid_mad=False)
        else: sub["flag_mid_mad"] = False 
        if sub.empty: return sub
        s_mid = sub.mid_px.dropna()[lambda x: x > 0]
        if len(s_mid) >= 5:
            mad_m = median_abs_deviation(s_mid, scale='normal', nan_policy='omit')
            if not pd.isna(mad_m): sub["flag_mid_mad"] = ~sub.mid_px.between(s_mid.median() - 5*(mad_m if mad_m >1e-9 else 1e-9), s_mid.median() + 5*(mad_m if mad_m >1e-9 else 1e-9)) | sub.mid_px.isna() | (sub.mid_px <= 0)
            else: sub["flag_mid_mad"] = sub.mid_px.isna() | (sub.mid_px <= 0)
        else: sub["flag_mid_mad"] = sub.mid_px.isna() | (sub.mid_px <= 0)
        sub["flag_mid_mad"] = sub["flag_mid_mad"].fillna(True); return sub
    df_c = df_c.groupby(df_c.ts.dt.date, group_keys=False).apply(flag_day)
    logger.info(f"Outliers Mid-Price MAD (OB): {df_c['flag_mid_mad'].sum()}")
    return df_c

def reindex_ob(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or 'ts' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['ts']): return df.copy()
    df = df.dropna(subset=['ts'])
    if df.empty or (len(df) < 2 and not (len(df) == 1 and df['ts'].iloc[0] == df['ts'].iloc[-1])): return df.copy()
    if not df['ts'].is_unique: df = df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
    df = df.set_index("ts").sort_index()
    min_ts, max_ts = df.index.min(), df.index.max()
    if pd.isna(min_ts) or pd.isna(max_ts) or min_ts > max_ts: return df.reset_index().copy()
    try:
        full_index = pd.date_range(start=min_ts, end=max_ts, freq=freq)
        if full_index.empty and min_ts == max_ts: full_index = pd.DatetimeIndex([min_ts])
        elif full_index.empty: full_index = df.index
    except ValueError: return df.reset_index().copy()
    out = df.reindex(full_index, method="ffill").rename_axis("ts").reset_index()
    if "symbol_id" in out.columns: out["symbol_id"] = out["symbol_id"].fillna(df["symbol_id"].iloc[0] if "symbol_id" in df.columns and not df.empty else "UNKNOWN_OB")
    for flag_col_name in ["flag_ob_bad_structure", "flag_mid_mad"]:
        if flag_col_name not in out.columns: out[flag_col_name] = False 
        out[flag_col_name] = out[flag_col_name].fillna(False).astype(bool)
    if 'date' in out.columns: out['date'] = out['date'].ffill().bfill()
    logger.info(f"Reindexado OB: {len(df)} -> {len(out)} filas.")
    return out.copy()

def impute_ob(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df_c = df.copy(); px_cols = [c for c in df_c.columns if c.endswith("_px") and c != "mid_px"]; sz_cols = [c for c in df_c.columns if c.endswith("_sz")]
    bad_mask = pd.Series(False, index=df_c.index)
    for flag in ["flag_ob_bad_structure", "flag_mid_mad"]:
        if flag in df_c.columns and pd.api.types.is_bool_dtype(df_c[flag]): bad_mask |= df_c[flag]
    if sz_cols:
        for sz_col in sz_cols:
            if sz_col in df_c.columns:
                large_sizes = df_c[sz_col] >= ORDERBOOK_SIZE_CAP_THRESHOLD
                if large_sizes.any():
                    logger.warning(f"Capeando {large_sizes.sum()} tamaños en '{sz_col}' >= {ORDERBOOK_SIZE_CAP_THRESHOLD:.0e}")
                    df_c.loc[large_sizes, sz_col] = np.nan
    cols_to_nanify = [c for c in px_cols + sz_cols if c in df_c.columns]
    if (num_to_nan := bad_mask.sum()) > 0 and cols_to_nanify:
        df_c.loc[bad_mask, cols_to_nanify] = np.nan
    if 'ts' not in df_c.columns or not pd.api.types.is_datetime64_any_dtype(df_c['ts']):
        if cols_to_nanify: df_c[cols_to_nanify] = df_c[cols_to_nanify].ffill().bfill()
        if sz_cols: df_c[sz_cols] = df_c[sz_cols].fillna(0)
    else:
        if not df_c['ts'].is_unique: df_c = df_c.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
        df_c = df_c.set_index("ts").sort_index()
        if px_cols: df_c[px_cols] = df_c[px_cols].ffill().bfill()
        if sz_cols: df_c[sz_cols] = df_c[sz_cols].ffill().fillna(0)
        if "bid1_px" in df_c.columns and "ask1_px" in df_c.columns: 
            df_c["mid_px"] = (df_c["bid1_px"] + df_c["ask1_px"]) / 2
            df_c["mid_px"] = df_c["mid_px"].ffill().bfill()
        df_c = df_c.reset_index()
    if 'spread' in df_c.columns: df_c = df_c.drop(columns=['spread'])
    return df_c

def aggregate_ob_to_candle(df_ob: pd.DataFrame, freq: str) -> pd.DataFrame | None:
    req_cols = ['ts', 'mid_px'] + [f"{s}{l}_sz" for s in ["bid","ask"] for l in [1,2,3]]
    if df_ob.empty or not all(c in df_ob.columns for c in req_cols) or not pd.api.types.is_datetime64_any_dtype(df_ob['ts']): return None
    df = df_ob.copy(); 
    if not df['ts'].is_unique: df = df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
    df = df.set_index('ts').sort_index(); 
    if df.empty: return None
    ohlc = df['mid_px'].resample(freq).ohlc()
    df['bid_qty'] = df[['bid1_sz', 'bid2_sz', 'bid3_sz']].sum(axis=1)
    df['ask_qty'] = df[['ask1_sz', 'ask2_sz', 'ask3_sz']].sum(axis=1)
    df['total_sz'] = df['bid_qty'] + df['ask_qty']
    df['delta_sz'] = df['total_sz'].diff().abs() if len(df) > 1 else 0
    agg = ohlc.join(df['delta_sz'].resample(freq).sum().rename("volume_proxy"), how='left')\
              .join(df['mid_px'].resample(freq).count().rename("trades_proxy"), how='left')
    agg[['volume_proxy', 'trades_proxy']] = agg[['volume_proxy', 'trades_proxy']].fillna(0)
    agg['trades_proxy'] = pd.to_numeric(agg['trades_proxy'], errors='coerce').fillna(0).astype(int)
    agg = agg.reset_index().rename(columns={"ts": "time_period_start", "open": "ob_open", "high": "ob_high", "low": "ob_low", "close": "ob_close"})
    if (dropped := len(agg) - len(agg.dropna(subset=["ob_open", "ob_high", "ob_low", "ob_close"], how='all'))) > 0:
         agg = agg.dropna(subset=["ob_open", "ob_high", "ob_low", "ob_close"], how='all')
    logger.info(f"Agregación OB: {len(agg)} velas generadas.")
    return agg.copy()

def fill_ohlcv_from_ob(df_ohlcv: pd.DataFrame, df_ob_agg: pd.DataFrame) -> pd.DataFrame:
    if df_ohlcv.empty: return df_ohlcv.copy()
    if df_ob_agg is None or df_ob_agg.empty: return df_ohlcv.copy()
    df = df_ohlcv.copy()
    for df_check, name in [(df, "OHLCV"), (df_ob_agg, "OB Agg")]:
        if 'time_period_start' not in df_check.columns or not pd.api.types.is_datetime64_any_dtype(df_check['time_period_start']):
            logger.error(f"'time_period_start' en {name} inválido."); return df
    df_ob_agg_dedup = df_ob_agg.drop_duplicates(subset=['time_period_start'], keep='last')
    df_merged = pd.merge(df, df_ob_agg_dedup, on="time_period_start", how="left", suffixes=("", "_ob"))
    mapping = {"price_open": "ob_open", "price_high": "ob_high", "price_low": "ob_low", "price_close": "ob_close",
               "volume_traded": "volume_proxy", "trades_count": "trades_proxy"}
    for target, source in mapping.items():
        if source in df_merged.columns and target in df_merged.columns:
            fill_values = df_merged.loc[df_merged[target].isna(), source].dropna()
            df_merged.loc[fill_values.index, target] = fill_values
    cols_to_drop = [s for t,s in mapping.items() if s != t and s in df_merged.columns]
    if cols_to_drop: df_merged = df_merged.drop(columns=list(set(cols_to_drop)))
    return df_merged.copy()

def ob_data_check(df: pd.DataFrame) -> None: final_data_check(df, source="OB")

# --- Main Execution & Persistence ---
def setup_clean_tables(db_path):
     logger.info(f"Verificando/Creando tablas _clean en {db_path}...")
     conn = None
     try:
        conn = sqlite3.connect(db_path); cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_ohlcv_clean (
            symbol TEXT NOT NULL, time_period_start TEXT NOT NULL, time_period_end TEXT,
            time_open TEXT, time_close TEXT, price_open REAL, price_high REAL, price_low REAL,
            price_close REAL, volume_traded REAL, trades_count INTEGER, flag_bad_structure INTEGER,
            flag_outlier_mad INTEGER, flag_jump INTEGER, PRIMARY KEY (symbol, time_period_start)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_orderbook_clean (
            symbol_id TEXT NOT NULL, ts TEXT NOT NULL, date TEXT, bid1_px REAL, bid1_sz REAL,
            bid2_px REAL, bid2_sz REAL, bid3_px REAL, bid3_sz REAL, ask1_px REAL, ask1_sz REAL,
            ask2_px REAL, ask2_sz REAL, ask3_px REAL, ask3_sz REAL, flag_ob_bad_structure INTEGER,
            flag_mid_mad INTEGER, PRIMARY KEY (symbol_id, ts)
        );""")
        conn.commit(); logger.info("Tablas _clean verificadas/creadas.")
     except Exception as e: logger.error(f"Error al verificar/crear tablas _clean: {e}", exc_info=True);
     finally:
        if conn: conn.close()

def persist_data(df: pd.DataFrame, db_path: str, table_name: str, symbol: str, data_type: str):
    if df.empty: logger.warning(f"[{data_type}] DataFrame vacío para {symbol}. No se persiste."); return
    logger.info(f"[{data_type}] Persistiendo {len(df)} filas para {symbol} en {table_name}...")
    cols_map = {
        'coinapi_ohlcv_clean': ["symbol","time_period_start","time_period_end","time_open","time_close","price_open","price_high","price_low","price_close","volume_traded","trades_count","flag_bad_structure","flag_outlier_mad","flag_jump"],
        'coinapi_orderbook_clean': ["symbol_id","ts","date","bid1_px","bid1_sz","bid2_px","bid2_sz","bid3_px","bid3_sz","ask1_px","ask1_sz","ask2_px","ask2_sz","ask3_px","ask3_sz","flag_ob_bad_structure","flag_mid_mad"]
    }
    if table_name not in cols_map: logger.error(f"[{data_type}] Tabla desconocida: {table_name}"); return
    cols_to_persist = [c for c in cols_map[table_name] if c in df.columns]
    if not cols_to_persist: logger.error(f"[{data_type}] No hay columnas válidas para persistir."); return

    df_persist = df[cols_to_persist].copy()
    try:
        for col in df_persist.select_dtypes(include=['datetime64[ns]']).columns:
             df_persist[col] = df_persist[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(x) else None)
        for col in df_persist.select_dtypes(include=['bool']).columns: df_persist[col] = df_persist[col].astype(int)
        for col in df_persist.columns:
             if col.endswith(('_px','_sz')) or col.startswith('price_') or col=='volume_traded': df_persist[col] = pd.to_numeric(df_persist[col], errors='coerce')
             elif col=='trades_count' or col.startswith('flag_'): df_persist[col] = pd.to_numeric(df_persist[col],errors='coerce').fillna(0).astype(int)
    except Exception as e: logger.error(f"[{data_type}] Error convirtiendo tipos para {symbol}: {e}"); raise RuntimeError(f"Fallo conversión tipos {symbol}") from e
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        tuples = [tuple(None if pd.isna(x) else x for x in r) for r in df_persist.itertuples(index=False, name=None)]
        if not tuples: logger.warning(f"[{data_type}] No hay tuplas para insertar para {symbol}."); return
        placeholders = ','.join('?' * len(cols_to_persist))
        sql = f"INSERT OR REPLACE INTO {table_name} ({','.join(cols_to_persist)}) VALUES ({placeholders})"
        conn.executemany(sql, tuples); conn.commit()
        logger.info(f"[{data_type}] {table_name}: {len(tuples)} filas insertadas/reemplazadas para {symbol}")
    except Exception as e: logger.error(f"[{data_type}] Error persistiendo en {table_name} para {symbol}: {e}"); raise
    finally:
        if conn: conn.close()

def main():
    parser = argparse.ArgumentParser(description="Pipeline de limpieza de datos OHLCV y orderbook", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", "-s", nargs="+", help="Símbolo(s) a procesar.")
    parser.add_argument("--db-file", "-d", default=DEFAULT_DB_FILE, help="Ruta a BD SQLite.")
    parser.add_argument("--freq-candle", default=DEFAULT_FREQ_CANDLE, help="Frecuencia velas OHLCV.")
    parser.add_argument("--freq-orderbook", default=DEFAULT_FREQ_ORDERBOOK, help="Frecuencia snapshots OB.")
    parser.add_argument("--max-gap-candles", type=int, default=DEFAULT_MAX_GAP_CANDLES, help="Máx. velas OHLCV a interpolar.")
    parser.add_argument("--skip-ohlcv", action="store_true", help="Omitir limpieza OHLCV.")
    parser.add_argument("--skip-orderbook", action="store_true", help="Omitir limpieza Orderbook.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Nivel de logs.")
    args = parser.parse_args()
    
    try: 
        log_level_upper = args.log_level.upper()
        logging.getLogger().setLevel(log_level_upper); logger.setLevel(log_level_upper)
        log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s [%(funcName)s] %(message)s")
        for handler in logging.getLogger().handlers: handler.setFormatter(log_formatter)
        logger.info(f"Nivel de logging: {log_level_upper}")
    except ValueError: logger.error(f"Nivel de logging inválido: {args.log_level}. Usando INFO."); logging.getLogger().setLevel(logging.INFO); logger.setLevel(logging.INFO)

    db_path, freq_candle, freq_orderbook, max_gap_candles = args.db_file, args.freq_candle, args.freq_orderbook, args.max_gap_candles
    symbols_to_process = []
    try: 
        logger.info(f"Verificando BD: {db_path}")
        if not os.path.exists(db_path): logger.critical(f"BD NO EXISTE: {db_path}"); return
        conn_check = sqlite3.connect(db_path); cur_check = conn_check.cursor()
        def table_exists(c, tn): c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;",(tn,)); return c.fetchone() is not None
        ohlcv_exists, ob_exists = table_exists(cur_check, 'coinapi_ohlcv'), table_exists(cur_check, 'coinapi_orderbook')
        if args.symbol:
            symbols_to_process = args.symbol; logger.info(f"Símbolos especificados: {symbols_to_process}")
            if not args.skip_ohlcv and ohlcv_exists:
                 cur_check.execute(f"SELECT DISTINCT symbol FROM coinapi_ohlcv WHERE symbol IN ({','.join('?'*len(symbols_to_process))})", symbols_to_process)
                 found = {r[0] for r in cur_check.fetchall()}
                 if missing := [s for s in symbols_to_process if s not in found]: logger.warning(f"Símbolos no en coinapi_ohlcv: {missing}")
                 symbols_to_process = [s for s in symbols_to_process if s in found]
            elif not args.skip_ohlcv: logger.error("'coinapi_ohlcv' no existe."); symbols_to_process = []
        elif ohlcv_exists: cur_check.execute("SELECT DISTINCT symbol FROM coinapi_ohlcv;"); symbols_to_process = [r[0] for r in cur_check.fetchall()]; logger.info(f"Símbolos en coinapi_ohlcv: {len(symbols_to_process)}")
        else: logger.error("'coinapi_ohlcv' no existe y no se especificaron símbolos."); symbols_to_process = []
        if not args.skip_ohlcv and not ohlcv_exists: logger.warning("'coinapi_ohlcv' no existe. Saltando OHLCV."); args.skip_ohlcv = True
        if not args.skip_orderbook and not ob_exists: logger.warning("'coinapi_orderbook' no existe. Saltando OB."); args.skip_orderbook = True
        conn_check.close()
    except Exception as e: logger.critical(f"Error config inicial: {e}", exc_info=True); return
    if not symbols_to_process: logger.critical("No hay símbolos válidos para procesar."); return
    
    setup_clean_tables(db_path) 

    total_symbols, ohlcv_ok, ob_ok, skipped_symbols = len(symbols_to_process), 0, 0, []
    for i, symbol in enumerate(symbols_to_process): 
        logger.info(f"========== Procesando Símbolo {i+1}/{total_symbols}: {symbol} ==========")
        symbol_failed, df_ob_agg = False, None
        if not args.skip_orderbook:
            logger.info(f"--- Iniciando Pipeline Orderbook para {symbol} ---")
            try:
                df_ob = load_orderbook(db_path, symbol)
                if not df_ob.empty:
                    df_ob = detect_ob_structure(df_ob) 
                    df_ob = detect_ob_outliers(df_ob) 
                    df_ob = validate_schema(df_ob, ORDERBOOK_SCHEMA, symbol) 
                    df_ob = reindex_ob(df_ob, freq_orderbook)
                    df_ob = impute_ob(df_ob)
                    ob_data_check(df_ob)
                    if not df_ob.empty:
                        persist_data(df_ob, db_path, 'coinapi_orderbook_clean', symbol, 'OB'); ob_ok += 1
                        df_ob_agg = aggregate_ob_to_candle(df_ob, freq_candle)
                    else: logger.warning(f"Orderbook vacío después de la limpieza para {symbol}.")
                else: logger.warning(f"No se cargaron datos válidos de orderbook para {symbol}.")
            except Exception as e: logger.error(f"Error FataL pipeline OB {symbol}: {e}", exc_info=True); symbol_failed = True
        
        if not args.skip_ohlcv and not symbol_failed:
            logger.info(f"--- Iniciando Pipeline OHLCV para {symbol} ---")
            try:
                df_ohlcv = load_data(db_path, symbol)
                if not df_ohlcv.empty:
                    df_ohlcv = df_ohlcv.sort_values("time_period_start").drop_duplicates(subset=["time_period_start"], keep='last')
                    df_ohlcv = reindex_time(df_ohlcv, freq_candle) 
                    if df_ohlcv.empty: logger.warning(f"OHLCV vacío después de reindexar {symbol}."); symbol_failed = True
                    if not df_ohlcv.empty: 
                        if df_ob_agg is not None and not df_ob_agg.empty: df_ohlcv = fill_ohlcv_from_ob(df_ohlcv, df_ob_agg)
                        df_ohlcv = detect_structure(df_ohlcv) 
                        df_ohlcv = detect_outliers(df_ohlcv)  
                        df_ohlcv = impute_data(df_ohlcv, max_gap_candles) # Imputar ANTES de validar
                        df_ohlcv = validate_schema(df_ohlcv, OHLCV_SCHEMA, symbol) 
                        
                        df_ohlcv = detect_structure(df_ohlcv) # Re-check estructura post-imputación
                        if "flag_bad_structure" in df_ohlcv.columns and pd.api.types.is_bool_dtype(df_ohlcv["flag_bad_structure"]):
                             if (dropped := len(df_ohlcv) - len(df_ohlcv[~df_ohlcv["flag_bad_structure"]])) > 0:
                                 df_ohlcv = df_ohlcv[~df_ohlcv["flag_bad_structure"]]
                                 logger.info(f"Eliminadas {dropped} filas OHLCV con mala estructura post-imputación.")
                        
                        price_cols_final = ["price_open","price_high","price_low","price_close"]
                        existing_pc_final = [c for c in price_cols_final if c in df_ohlcv.columns]
                        if existing_pc_final:
                             if (dropped_nan_final := len(df_ohlcv) - len(df_ohlcv.dropna(subset=existing_pc_final))) > 0:
                                 df_ohlcv = df_ohlcv.dropna(subset=existing_pc_final)
                                 logger.warning(f"Eliminadas {dropped_nan_final} filas OHLCV con NaNs en precios antes de persistir.")
                        
                        if not df_ohlcv.empty:
                            final_cols_ohlcv = ["symbol","time_period_start","time_period_end","time_open","time_close","price_open","price_high","price_low","price_close","volume_traded","trades_count","flag_bad_structure","flag_outlier_mad","flag_jump"]
                            df_final_ohlcv = reduce_columns(df_ohlcv, final_cols_ohlcv) 
                            final_data_check(df_final_ohlcv, source="OHLCV")
                            persist_data(df_final_ohlcv, db_path, 'coinapi_ohlcv_clean', symbol, 'OHLCV'); ohlcv_ok += 1
                        else: logger.warning(f"OHLCV final vacío para {symbol}."); symbol_failed = True
                else: logger.warning(f"No se cargaron datos válidos de OHLCV para {symbol}."); symbol_failed = True
            except Exception as e: logger.error(f"Error FataL pipeline OHLCV {symbol}: {e}", exc_info=True); symbol_failed = True
        
        if symbol_failed: skipped_symbols.append(symbol); logger.error(f"Símbolo {symbol} marcado como fallido.")
        else: logger.info(f"Símbolo {symbol} procesado correctamente.")

    logger.info(f"{'='*20} Resumen del Proceso {'='*20}")
    logger.info(f"Símbolos totales intentados: {total_symbols}")
    if not args.skip_orderbook: logger.info(f"Orderbook procesado (persistencia activada): {ob_ok} OK")
    if not args.skip_ohlcv: logger.info(f"OHLCV procesado (persistencia activada): {ohlcv_ok} OK")
    if unique_skipped := sorted(list(set(skipped_symbols))): logger.warning(f"Símbolos con ERRORES ({len(unique_skipped)}): {unique_skipped}")
    else: logger.info("Todos los símbolos se procesaron sin errores fatales reportados.")
    logger.info("=" * (40 + len(" Resumen del Proceso ")))

if __name__ == "__main__":
    try: 
        main()
        logger.info("<<<<<<<<<< Proceso completado >>>>>>>>>>")
    except Exception as e_main: 
        logger.critical(f"!!!!!!!!!! Error FataL en ejecución principal: {e_main} !!!!!!!!!!", exc_info=True)

