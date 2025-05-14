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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
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


# --- Funciones de Filtrado Adicionales ---
def apply_min_start_date_filter(df: pd.DataFrame, date_col: str, min_start_date_str: str | None, symbol_context: str) -> pd.DataFrame:
    if min_start_date_str is None or df.empty or date_col not in df.columns:
        if min_start_date_str is not None and not df.empty : 
             logger.debug(f"[{symbol_context}] Omitiendo filtro de fecha de inicio (columna '{date_col}' no encontrada o df vacío).")
        return df

    try:
        min_start_date = pd.to_datetime(min_start_date_str, errors='coerce')
        if pd.isna(min_start_date):
            logger.warning(f"[{symbol_context}] Fecha de inicio mínima inválida: {min_start_date_str}. Omitiendo filtro.")
            return df
        
        df_copy = df.copy() 
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            logger.debug(f"[{symbol_context}] Columna '{date_col}' no es datetime, intentando convertir para filtro de fecha.")
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy.dropna(subset=[date_col], inplace=True)

        if df_copy.empty: 
            logger.info(f"[{symbol_context}] DataFrame vacío después de conversión de '{date_col}' para filtro de fecha.")
            return df_copy

        original_rows = len(df_copy)
        
        min_start_date_naive = min_start_date.tz_localize(None) if min_start_date.tz is not None else min_start_date
        
        current_col_to_compare = df_copy[date_col]
        if getattr(current_col_to_compare.dt, 'tz', None) is not None:
            current_col_to_compare = current_col_to_compare.dt.tz_convert('UTC').dt.tz_localize(None)
            
        df_filtered = df_copy[current_col_to_compare >= min_start_date_naive]
            
        rows_after = len(df_filtered)
        logger.info(f"[{symbol_context}] Filtro de fecha de inicio ({min_start_date_str}) sobre '{date_col}': {original_rows} -> {rows_after} filas.")
        return df_filtered 
    except Exception as e:
        logger.error(f"[{symbol_context}] Error aplicando filtro de fecha de inicio ({min_start_date_str}) sobre '{date_col}': {e}", exc_info=True)
        return df 

def apply_min_initial_price_filter(df: pd.DataFrame, price_ref_series: pd.Series, min_initial_price: float | None, symbol_context: str, time_col_for_logging: str) -> pd.DataFrame:
    if min_initial_price is None or df.empty or price_ref_series.empty:
        if min_initial_price is not None and not df.empty:
            logger.debug(f"[{symbol_context}] Omitiendo filtro de precio inicial mínimo (price_ref_series vacía).")
        return df

    try:
        df_copy = df.copy() 
        price_ref_series_numeric = pd.to_numeric(price_ref_series, errors='coerce').dropna()
        if price_ref_series_numeric.empty:
            logger.warning(f"[{symbol_context}] Serie de referencia de precios vacía después de to_numeric/dropna. Omitiendo filtro de precio.")
            return df_copy

        if not df_copy.index.equals(price_ref_series_numeric.index):
            logger.debug(f"[{symbol_context}] Índices no coinciden para filtro de precio. Reindexando price_ref_series al índice de df_copy.")
            price_ref_series_numeric = price_ref_series_numeric.reindex(df_copy.index) 
        
        condition_met_series = (price_ref_series_numeric >= min_initial_price)
        if not condition_met_series.any(): 
            logger.info(f"[{symbol_context}] Precio inicial mínimo ({min_initial_price}) nunca alcanzado en la serie de referencia. Eliminando todas las {len(df_copy)} filas.")
            return df_copy.iloc[0:0]

        first_valid_index_label = condition_met_series.idxmax()
        
        original_rows = len(df_copy)
        df_filtered = df_copy.loc[first_valid_index_label:] 
        rows_after = len(df_filtered)
        
        first_retained_timestamp_str = "N/A"
        if not df_filtered.empty and time_col_for_logging in df_filtered.columns:
            first_retained_timestamp_val = df_filtered[time_col_for_logging].iloc[0]
            if pd.notna(first_retained_timestamp_val):
                 first_retained_timestamp_str = pd.to_datetime(first_retained_timestamp_val).isoformat()

        logger.info(f"[{symbol_context}] Filtro de precio inicial mínimo ({min_initial_price}): {original_rows} -> {rows_after} filas. Primer dato retenido en {first_retained_timestamp_str} (col: {time_col_for_logging}).")
        return df_filtered
    except Exception as e:
        logger.error(f"[{symbol_context}] Error aplicando filtro de precio inicial mínimo ({min_initial_price}): {e}", exc_info=True)
        return df 

# --- Funciones del Pipeline ---

def robust_to_datetime(series: pd.Series) -> pd.Series:
    series_name = series.name if series.name else "UnnamedSeries"
    if not isinstance(series, pd.Series):
        logger.warning(f"Input to robust_to_datetime for '{series_name}' is not a Series. Type: {type(series)}")
        return pd.Series([], dtype='datetime64[ns]', name=series_name)

    if series.empty:
        return pd.Series([], dtype='datetime64[ns]', name=series_name)

    s_work = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s_work.dtype):
        if getattr(s_work.dt, 'tz', None) is not None:
            logger.debug(f"'{series_name}': Original series is tz-aware ({s_work.dt.tz}), converting to naive UTC.")
            return s_work.dt.tz_convert('UTC').tz_localize(None)
        logger.debug(f"'{series_name}': Original series is already naive datetime.")
        return s_work

    converted_series = pd.to_datetime(s_work, errors='coerce', utc=True)

    if pd.api.types.is_datetime64_any_dtype(converted_series.dtype) and \
       getattr(converted_series.dt, 'tz', None) is not None:
        converted_series = converted_series.dt.tz_localize(None)
    
    na_mask = converted_series.isna()
    if na_mask.any() and not na_mask.all():
        logger.debug(f"'{series_name}': {na_mask.sum()} NaNs after initial to_datetime. Attempting fallbacks.")
        
        original_failed_indices = s_work[na_mask].index
        
        # Fallback 1: Epoch
        numeric_to_try_on_original = pd.to_numeric(s_work.loc[original_failed_indices], errors='coerce')
        
        if numeric_to_try_on_original.notna().any():
            for unit in ['s', 'ms']:
                # Consider only those that are still NaT in converted_series AND were numeric in original
                relevant_indices_for_epoch = numeric_to_try_on_original[converted_series.loc[numeric_to_try_on_original.index].isna()].dropna().index
                if relevant_indices_for_epoch.empty: continue

                temp_epoch_converted = pd.to_datetime(numeric_to_try_on_original.loc[relevant_indices_for_epoch], unit=unit, errors='coerce', utc=True)
                if pd.api.types.is_datetime64_any_dtype(temp_epoch_converted.dtype) and \
                   getattr(temp_epoch_converted.dt, 'tz', None) is not None:
                    temp_epoch_converted = temp_epoch_converted.dt.tz_localize(None)
                
                # Usar combine_first para rellenar NaNs en converted_series con valores de temp_epoch_converted
                # Esto crea una nueva serie, evitando SettingWithCopyWarning si converted_series fuera una vista.
                converted_series = converted_series.combine_first(temp_epoch_converted)
                logger.debug(f"'{series_name}': Fallback epoch '{unit}' (combine_first) intentó actualizar {len(relevant_indices_for_epoch)} valores, {temp_epoch_converted.notna().sum()} convertidos exitosamente.")
                if not converted_series.isna().any(): break 

        # Fallback 2: String replace 'T'
        na_mask_after_epoch = converted_series.isna() 
        if na_mask_after_epoch.any() and not na_mask_after_epoch.all():
            original_failed_after_epoch_indices = s_work[na_mask_after_epoch].index
            strings_to_try = s_work.loc[original_failed_after_epoch_indices].astype(str).str.replace(' ', 'T', regex=False)
            
            temp_str_converted = pd.to_datetime(strings_to_try, errors='coerce', utc=True)
            if pd.api.types.is_datetime64_any_dtype(temp_str_converted.dtype) and \
               getattr(temp_str_converted.dt, 'tz', None) is not None:
                temp_str_converted = temp_str_converted.dt.tz_localize(None)

            converted_series = converted_series.combine_first(temp_str_converted)
            logger.debug(f"'{series_name}': Fallback string replace 'T' (combine_first) intentó actualizar {na_mask_after_epoch.sum()} valores, {temp_str_converted.notna().sum()} convertidos exitosamente.")

    final_nans = converted_series.isna().sum()
    if final_nans > 0 and final_nans < len(converted_series): 
        logger.warning(f"'{series_name}': {final_nans} valores permanecen NaT después de todos los intentos de conversión.")
    elif final_nans == len(converted_series) and not series.empty : 
        logger.error(f"'{series_name}': Todos los {len(series)} valores fallaron la conversión a datetime y son NaT.")
        return pd.Series([pd.NaT] * len(s_work), index=s_work.index, name=series_name, dtype='datetime64[ns]')


    if not pd.api.types.is_datetime64_any_dtype(converted_series.dtype) and converted_series.notna().any():
        try:
            final_attempt_series = pd.to_datetime(converted_series, errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(final_attempt_series.dtype):
                return final_attempt_series
            else:
                logger.error(f"'{series_name}': No se pudo asegurar dtype datetime64[ns] al final. Dtype es {converted_series.dtype}")
        except Exception as e_final_coerce:
            logger.error(f"'{series_name}': Error en el último intento de coerción a datetime: {e_final_coerce}")
    elif converted_series.isna().all() and not series.empty:
         return pd.Series([pd.NaT] * len(s_work), index=s_work.index, name=series_name, dtype='datetime64[ns]')

    return converted_series


def load_data(db_path: str, symbol: str, min_start_date: str | None, min_initial_price: float | None) -> pd.DataFrame:
    logger.info(f"[{symbol}] Iniciando carga de datos OHLCV...")
    conn = sqlite3.connect(db_path); query = "SELECT * FROM coinapi_ohlcv WHERE symbol = ?"
    try:
        df = pd.read_sql(query, conn, params=[symbol])
        logger.info(f"[{symbol}] Filas OHLCV cargadas inicialmente: {len(df)}")
        if df.empty: logger.warning(f"[{symbol}] No datos OHLCV para {symbol}"); return pd.DataFrame()

        crit_cols = ["time_period_start", "time_period_end"]
        
        nan_mask = pd.Series(False, index=df.index)
        for col in crit_cols:
            if col not in df.columns: logger.error(f"[{symbol}] Falta col crítica '{col}'."); return pd.DataFrame()
            nan_mask |= df[col].isnull() | (df[col].astype(str).str.strip() == '' if pd.api.types.is_string_dtype(df[col]) else False)
        
        if (num_nulls := nan_mask.sum()) > 0:
            df = df[~nan_mask]; logger.info(f"[{symbol}] Eliminados {num_nulls} OHLCV con NULL/vacío en {crit_cols}. Filas restantes: {len(df)}")
            if df.empty: return pd.DataFrame()

        time_cols = ["time_period_start", "time_period_end", "time_open", "time_close"]
        for col in time_cols:
            if col in df.columns:
                logger.debug(f"[{symbol}] Convirtiendo columna de tiempo: {col}")
                df[col] = robust_to_datetime(df[col])
            elif col in crit_cols: logger.error(f"[{symbol}] Col crítica '{col}' desapareció."); return pd.DataFrame()
            else: df[col] = pd.NaT 
        
        failed_conv_mask = df[crit_cols].isna().any(axis=1)
        if (failed_conv_sum := failed_conv_mask.sum()) > 0:
            logger.warning(f"[{symbol}] {failed_conv_sum} filas OHLCV con NaT en {crit_cols} post-conversión robusta. Eliminándolas.")
            df = df.dropna(subset=crit_cols) 
            logger.info(f"[{symbol}] Filas restantes después de eliminar NaT en fechas críticas: {len(df)}")
            if df.empty: return pd.DataFrame()
        
        df = df.sort_values(by="time_period_start").reset_index(drop=True)

        df = apply_min_start_date_filter(df, "time_period_start", min_start_date, symbol)
        if df.empty: logger.info(f"[{symbol}] DataFrame vacío después de filtro de fecha."); return df
        
        price_cols_filter = ["price_open", "price_high", "price_low", "price_close"]
        existing_pc_filter = [c for c in price_cols_filter if c in df.columns]
        if existing_pc_filter:
            for col in existing_pc_filter: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if "price_close" in df.columns:
            df = apply_min_initial_price_filter(df.copy(), df["price_close"].copy(), min_initial_price, symbol, "time_period_start") 
        else:
            logger.warning(f"[{symbol}] Columna 'price_close' no encontrada. Omitiendo filtro de precio inicial mínimo.")
        if df.empty: logger.info(f"[{symbol}] DataFrame vacío después de filtro de precio."); return df

        if 'time_open' in df.columns: df['time_open'] = df['time_open'].fillna(df.get('time_period_start'))
        if 'time_close' in df.columns: df['time_close'] = df['time_close'].fillna(df.get('time_period_end'))
        
        price_cols = ["price_open", "price_high", "price_low", "price_close"]
        existing_pc = [c for c in price_cols if c in df.columns]
        if existing_pc:
            nan_in_prices_before_drop = df[existing_pc].isna().sum()
            if nan_in_prices_before_drop.any():
                logger.debug(f"[{symbol}] NaNs en columnas de precio antes de dropna: {nan_in_prices_before_drop[nan_in_prices_before_drop > 0].to_dict()}")

            rows_before_dropna_prices = len(df)
            df = df.dropna(subset=existing_pc)
            dropped_ohlc = rows_before_dropna_prices - len(df)
            if dropped_ohlc > 0:
                logger.info(f"[{symbol}] Eliminadas {dropped_ohlc} filas OHLCV sin precios válidos (NaNs). Filas restantes: {len(df)}")
        
        logger.info(f"[{symbol}] Filas OHLCV válidas post-carga y filtros iniciales: {len(df)}")
    except Exception as e: 
        logger.error(f"[{symbol}] Error carga OHLCV: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn: conn.close()
    return df.copy() 


def load_orderbook(db_path: str, symbol: str, min_start_date: str | None, min_initial_price: float | None) -> pd.DataFrame:
    logger.info(f"[{symbol}] Iniciando carga de datos Orderbook...")
    conn = sqlite3.connect(db_path); query = "SELECT * FROM coinapi_orderbook WHERE symbol_id = ?"
    try:
        df = pd.read_sql(query, conn, params=[symbol])
        logger.info(f"[{symbol}] Filas Orderbook cargadas inicialmente: {len(df)}")
        if df.empty: logger.warning(f"[{symbol}] No datos Orderbook para {symbol}"); return pd.DataFrame()

        nan_mask = df['ts'].isnull() | (df['ts'].astype(str).str.strip() == '' if pd.api.types.is_string_dtype(df['ts']) else False)
        if (num_nulls := nan_mask.sum()) > 0: 
            df = df[~nan_mask]; logger.info(f"[{symbol}] Eliminados {num_nulls} OB con 'ts' NULL/vacío. Filas restantes: {len(df)}")
        if df.empty: return pd.DataFrame()
        
        logger.debug(f"[{symbol}] Convirtiendo columna de tiempo: ts (Orderbook)")
        df['ts'] = robust_to_datetime(df['ts'])
        if (failed_conv := df['ts'].isna().sum()) > 0: 
            df = df.dropna(subset=['ts']); logger.warning(f"[{symbol}] Eliminados {failed_conv} OB con 'ts' NaT. Filas restantes: {len(df)}")
        if df.empty: return pd.DataFrame()

        df = df.sort_values(by="ts").reset_index(drop=True)
        
        df = apply_min_start_date_filter(df, "ts", min_start_date, symbol)
        if df.empty: logger.info(f"[{symbol}] DataFrame OB vacío después de filtro de fecha."); return df
        
        ob_price_cols_filter = ["bid1_px", "ask1_px"]
        can_filter_price = True
        df_for_price_filter = df.copy() 
        for col in ob_price_cols_filter:
            if col in df_for_price_filter.columns:
                df_for_price_filter[col] = pd.to_numeric(df_for_price_filter[col], errors='coerce')
            else: 
                logger.warning(f"[{symbol}] Columna {col} no encontrada para calcular mid_price para filtro OB. Omitiendo filtro de precio inicial.")
                can_filter_price = False
                break 
        
        if can_filter_price and min_initial_price is not None:
            df_temp_mid = df_for_price_filter.dropna(subset=["bid1_px", "ask1_px"])
            if not df_temp_mid.empty:
                mid_price_temp = (df_temp_mid["bid1_px"] + df_temp_mid["ask1_px"]) / 2
                mid_price_temp.name = "mid_price_temp_filter"
                df_temp_mid_filtered = apply_min_initial_price_filter(df_temp_mid, mid_price_temp, min_initial_price, symbol, "ts")
                df = df.loc[df_temp_mid_filtered.index] 
            else:
                logger.warning(f"[{symbol}] No hay datos válidos de bid1_px/ask1_px para calcular mid_price para filtro OB.")

        if df.empty: logger.info(f"[{symbol}] DataFrame OB vacío después de filtro de precio."); return df

        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce') 
        
        req_cols = [f"{s}{l}_{t}" for s in ["bid","ask"] for l in [1,2,3] for t in ["px","sz"]]
        missing_req = [c for c in req_cols if c not in df.columns]
        if missing_req: 
            logger.error(f"[{symbol}] Faltan columnas OB requeridas: {missing_req}. No se puede continuar con este símbolo para OB.")
            return pd.DataFrame()
            
        for col in req_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        rows_before_dropna_pxsz = len(df)
        df = df.dropna(subset=req_cols) 
        dropped_pxsz = rows_before_dropna_pxsz - len(df)
        if dropped_pxsz > 0:
            logger.warning(f"[{symbol}] Eliminados {dropped_pxsz} OB con NaNs en px/sz. Filas restantes: {len(df)}")
        if df.empty: return pd.DataFrame()
            
        logger.info(f"[{symbol}] Carga OB completada. Filas válidas post-filtros iniciales: {len(df)}")
    except Exception as e: 
        logger.error(f"[{symbol}] Error carga OB: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn: conn.close()
    return df.reset_index(drop=True).copy()


def validate_schema(df: pd.DataFrame, schema: DataFrameSchema, context_symbol: str) -> pd.DataFrame:
    s_name = schema.name if hasattr(schema, 'name') else 'Schema'
    logger.info(f"[{context_symbol}] Validando {len(df)} filas contra {s_name}...")
    if df.empty:
        logger.info(f"[{context_symbol}] DataFrame vacío, omitiendo validación de esquema para {s_name}.")
        return df 
    try: 
        df_val = df.copy() 
        return schema.validate(df_val, lazy=True)
    except pa.errors.SchemaErrors as err:
        logger.error(f"[{context_symbol}] Errores validación {s_name}:\n{err.failure_cases.head(3)}")
        if err.data is not None: logger.error(f"[{context_symbol}] DF que falló ({len(err.data)} filas) head:\n{err.data.head(3)}")
        raise
    except Exception as e: 
        logger.error(f"[{context_symbol}] Error inesperado validación {s_name}: {e}", exc_info=True)
        raise

def reindex_time(df: pd.DataFrame, freq_candle: str, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Iniciando reindexado de OHLCV a frecuencia {freq_candle}.")
    if df.empty or 'time_period_start' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['time_period_start']):
        logger.warning(f"[{symbol_context}] DataFrame vacío o 'time_period_start' inválido para reindexar.")
        return df.copy()
    
    df_copy = df.copy() 
    df_copy = df_copy.dropna(subset=['time_period_start']) 
    if df_copy.empty:
        logger.warning(f"[{symbol_context}] DataFrame vacío después de dropna en 'time_period_start' para reindexar.")
        return df_copy

    if not df_copy['time_period_start'].is_unique:
        logger.warning(f"[{symbol_context}] Timestamps duplicados en 'time_period_start' antes de reindexar. Eliminando duplicados, manteniendo el último.")
        df_copy = df_copy.sort_values('time_period_start').drop_duplicates(subset=['time_period_start'], keep='last')
    
    df_copy = df_copy.set_index("time_period_start").sort_index()
    min_ts, max_ts = df_copy.index.min(), df_copy.index.max()
    logger.info(f"[{symbol_context}] Rango de tiempo para reindexar: {min_ts} a {max_ts}")

    if pd.isna(min_ts) or pd.isna(max_ts) or min_ts > max_ts:
        logger.warning(f"[{symbol_context}] Rango de tiempo inválido ({min_ts} a {max_ts}). Devolviendo DataFrame original post-set_index.")
        return df_copy.reset_index() 
    
    try:
        full_index = pd.date_range(start=min_ts, end=max_ts, freq=freq_candle, name="time_period_start")
        if full_index.empty and min_ts == max_ts: 
            full_index = pd.DatetimeIndex([min_ts], name="time_period_start")
        elif full_index.empty: 
            logger.warning(f"[{symbol_context}] pd.date_range devolvió un índice vacío para {min_ts}-{max_ts} con freq {freq_candle}. Usando índice original.")
            full_index = df_copy.index
    except ValueError as ve:
        logger.error(f"[{symbol_context}] ValueError al crear date_range para {min_ts}-{max_ts} con freq {freq_candle}: {ve}. Devolviendo original.")
        return df_copy.reset_index()

    out = df_copy.reindex(full_index) 
    
    if "symbol" in out.columns and "symbol" in df_copy.columns and not df_copy.empty:
         out["symbol"] = out["symbol"].fillna(df_copy["symbol"].iloc[0])
    elif "symbol" not in out.columns and "symbol" in df_copy.columns and not df_copy.empty:
        out["symbol"] = df_copy["symbol"].iloc[0]
    elif "symbol" not in out.columns:
        out["symbol"] = symbol_context 

    # Rellenar time_open y time_close para las nuevas filas del índice
    time_open_fill_values = pd.Series(out.index, index=out.index) # El índice de 'out' es time_period_start
    if 'time_open' not in out.columns: 
        out['time_open'] = pd.NaT 
    out['time_open'] = out['time_open'].combine_first(time_open_fill_values)


    out["time_period_end"] = out.index + pd.Timedelta(freq_candle) 
    time_close_fill_values = pd.Series(out['time_period_end'].values, index=out.index) 
    if 'time_close' not in out.columns:
        out['time_close'] = pd.NaT
    out['time_close'] = out['time_close'].combine_first(time_close_fill_values)


    out["volume_traded"] = out.get("volume_traded", pd.Series(dtype=float, index=out.index)).fillna(0)
    out["trades_count"] = pd.to_numeric(out.get("trades_count"), errors='coerce').fillna(0).astype(int)
    
    for flag_col_name in ["flag_bad_structure", "flag_outlier_mad", "flag_jump"]:
        if flag_col_name in out.columns: 
            out[flag_col_name] = out[flag_col_name].fillna(False).astype(bool)
        else: 
            out[flag_col_name] = False 
            
    out = out.reset_index() 
    logger.info(f"[{symbol_context}] Reindexado OHLCV: {len(df_copy)} -> {len(out)} filas.")
    return out

def detect_structure(df: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Detectando estructura OHLCV...")
    df_c = df.copy()
    if "flag_bad_structure" not in df_c.columns: df_c["flag_bad_structure"] = False
    else: df_c["flag_bad_structure"] = False 

    if df_c.empty: return df_c
    price_cols = ["price_open", "price_high", "price_low", "price_close"]; vol_cols = ["volume_traded", "trades_count"]
    
    missing_cols = [col for col in price_cols if col not in df_c.columns]
    if missing_cols:
        logger.warning(f"[{symbol_context}] Faltan columnas de precio para detección de estructura: {missing_cols}. Marcando todas las filas como mala estructura.")
        df_c["flag_bad_structure"] = True
        return df_c

    for col in price_cols + vol_cols: 
        if col not in df_c.columns: df_c[col] = np.nan
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce') 

    conds = (
        df_c["price_high"].ge(df_c["price_open"]) &
        df_c["price_high"].ge(df_c["price_low"]) &
        df_c["price_high"].ge(df_c["price_close"]) &
        df_c["price_low"].le(df_c["price_open"]) &
        df_c["price_low"].le(df_c["price_high"]) & 
        df_c["price_low"].le(df_c["price_close"]) &
        df_c["volume_traded"].ge(0) &
        df_c["trades_count"].ge(0) &
        df_c[price_cols].notna().all(axis=1) 
    )
    df_c["flag_bad_structure"] = ~conds
    if (num_bad := df_c["flag_bad_structure"].sum()) > 0: 
        logger.info(f"[{symbol_context}] {num_bad} filas OHLCV con estructura inválida detectada.")
    else:
        logger.info(f"[{symbol_context}] No se encontraron filas con mala estructura.")
    return df_c

def detect_outliers(df: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Detectando outliers y saltos en OHLCV...")
    df_c = df.copy()
    for flag_col in ["flag_outlier_mad", "flag_jump"]:
        if flag_col not in df_c.columns: df_c[flag_col] = False
        else: df_c[flag_col] = False 

    if df_c.empty: return df_c
    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    existing_pc = [c for c in price_cols if c in df_c.columns]
    if not existing_pc: 
        logger.warning(f"[{symbol_context}] No hay columnas de precio para detección de outliers.")
        return df_c
        
    for col in existing_pc: df_c[col] = pd.to_numeric(df_c[col], errors='coerce')

    def flag_mad_daily(sub_df):
        sub_df_copy = sub_df.copy() 
        if sub_df_copy.empty or not existing_pc or sub_df_copy[existing_pc].isna().all().all(): 
            if "flag_outlier_mad" not in sub_df_copy.columns: sub_df_copy["flag_outlier_mad"] = False
            return sub_df_copy
        
        current_mask = pd.Series(False, index=sub_df_copy.index) 
        for col_price in existing_pc:
            s = sub_df_copy[col_price].dropna()
            if len(s) < 5: continue 
            
            median_s = s.median()
            mad_val = median_abs_deviation(s, scale='normal', nan_policy='omit')
            if pd.isna(mad_val) or pd.isna(median_s): continue
            
            mad_thresholded = mad_val if mad_val > 1e-9 else 1e-9 
            lower_bound = median_s - 5 * mad_thresholded
            upper_bound = median_s + 5 * mad_thresholded
            
            current_mask |= (~sub_df_copy[col_price].between(lower_bound, upper_bound)) | sub_df_copy[col_price].isna()
        
        sub_df_copy["flag_outlier_mad"] = current_mask
        return sub_df_copy

    if 'time_period_start' in df_c.columns and pd.api.types.is_datetime64_any_dtype(df_c['time_period_start']):
        df_c_no_nat_time = df_c.dropna(subset=['time_period_start'])
        if not df_c_no_nat_time.empty:
            grouped = df_c_no_nat_time.groupby(df_c_no_nat_time['time_period_start'].dt.date, group_keys=False)
            
            processed_groups = []
            for _, group in grouped:
                processed_groups.append(flag_mad_daily(group))
            
            if processed_groups:
                df_c = pd.concat(processed_groups).sort_values('time_period_start').reset_index(drop=True)
            else: 
                logger.warning(f"[{symbol_context}] No groups processed for MAD calculation.")
                if "flag_outlier_mad" not in df_c.columns: df_c["flag_outlier_mad"] = False
        else:
            logger.warning(f"[{symbol_context}] DataFrame vacío después de dropna en 'time_period_start' para MAD.")
            if "flag_outlier_mad" not in df_c.columns: df_c["flag_outlier_mad"] = False 
    else:
        logger.warning(f"[{symbol_context}] 'time_period_start' no es datetime, no se puede agrupar por día para MAD.")
        if "flag_outlier_mad" not in df_c.columns: df_c["flag_outlier_mad"] = False

    if 'price_open' in df_c.columns and df_c['price_open'].notna().any() and not df_c.empty: 
        df_c = df_c.sort_values('time_period_start') 
        delta = df_c['price_open'].diff().abs()
        sigma = delta.rolling(window=30, min_periods=15).std()
        sigma = sigma.replace(0, np.nan).ffill().bfill() 
        
        df_c["flag_jump"] = ((delta > (10 * sigma)) & sigma.notna()).fillna(False)
    elif "flag_jump" not in df_c.columns: 
        df_c["flag_jump"] = False
    
    sum_mad = df_c['flag_outlier_mad'].sum() if 'flag_outlier_mad' in df_c.columns and pd.api.types.is_numeric_dtype(df_c['flag_outlier_mad']) else 0
    sum_jump = df_c['flag_jump'].sum() if 'flag_jump' in df_c.columns and pd.api.types.is_numeric_dtype(df_c['flag_jump']) else 0
    logger.info(f"[{symbol_context}] Outliers OHLCV MAD: {sum_mad}, Jumps: {sum_jump}")
    return df_c

def impute_data(df: pd.DataFrame, max_gap_candles: int, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Iniciando imputación de datos OHLCV (max_gap_candles={max_gap_candles}).")
    if df.empty: return df.copy()
    
    df_c = df.copy()
    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    existing_pc = [c for c in price_cols if c in df_c.columns]

    if "flag_outlier_mad" in df_c.columns and existing_pc and pd.api.types.is_bool_dtype(df_c["flag_outlier_mad"]):
        outlier_mask = df_c["flag_outlier_mad"]
        if (num_outliers_to_nan := outlier_mask.sum()) > 0:
            logger.info(f"[{symbol_context}] Convirtiendo {num_outliers_to_nan} outliers (flag_outlier_mad=True) a NaN antes de la interpolación.")
            for col in existing_pc:
                 df_c.loc[outlier_mask, col] = np.nan
    
    if existing_pc:
        nans_before_interp = df_c[existing_pc].isna().sum()
        logger.debug(f"[{symbol_context}] NaNs en precios ANTES de interpolación: {nans_before_interp[nans_before_interp > 0].to_dict()}")

    if 'time_period_start' not in df_c.columns or not pd.api.types.is_datetime64_any_dtype(df_c['time_period_start']):
        logger.warning(f"[{symbol_context}] 'time_period_start' no es datetime o no existe. Usando ffill/bfill para imputación.")
        if existing_pc: df_c[existing_pc] = df_c[existing_pc].ffill().bfill()
    else:
        if not df_c['time_period_start'].is_unique:
            logger.warning(f"[{symbol_context}] Timestamps duplicados en 'time_period_start' antes de imputar. Eliminando duplicados, manteniendo el último.")
            df_c = df_c.sort_values('time_period_start').drop_duplicates(subset=['time_period_start'], keep='last')
        
        df_c = df_c.set_index("time_period_start").sort_index()
        if existing_pc: 
            if df_c.index.is_monotonic_increasing and df_c[existing_pc].notna().any().any():
                 df_c[existing_pc] = df_c[existing_pc].interpolate(method="time", limit=max_gap_candles, limit_area="inside")
            else:
                logger.warning(f"[{symbol_context}] No se pudo interpolar por tiempo (índice no monotónico o sin datos no-NaN).")
        df_c = df_c.reset_index()

    if "volume_traded" in df_c.columns: df_c["volume_traded"] = df_c["volume_traded"].fillna(0)
    if "trades_count" in df_c.columns: df_c["trades_count"] = pd.to_numeric(df_c["trades_count"], errors='coerce').fillna(0).astype(int)
    
    if existing_pc: 
        df_c[existing_pc] = df_c[existing_pc].ffill().bfill() 
        nans_after_impute = df_c[existing_pc].isna().sum()
        logger.debug(f"[{symbol_context}] NaNs en precios DESPUÉS de imputación completa: {nans_after_impute[nans_after_impute > 0].to_dict()}")

    if all(c in df_c.columns for c in price_cols) and existing_pc:
        if not df_c[price_cols].isna().any().any():
            df_c["price_high"] = df_c[price_cols].max(axis=1)
            df_c["price_low"] = df_c[price_cols].min(axis=1)
            logger.debug(f"[{symbol_context}] price_high y price_low reajustados post-imputación.")
        else:
            logger.warning(f"[{symbol_context}] NaNs persistentes en columnas de precio. No se reajustará high/low.")
            
    logger.info(f"[{symbol_context}] Imputación de datos OHLCV completada.")
    return df_c

def reduce_columns(df: pd.DataFrame, final_cols: list, symbol_context: str) -> pd.DataFrame:
    logger.debug(f"[{symbol_context}] Reduciendo columnas a: {final_cols}")
    existing = [c for c in final_cols if c in df.columns]
    if missing := [c for c in final_cols if c not in existing]: 
        logger.warning(f"[{symbol_context}] Faltan columnas finales para reducir: {missing}. Se usarán solo las existentes: {existing}")
    if not existing:
        logger.error(f"[{symbol_context}] No hay ninguna de las columnas finales esperadas en el DataFrame. Devolviendo DataFrame vacío.")
        return pd.DataFrame(columns=final_cols)
    return df[existing].copy()

def final_data_check(df: pd.DataFrame, source: str = "Data", symbol_context: str = "") -> None:
    tag = f"[{symbol_context}][{source}][FinalCheck]"
    if not isinstance(df, pd.DataFrame) or df.empty: 
        logger.warning(f"{tag} DataFrame vacío."); return
        
    logger.info(f"{tag} Filas: {len(df)}, Columnas: {list(df.columns)}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{tag} Dtypes:\n{df.dtypes}")
        
    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    if nulls_present.empty: 
        logger.info(f"{tag} No NaNs en DataFrame final.")
    else: 
        logger.warning(f"{tag} ¡ALERTA! NaNs en DataFrame final:\n{nulls_present}")

def ob_data_check(df: pd.DataFrame, symbol_context: str) -> None: 
    final_data_check(df, source="OB", symbol_context=symbol_context)

def persist_data(df: pd.DataFrame, db_path: str, table_name: str, symbol_context: str, data_type: str):
    if df.empty: 
        logger.warning(f"[{symbol_context}][{data_type}] DataFrame vacío. No se persiste en {table_name}.")
        return
        
    logger.info(f"[{symbol_context}][{data_type}] Persistiendo {len(df)} filas en {table_name}...")
    
    actual_symbol_value = symbol_context 
    if data_type == 'OHLCV' and 'symbol' in df.columns and not df['symbol'].empty:
        actual_symbol_value = df['symbol'].iloc[0]
    elif data_type == 'OB' and 'symbol_id' in df.columns and not df['symbol_id'].empty:
        actual_symbol_value = df['symbol_id'].iloc[0]
    
    cols_map = {
        'coinapi_ohlcv_clean': ["symbol","time_period_start","time_period_end","time_open","time_close","price_open","price_high","price_low","price_close","volume_traded","trades_count","flag_bad_structure","flag_outlier_mad","flag_jump"],
        'coinapi_orderbook_clean': ["symbol_id","ts","date","bid1_px","bid1_sz","bid2_px","bid2_sz","bid3_px","bid3_sz","ask1_px","ask1_sz","ask2_px","ask2_sz","ask3_px","ask3_sz","flag_ob_bad_structure","flag_mid_mad"]
    }
    if table_name not in cols_map: 
        logger.error(f"[{actual_symbol_value}][{data_type}] Tabla desconocida para persistencia: {table_name}"); return
        
    cols_to_persist = [c for c in cols_map[table_name] if c in df.columns]
    if not cols_to_persist: 
        logger.error(f"[{actual_symbol_value}][{data_type}] No hay columnas válidas para persistir en {table_name}. Columnas disponibles: {list(df.columns)}"); return

    df_persist = df[cols_to_persist].copy()
    
    try:
        for col in df_persist.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns: 
             df_persist[col] = df_persist[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(x) else None)
        for col in df_persist.select_dtypes(include=['bool']).columns: 
            df_persist[col] = df_persist[col].astype(int)
        for col in df_persist.columns:
             if col.endswith(('_px','_sz')) or col.startswith('price_') or col=='volume_traded': 
                 df_persist[col] = pd.to_numeric(df_persist[col], errors='coerce') 
             elif col=='trades_count' or col.startswith('flag_'): 
                 df_persist[col] = pd.to_numeric(df_persist[col],errors='coerce').fillna(0).astype(int) 
    except Exception as e: 
        logger.error(f"[{actual_symbol_value}][{data_type}] Error convirtiendo tipos para SQL: {e}", exc_info=True)
        raise RuntimeError(f"Fallo conversión tipos para {actual_symbol_value}") from e
        
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        tuples = [tuple(None if pd.isna(x) else x for x in r) for r in df_persist.itertuples(index=False, name=None)]
        
        if not tuples: 
            logger.warning(f"[{actual_symbol_value}][{data_type}] No hay tuplas para insertar en {table_name}."); return
            
        placeholders = ','.join('?' * len(cols_to_persist))
        sql = f"INSERT OR REPLACE INTO {table_name} ({','.join(cols_to_persist)}) VALUES ({placeholders})"
        
        conn.executemany(sql, tuples)
        conn.commit()
        logger.info(f"[{actual_symbol_value}][{data_type}] {table_name}: {len(tuples)} filas insertadas/reemplazadas.")
    except sqlite3.Error as e_sql: 
        logger.error(f"[{actual_symbol_value}][{data_type}] Error SQLite al persistir en {table_name}: {e_sql}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[{actual_symbol_value}][{data_type}] Error general al persistir en {table_name}: {e}", exc_info=True)
        raise
    finally:
        if conn: conn.close()

def setup_clean_tables(db_path):
     logger.info(f"Configurando tablas _clean en {db_path}...")
     conn = None
     try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_ohlcv_clean (
            symbol TEXT NOT NULL, time_period_start TEXT NOT NULL, time_period_end TEXT,
            time_open TEXT, time_close TEXT, price_open REAL, price_high REAL, price_low REAL,
            price_close REAL, volume_traded REAL, trades_count INTEGER, 
            flag_bad_structure INTEGER, flag_outlier_mad INTEGER, flag_jump INTEGER, 
            PRIMARY KEY (symbol, time_period_start)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS coinapi_orderbook_clean (
            symbol_id TEXT NOT NULL, ts TEXT NOT NULL, date TEXT, 
            bid1_px REAL, bid1_sz REAL, bid2_px REAL, bid2_sz REAL, bid3_px REAL, bid3_sz REAL, 
            ask1_px REAL, ask1_sz REAL, ask2_px REAL, ask2_sz REAL, ask3_px REAL, ask3_sz REAL, 
            flag_ob_bad_structure INTEGER, flag_mid_mad INTEGER, 
            PRIMARY KEY (symbol_id, ts)
        );""")
        logger.info("Tablas _clean verificadas/creadas.")

        logger.info("Eliminando datos existentes de 'coinapi_ohlcv_clean' y 'coinapi_orderbook_clean'...")
        cur.execute("DELETE FROM coinapi_ohlcv_clean;")
        rows_deleted_ohlcv = cur.rowcount
        cur.execute("DELETE FROM coinapi_orderbook_clean;")
        rows_deleted_ob = cur.rowcount
        conn.commit()
        logger.info(f"Se eliminaron {rows_deleted_ohlcv} filas de 'coinapi_ohlcv_clean'.")
        logger.info(f"Se eliminaron {rows_deleted_ob} filas de 'coinapi_orderbook_clean'.")
        logger.info("Tablas _clean limpiadas y listas para nuevos datos.")
     except Exception as e: 
        logger.error(f"Error al configurar/limpiar tablas _clean: {e}", exc_info=True)
        if conn: conn.rollback() 
     finally:
        if conn: conn.close()

def aggregate_ob_to_candle(df_ob: pd.DataFrame, freq: str, symbol_context: str) -> pd.DataFrame | None:
    logger.info(f"[{symbol_context}] Agregando Orderbook a velas de {freq}...")
    if "bid1_px" not in df_ob.columns or "ask1_px" not in df_ob.columns:
        logger.error(f"[{symbol_context}] Faltan bid1_px o ask1_px para agregar OB a velas.")
        return None
    
    df_ob_calc = df_ob.copy() 
    if 'mid_px' not in df_ob_calc.columns:
        if pd.api.types.is_numeric_dtype(df_ob_calc['bid1_px']) and pd.api.types.is_numeric_dtype(df_ob_calc['ask1_px']):
            df_ob_calc['mid_px'] = (df_ob_calc['bid1_px'] + df_ob_calc['ask1_px']) / 2
        else:
            logger.error(f"[{symbol_context}] bid1_px o ask1_px no son numéricos, no se puede calcular mid_px para agregar OB.")
            return None

    sz_cols_needed = [f"{s}{l}_sz" for s in ["bid","ask"] for l in [1,2,3]]
    req_cols_for_agg = ['ts', 'mid_px'] + sz_cols_needed

    missing_cols_for_agg = [c for c in req_cols_for_agg if c not in df_ob_calc.columns]
    if missing_cols_for_agg:
        logger.warning(f"[{symbol_context}] Faltan columnas para agregar OB a velas: {missing_cols_for_agg}. No se puede agregar.")
        return None

    if df_ob_calc.empty or not pd.api.types.is_datetime64_any_dtype(df_ob_calc['ts']):
        logger.warning(f"[{symbol_context}] DataFrame OB vacío o 'ts' inválido para agregar a velas.")
        return None
        
    df = df_ob_calc.copy()
    if not df['ts'].is_unique: df = df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
    df = df.set_index('ts').sort_index()
    if df.empty: return None
    
    ohlc = df['mid_px'].resample(freq).ohlc()
    df['bid_qty'] = df[[c for c in ['bid1_sz', 'bid2_sz', 'bid3_sz'] if c in df.columns]].sum(axis=1)
    df['ask_qty'] = df[[c for c in ['ask1_sz', 'ask2_sz', 'ask3_sz'] if c in df.columns]].sum(axis=1)
    df['total_depth_sz'] = df['bid_qty'] + df['ask_qty']
    df['delta_depth_sz'] = df['total_depth_sz'].diff().abs().fillna(0) 

    agg = ohlc.join(df['delta_depth_sz'].resample(freq).sum().rename("volume_proxy"), how='left')\
              .join(df['mid_px'].resample(freq).count().rename("trades_proxy"), how='left') 
              
    agg[['volume_proxy', 'trades_proxy']] = agg[['volume_proxy', 'trades_proxy']].fillna(0)
    agg['trades_proxy'] = pd.to_numeric(agg['trades_proxy'], errors='coerce').fillna(0).astype(int)
    agg = agg.reset_index().rename(columns={"ts": "time_period_start", "open": "ob_open", "high": "ob_high", "low": "ob_low", "close": "ob_close"})
    
    price_cols_ob_agg = ["ob_open", "ob_high", "ob_low", "ob_close"]
    if (dropped := len(agg) - len(agg.dropna(subset=price_cols_ob_agg, how='all'))) > 0:
         agg = agg.dropna(subset=price_cols_ob_agg, how='all')
         logger.debug(f"[{symbol_context}] Eliminadas {dropped} velas OB agregadas sin datos de precio.")
         
    logger.info(f"[{symbol_context}] Agregación OB: {len(agg)} velas generadas.")
    return agg.copy()

def fill_ohlcv_from_ob(df_ohlcv: pd.DataFrame, df_ob_agg: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Intentando rellenar OHLCV desde OB agregado...")
    if df_ohlcv.empty: 
        logger.info(f"[{symbol_context}] OHLCV original vacío, no se rellena.")
        return df_ohlcv.copy()
    if df_ob_agg is None or df_ob_agg.empty: 
        logger.info(f"[{symbol_context}] OB agregado vacío, no se rellena OHLCV.")
        return df_ohlcv.copy()
        
    df = df_ohlcv.copy()
    for df_check, name in [(df, "OHLCV"), (df_ob_agg, "OB Agg")]:
        if 'time_period_start' not in df_check.columns or not pd.api.types.is_datetime64_any_dtype(df_check['time_period_start']):
            logger.error(f"[{symbol_context}] 'time_period_start' en {name} inválido para rellenar."); return df
            
    if not pd.api.types.is_datetime64_any_dtype(df['time_period_start']):
        df['time_period_start'] = pd.to_datetime(df['time_period_start'], errors='coerce')
    df_ob_agg_copy = df_ob_agg.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ob_agg_copy['time_period_start']):
        df_ob_agg_copy['time_period_start'] = pd.to_datetime(df_ob_agg_copy['time_period_start'], errors='coerce')

    df_ob_agg_dedup = df_ob_agg_copy.dropna(subset=['time_period_start']).drop_duplicates(subset=['time_period_start'], keep='last')
    
    df_merged = pd.merge(df, df_ob_agg_dedup, on="time_period_start", how="left", suffixes=("", "_ob"))
    
    mapping = {"price_open": "ob_open", "price_high": "ob_high", "price_low": "ob_low", "price_close": "ob_close",
               "volume_traded": "volume_proxy", "trades_count": "trades_proxy"}
    
    cols_filled_counts = {target: 0 for target in mapping}

    for target, source in mapping.items():
        if source in df_merged.columns and target in df_merged.columns:
            fill_mask = df_merged[target].isna() & df_merged[source].notna()
            if fill_mask.any():
                df_merged.loc[fill_mask, target] = df_merged.loc[fill_mask, source]
                cols_filled_counts[target] = fill_mask.sum()

    for col, count in cols_filled_counts.items():
        if count > 0: logger.info(f"[{symbol_context}] Rellenados {count} NaNs en '{col}' usando datos de OB agregado.")
        
    cols_to_drop = [s for t,s in mapping.items() if s != t and s in df_merged.columns] 
    if cols_to_drop: df_merged = df_merged.drop(columns=list(set(cols_to_drop)))
    
    logger.info(f"[{symbol_context}] Relleno de OHLCV desde OB completado.")
    return df_merged.copy()

def detect_ob_structure(df: pd.DataFrame, symbol_context: str) -> pd.DataFrame: 
    logger.info(f"[{symbol_context}] Detectando estructura Orderbook...")
    df_c = df.copy()
    if "flag_ob_bad_structure" not in df_c.columns: df_c["flag_ob_bad_structure"] = False
    else: df_c["flag_ob_bad_structure"] = False 

    if df_c.empty: return df_c
    px_cols = [f"{s}{l}_px" for s in ["bid","ask"] for l in [1,2,3]]
    sz_cols = [c for c in df_c.columns if c.endswith("_sz")]
    
    if any(c not in df_c.columns for c in px_cols):
        logger.warning(f"[{symbol_context}] Faltan columnas de precio para detección de estructura OB. Marcando todo como malo.")
        df_c["flag_ob_bad_structure"]=True
        return df_c
        
    for col in px_cols + sz_cols:
        if col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
    
    cond = (
        df_c["bid1_px"].ge(df_c["bid2_px"]) & 
        df_c["bid2_px"].ge(df_c["bid3_px"]) &
        df_c["ask1_px"].le(df_c["ask2_px"]) & 
        df_c["ask2_px"].le(df_c["ask3_px"]) &
        df_c["bid1_px"].lt(df_c["ask1_px"]) 
    )
    if sz_cols: 
        sz_conditions = pd.DataFrame({sz_col: (df_c[sz_col] > 1e-9) for sz_col in sz_cols if sz_col in df_c})
        if not sz_conditions.empty:
            cond &= sz_conditions.all(axis=1)
            
    all_px_sz_cols = [c for c in px_cols + sz_cols if c in df_c.columns]
    if all_px_sz_cols:
        cond &= df_c[all_px_sz_cols].notna().all(axis=1)
        
    df_c["flag_ob_bad_structure"] = ~cond
    if (nbad := df_c["flag_ob_bad_structure"].sum()) > 0: 
        logger.info(f"[{symbol_context}] {nbad} filas OB con estructura inválida detectada.")
    else:
        logger.info(f"[{symbol_context}] No se encontraron filas OB con mala estructura.")
    return df_c

def detect_ob_outliers(df: pd.DataFrame, symbol_context: str) -> pd.DataFrame: 
    logger.info(f"[{symbol_context}] Detectando outliers en mid-price de Orderbook...")
    df_c = df.copy() 
    if "flag_mid_mad" not in df_c.columns: df_c["flag_mid_mad"] = False
    else: df_c["flag_mid_mad"] = False 

    req_cols = ["bid1_px", "ask1_px", "ts"]
    if df_c.empty or not all(c in df_c.columns for c in req_cols) or not pd.api.types.is_datetime64_any_dtype(df_c['ts']):
        logger.warning(f"[{symbol_context}] DataFrame vacío o columnas/tipos inválidos para detectar outliers OB.")
        if "flag_mid_mad" not in df_c.columns: df_c["flag_mid_mad"] = False 
        return df_c 
        
    df_c['mid_px'] = (df_c['bid1_px'] + df_c['ask1_px']) / 2
    
    def flag_day_ob(sub_df):
        sub_df_copy = sub_df.copy()
        if sub_df_copy.empty: 
            if "flag_mid_mad" not in sub_df_copy.columns: sub_df_copy["flag_mid_mad"] = False
            return sub_df_copy
        
        current_mask = pd.Series(False, index=sub_df_copy.index)
        if 'mid_px' not in sub_df_copy.columns:
            logger.error(f"[{symbol_context}] 'mid_px' no encontrado en sub_df para flag_day_ob.")
            sub_df_copy["flag_mid_mad"] = True 
            return sub_df_copy

        s_mid = sub_df_copy["mid_px"].dropna()[lambda x: x > 0] 
        
        if len(s_mid) >= 5:
            median_s_mid = s_mid.median()
            mad_m = median_abs_deviation(s_mid, scale='normal', nan_policy='omit')
            if not pd.isna(mad_m) and not pd.isna(median_s_mid):
                mad_thresholded = mad_m if mad_m > 1e-9 else 1e-9
                lower_bound = median_s_mid - 5 * mad_thresholded
                upper_bound = median_s_mid + 5 * mad_thresholded
                current_mask = (~sub_df_copy["mid_px"].between(lower_bound, upper_bound)) | \
                               sub_df_copy["mid_px"].isna() | \
                               (sub_df_copy["mid_px"] <= 0) 
            else: 
                current_mask = sub_df_copy["mid_px"].isna() | (sub_df_copy["mid_px"] <= 0)
        else: 
            current_mask = sub_df_copy["mid_px"].isna() | (sub_df_copy["mid_px"] <= 0)
            
        sub_df_copy["flag_mid_mad"] = current_mask.fillna(True) 
        return sub_df_copy

    df_c_no_nat_time = df_c.dropna(subset=['ts'])
    if not df_c_no_nat_time.empty:
        grouped = df_c_no_nat_time.groupby(df_c_no_nat_time['ts'].dt.date, group_keys=False)
        processed_groups = [flag_day_ob(group) for _, group in grouped]
        if processed_groups:
            df_c = pd.concat(processed_groups).sort_values('ts').reset_index(drop=True)
        else:
            logger.warning(f"[{symbol_context}] No groups processed for OB MAD calculation.")
            if "flag_mid_mad" not in df_c.columns: df_c["flag_mid_mad"] = False
    else:
        logger.warning(f"[{symbol_context}] DataFrame OB vacío después de dropna en 'ts' para MAD.")
        if "flag_mid_mad" not in df_c.columns: df_c["flag_mid_mad"] = False
        
    sum_flag_mid_mad = df_c['flag_mid_mad'].sum() if 'flag_mid_mad' in df_c.columns else 0
    logger.info(f"[{symbol_context}] Outliers Mid-Price MAD (OB): {sum_flag_mid_mad}")
    return df_c


def impute_ob(df: pd.DataFrame, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Iniciando imputación de datos Orderbook...")
    if df.empty: return df.copy()
    
    df_c = df.copy()
    px_cols = [c for c in df_c.columns if c.endswith("_px") and c != "mid_px"] 
    sz_cols = [c for c in df_c.columns if c.endswith("_sz")]
    
    bad_mask = pd.Series(False, index=df_c.index)
    flags_to_check = ["flag_ob_bad_structure", "flag_mid_mad"]
    for flag in flags_to_check:
        if flag in df_c.columns and pd.api.types.is_bool_dtype(df_c[flag]):
            bad_mask |= df_c[flag]
            logger.debug(f"[{symbol_context}] Filas marcadas por {flag}: {df_c[flag].sum()}")
            
    if sz_cols:
        for sz_col in sz_cols:
            if sz_col in df_c.columns:
                large_sizes = df_c[sz_col] >= ORDERBOOK_SIZE_CAP_THRESHOLD
                if large_sizes.any():
                    num_capped = large_sizes.sum()
                    logger.warning(f"[{symbol_context}] Capeando {num_capped} tamaños en '{sz_col}' >= {ORDERBOOK_SIZE_CAP_THRESHOLD:.0e} a NaN.")
                    df_c.loc[large_sizes, sz_col] = np.nan 
                    bad_mask |= large_sizes 

    cols_to_nanify = [c for c in px_cols + sz_cols if c in df_c.columns]
    if (num_to_nan := bad_mask.sum()) > 0 and cols_to_nanify:
        logger.info(f"[{symbol_context}] Convirtiendo {num_to_nan} filas OB marcadas (mala estructura/outlier/tamaño excesivo) a NaN en columnas px/sz.")
        df_c.loc[bad_mask, cols_to_nanify] = np.nan
    
    if 'ts' not in df_c.columns or not pd.api.types.is_datetime64_any_dtype(df_c['ts']):
        logger.warning(f"[{symbol_context}] 'ts' no es datetime o no existe en OB. Usando ffill/bfill simple si hay columnas.")
        if cols_to_nanify: df_c[cols_to_nanify] = df_c[cols_to_nanify].ffill().bfill()
        if sz_cols: df_c[sz_cols] = df_c[sz_cols].fillna(0) 
    else:
        if not df_c['ts'].is_unique:
            logger.warning(f"[{symbol_context}] Timestamps duplicados en 'ts' OB antes de imputar. Eliminando duplicados, manteniendo el último.")
            df_c = df_c.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
        
        df_c = df_c.set_index("ts").sort_index()
        if px_cols: df_c[px_cols] = df_c[px_cols].ffill().bfill()
        if sz_cols: 
            df_c[sz_cols] = df_c[sz_cols].ffill() 
            df_c[sz_cols] = df_c[sz_cols].fillna(0) 
        df_c = df_c.reset_index()

    if 'mid_px' in df_c.columns: df_c = df_c.drop(columns=['mid_px']) 
    if 'spread' in df_c.columns: df_c = df_c.drop(columns=['spread'])
    
    logger.info(f"[{symbol_context}] Imputación de datos Orderbook completada.")
    return df_c

def reindex_ob(df: pd.DataFrame, freq: str, symbol_context: str) -> pd.DataFrame:
    logger.info(f"[{symbol_context}] Iniciando reindexado de Orderbook a frecuencia {freq}.")
    if df.empty or 'ts' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['ts']):
        logger.warning(f"[{symbol_context}] DataFrame OB vacío o 'ts' inválido para reindexar.")
        return df.copy()

    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['ts'])
    if df_copy.empty:
        logger.warning(f"[{symbol_context}] DataFrame OB vacío después de dropna en 'ts' para reindexar.")
        return df_copy

    if not df_copy['ts'].is_unique:
        logger.warning(f"[{symbol_context}] Timestamps duplicados en 'ts' OB antes de reindexar. Eliminando duplicados, manteniendo el último.")
        df_copy = df_copy.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')
    
    df_copy = df_copy.set_index("ts").sort_index()
    min_ts, max_ts = df_copy.index.min(), df_copy.index.max()
    logger.info(f"[{symbol_context}] Rango de tiempo OB para reindexar: {min_ts} a {max_ts}")

    if pd.isna(min_ts) or pd.isna(max_ts) or min_ts > max_ts:
        logger.warning(f"[{symbol_context}] Rango de tiempo OB inválido ({min_ts} a {max_ts}). Devolviendo original.")
        return df_copy.reset_index()
        
    try:
        full_index = pd.date_range(start=min_ts, end=max_ts, freq=freq, name="ts")
        if full_index.empty and min_ts == max_ts: 
            full_index = pd.DatetimeIndex([min_ts], name="ts")
        elif full_index.empty: 
            logger.warning(f"[{symbol_context}] pd.date_range OB devolvió un índice vacío. Usando índice original.")
            full_index = df_copy.index
    except ValueError as ve:
        logger.error(f"[{symbol_context}] ValueError al crear date_range OB: {ve}. Devolviendo original.")
        return df_copy.reset_index()

    out = df_copy.reindex(full_index, method="ffill").reset_index() 
    
    if "symbol_id" in out.columns and "symbol_id" in df_copy.columns and not df_copy.empty:
        out["symbol_id"] = out["symbol_id"].fillna(df_copy["symbol_id"].iloc[0])
    elif "symbol_id" not in out.columns: 
        out["symbol_id"] = symbol_context 

    for flag_col_name in ["flag_ob_bad_structure", "flag_mid_mad"]:
        if flag_col_name not in out.columns: out[flag_col_name] = False 
        out[flag_col_name] = out[flag_col_name].fillna(False).astype(bool) 
        
    if 'date' in out.columns: out['date'] = out['date'].ffill().bfill() 
    
    logger.info(f"[{symbol_context}] Reindexado OB: {len(df_copy)} -> {len(out)} filas.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Pipeline de limpieza de datos OHLCV y orderbook", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", "-s", nargs="+", help="Símbolo(s) a procesar (usados para OHLCV y como base para OB).")
    parser.add_argument("--db-file", "-d", default=DEFAULT_DB_FILE, help="Ruta a BD SQLite.")
    parser.add_argument("--freq-candle", default=DEFAULT_FREQ_CANDLE, help="Frecuencia velas OHLCV.")
    parser.add_argument("--freq-orderbook", default=DEFAULT_FREQ_ORDERBOOK, help="Frecuencia snapshots OB.")
    parser.add_argument("--max-gap-candles", type=int, default=DEFAULT_MAX_GAP_CANDLES, help="Máx. velas OHLCV a interpolar.")
    parser.add_argument("--skip-ohlcv", action="store_true", help="Omitir limpieza OHLCV.")
    parser.add_argument("--skip-orderbook", action="store_true", help="Omitir limpieza Orderbook.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Nivel de logs.")
    parser.add_argument("--min-start-date", type=str, default=None, help="Fecha de inicio mínima (YYYY-MM-DD) para filtrar datos. Aplica a OHLCV y OB.")
    parser.add_argument("--min-initial-price", type=float, default=None, help="Precio inicial mínimo para retener datos. Aplica a OHLCV (price_close) y OB (mid_price).")

    args = parser.parse_args()
    
    try: 
        log_level_upper = args.log_level.upper()
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=log_level_upper, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
        logger.info(f"Nivel de logging configurado a: {log_level_upper}")
    except ValueError: 
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
        logger.error(f"Nivel de logging inválido: {args.log_level}. Usando INFO por defecto.")

    db_path, freq_candle, freq_orderbook, max_gap_candles = args.db_file, args.freq_candle, args.freq_orderbook, args.max_gap_candles
    
    symbols_to_process_input = []
    conn_check = None
    try: 
        logger.info(f"Verificando BD y tablas fuente en: {db_path}")
        if not os.path.exists(db_path): 
            logger.critical(f"BD NO EXISTE: {db_path}. Por favor, inicializa la BD primero."); return
            
        conn_check = sqlite3.connect(db_path)
        cur_check = conn_check.cursor()
        def table_exists(c, tn): c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;",(tn,)); return c.fetchone() is not None
        
        ohlcv_source_exists = table_exists(cur_check, 'coinapi_ohlcv')
        ob_source_exists = table_exists(cur_check, 'coinapi_orderbook')

        if not ohlcv_source_exists and not args.skip_ohlcv:
            logger.warning("'coinapi_ohlcv' no existe. Saltando procesamiento OHLCV.")
            args.skip_ohlcv = True
        if not ob_source_exists and not args.skip_orderbook:
            logger.warning("'coinapi_orderbook' no existe. Saltando procesamiento Orderbook.")
            args.skip_orderbook = True
            
        if args.symbol:
            symbols_to_process_input = args.symbol
            logger.info(f"Símbolos especificados por el usuario: {symbols_to_process_input}")
        else: 
            symbols_from_ohlcv = set()
            symbols_from_ob = set()
            if ohlcv_source_exists and not args.skip_ohlcv:
                cur_check.execute("SELECT DISTINCT symbol FROM coinapi_ohlcv;")
                symbols_from_ohlcv = {r[0] for r in cur_check.fetchall()}
                logger.info(f"Símbolos encontrados en 'coinapi_ohlcv': {len(symbols_from_ohlcv)}")
            if ob_source_exists and not args.skip_orderbook:
                cur_check.execute("SELECT DISTINCT symbol_id FROM coinapi_orderbook;")
                symbols_from_ob = {r[0] for r in cur_check.fetchall()}
                logger.info(f"Símbolos (IDs) encontrados en 'coinapi_orderbook': {len(symbols_from_ob)}")
            
            symbols_to_process_input = sorted(list(symbols_from_ohlcv.union(symbols_from_ob)))
            if not symbols_to_process_input:
                 logger.critical("No se especificaron símbolos y no se encontraron símbolos en las tablas fuente activas.")
                 return
            logger.info(f"Procesando todos los símbolos/IDs únicos encontrados: {symbols_to_process_input}")

    except Exception as e: 
        logger.critical(f"Error durante la configuración inicial y obtención de lista de símbolos: {e}", exc_info=True)
        return
    finally:
        if conn_check: conn_check.close()
        
    if not symbols_to_process_input: 
        logger.critical("Lista final de símbolos a procesar está vacía. Terminando.")
        return
        
    setup_clean_tables(db_path) 

    total_symbols_attempted = len(symbols_to_process_input)
    ohlcv_processed_ok_count = 0
    ob_processed_ok_count = 0
    symbols_with_errors = []

    for i, symbol_key in enumerate(symbols_to_process_input): 
        logger.info(f"{'='*15} Procesando Símbolo/Clave {i+1}/{total_symbols_attempted}: {symbol_key} {'='*15}")
        
        ohlcv_symbol_name_to_use = None
        ob_symbol_id_to_use = None
        symbol_failed_this_iteration = False
        df_ob_agg_for_ohlcv_fill = None

        if isinstance(symbol_key, str):
            if "FTS_PERP_" in symbol_key: 
                ob_symbol_id_to_use = symbol_key
                try: ohlcv_symbol_name_to_use = f"MEXC_SPOT_{symbol_key.split('FTS_PERP_')[1]}"
                except IndexError: logger.error(f"Error al derivar SPOT de PERP: {symbol_key}")
            elif "SPOT_" in symbol_key: 
                ob_symbol_id_to_use = symbol_key
                ohlcv_symbol_name_to_use = symbol_key
            else: 
                logger.warning(f"'{symbol_key}' no es PERP ni SPOT ID claro. Asumiendo es nombre OHLCV y base para OB ID.")
                ohlcv_symbol_name_to_use = symbol_key
                if "_" in symbol_key and not symbol_key.startswith("MEXC_"):
                     ob_symbol_id_to_use = f"MEXC_SPOT_{symbol_key}"
                else: 
                     ob_symbol_id_to_use = symbol_key 
                if not args.skip_orderbook:
                    logger.info(f"Intentando OB con ID derivado/asumido: {ob_symbol_id_to_use} (desde clave: {symbol_key})")
        else:
            logger.error(f"Clave de símbolo no es string: {symbol_key}. Saltando.")
            symbols_with_errors.append(str(symbol_key)) 
            continue

        if not args.skip_orderbook and ob_symbol_id_to_use:
            logger.info(f"--- Iniciando Pipeline Orderbook para ID: {ob_symbol_id_to_use} ---")
            try:
                df_ob = load_orderbook(db_path, ob_symbol_id_to_use, args.min_start_date, args.min_initial_price)
                if not df_ob.empty:
                    df_ob = detect_ob_structure(df_ob, ob_symbol_id_to_use) 
                    df_ob = detect_ob_outliers(df_ob, ob_symbol_id_to_use) 
                    df_ob = validate_schema(df_ob, ORDERBOOK_SCHEMA, ob_symbol_id_to_use) 
                    df_ob = reindex_ob(df_ob, freq_orderbook, ob_symbol_id_to_use)
                    df_ob = impute_ob(df_ob, ob_symbol_id_to_use)
                    ob_data_check(df_ob, ob_symbol_id_to_use) 
                    if not df_ob.empty:
                        persist_data(df_ob, db_path, 'coinapi_orderbook_clean', ob_symbol_id_to_use, 'OB')
                        ob_processed_ok_count += 1
                        df_ob_agg_for_ohlcv_fill = aggregate_ob_to_candle(df_ob, freq_candle, ob_symbol_id_to_use)
                    else: logger.warning(f"[{ob_symbol_id_to_use}] Orderbook vacío después de la limpieza.")
                else: logger.warning(f"[{ob_symbol_id_to_use}] No se cargaron datos válidos de orderbook.")
            except Exception as e: 
                logger.error(f"[{ob_symbol_id_to_use}] Error FataL en pipeline OB: {e}", exc_info=True)
                symbol_failed_this_iteration = True
        elif not args.skip_orderbook:
             logger.debug(f"Saltando OB para clave '{symbol_key}' (ID no determinado o skip flag).")
        
        if not args.skip_ohlcv and ohlcv_symbol_name_to_use and not symbol_failed_this_iteration:
            logger.info(f"--- Iniciando Pipeline OHLCV para Símbolo: {ohlcv_symbol_name_to_use} ---")
            try:
                df_ohlcv = load_data(db_path, ohlcv_symbol_name_to_use, args.min_start_date, args.min_initial_price)
                if not df_ohlcv.empty:
                    if 'symbol' not in df_ohlcv.columns or df_ohlcv['symbol'].iloc[0] != ohlcv_symbol_name_to_use:
                        df_ohlcv['symbol'] = ohlcv_symbol_name_to_use

                    df_ohlcv = reindex_time(df_ohlcv, freq_candle, ohlcv_symbol_name_to_use) 
                    if df_ohlcv.empty: 
                        logger.warning(f"[{ohlcv_symbol_name_to_use}] OHLCV vacío después de reindexar.")
                    else: 
                        if df_ob_agg_for_ohlcv_fill is not None and not df_ob_agg_for_ohlcv_fill.empty: 
                            df_ohlcv = fill_ohlcv_from_ob(df_ohlcv, df_ob_agg_for_ohlcv_fill, ohlcv_symbol_name_to_use)
                        
                        df_ohlcv = detect_structure(df_ohlcv, ohlcv_symbol_name_to_use) 
                        df_ohlcv = detect_outliers(df_ohlcv, ohlcv_symbol_name_to_use)  
                        df_ohlcv = impute_data(df_ohlcv, max_gap_candles, ohlcv_symbol_name_to_use)
                        df_ohlcv = validate_schema(df_ohlcv, OHLCV_SCHEMA, ohlcv_symbol_name_to_use) 
                        
                        df_ohlcv = detect_structure(df_ohlcv, ohlcv_symbol_name_to_use) 
                        if "flag_bad_structure" in df_ohlcv.columns and pd.api.types.is_bool_dtype(df_ohlcv["flag_bad_structure"]):
                             if (dropped_post_impute := df_ohlcv["flag_bad_structure"].sum()) > 0: 
                                 df_ohlcv = df_ohlcv[~df_ohlcv["flag_bad_structure"]] 
                                 logger.info(f"[{ohlcv_symbol_name_to_use}] Eliminadas {dropped_post_impute} filas OHLCV con mala estructura post-imputación.")
                        
                        price_cols_final = ["price_open","price_high","price_low","price_close"]
                        existing_pc_final_check = [c for c in price_cols_final if c in df_ohlcv.columns]
                        if existing_pc_final_check:
                             rows_before_final_dropna = len(df_ohlcv)
                             df_ohlcv = df_ohlcv.dropna(subset=existing_pc_final_check) 
                             dropped_final_nans = rows_before_final_dropna - len(df_ohlcv)
                             if dropped_final_nans > 0:
                                 logger.warning(f"[{ohlcv_symbol_name_to_use}] Eliminadas {dropped_final_nans} filas OHLCV con NaNs en precios antes de persistir.")
                        
                        if not df_ohlcv.empty:
                            final_cols_ohlcv = list(OHLCV_SCHEMA.columns.keys()) 
                            df_final_ohlcv = reduce_columns(df_ohlcv, final_cols_ohlcv, ohlcv_symbol_name_to_use) 
                            final_data_check(df_final_ohlcv, source="OHLCV", symbol_context=ohlcv_symbol_name_to_use)
                            persist_data(df_final_ohlcv, db_path, 'coinapi_ohlcv_clean', ohlcv_symbol_name_to_use, 'OHLCV')
                            ohlcv_processed_ok_count += 1
                        else: 
                            logger.warning(f"[{ohlcv_symbol_name_to_use}] OHLCV final vacío después de limpieza completa.")
                else: 
                    logger.warning(f"[{ohlcv_symbol_name_to_use}] No se cargaron datos válidos de OHLCV.")
            except Exception as e: 
                logger.error(f"[{ohlcv_symbol_name_to_use}] Error FataL en pipeline OHLCV: {e}", exc_info=True)
                symbol_failed_this_iteration = True
        elif not args.skip_ohlcv:
            logger.debug(f"Saltando OHLCV para clave '{symbol_key}' (nombre no determinado o pipeline OB falló).")
        
        if symbol_failed_this_iteration: 
            symbols_with_errors.append(symbol_key)
            logger.error(f"Símbolo/Clave {symbol_key} marcado con ERRORES durante el procesamiento.")
        else: 
            logger.info(f"Símbolo/Clave {symbol_key} procesado.")

    logger.info(f"{'='*20} Resumen del Proceso {'='*20}")
    logger.info(f"Símbolos/Claves totales intentados: {total_symbols_attempted}")
    if not args.skip_orderbook: logger.info(f"Orderbook procesado y persistido (OK): {ob_processed_ok_count} símbolos/IDs.")
    if not args.skip_ohlcv: logger.info(f"OHLCV procesado y persistido (OK): {ohlcv_processed_ok_count} símbolos.")
    if symbols_with_errors: 
        logger.warning(f"Símbolos/Claves con ERRORES FATALES durante su pipeline ({len(symbols_with_errors)}): {sorted(list(set(symbols_with_errors)))}")
    else: 
        logger.info("Todos los símbolos/claves se procesaron sin errores fatales reportados en sus respectivos pipelines.")
    logger.info("=" * (40 + len(" Resumen del Proceso ")))

if __name__ == "__main__":
    try: 
        main()
        logger.info("<<<<<<<<<< Proceso de Limpieza de Tablas Completado >>>>>>>>>>")
    except Exception as e_main: 
        logger.critical(f"!!!!!!!!!! Error FataL en ejecución principal de clean_tables.py: {e_main} !!!!!!!!!!", exc_info=True)
