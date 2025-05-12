#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
from dotenv import load_dotenv
import logging
from datetime import datetime # Importar datetime para isinstance

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Carga de Variables de Entorno y Constantes ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
DEFAULT_FREQ = "5min"
DEFAULT_CROSS_CHECK_THRESHOLD = 0.02

def setup_output_tables(conn):
    """Crea las tablas de salida. Elimina las existentes para asegurar el esquema correcto."""
    with conn:
        logger.info("Eliminando tablas 'mark_price_vwap' y 'mark_price_anomalies' si existen para recrearlas...")
        conn.execute("DROP TABLE IF EXISTS mark_price_vwap;")
        conn.execute("DROP TABLE IF EXISTS mark_price_anomalies;")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mark_price_vwap (
            symbol_id TEXT NOT NULL,
            ts_start TEXT NOT NULL,
            ts_end TEXT NOT NULL,
            mark_price REAL,
            depth_sum_sz REAL, 
            n_snapshots INTEGER,
            PRIMARY KEY (symbol_id, ts_start)
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mark_price_anomalies (
            symbol_id TEXT,
            ts_start TEXT,
            mark_price REAL,
            ohlcv_price_close REAL,
            relative_difference REAL
        );
        """)
        logger.info("Tablas 'mark_price_vwap' y 'mark_price_anomalies' creadas/recreadas.")

def load_orderbook_clean(db_path: str, symbol: str, date_filter: str = None) -> pd.DataFrame:
    logger.info(f"Cargando datos de orderbook limpio para {symbol}" + (f" en fecha {date_filter}" if date_filter else ""))
    conn = sqlite3.connect(db_path)
    query = "SELECT ts, date, bid1_px, bid1_sz, bid2_px, bid2_sz, bid3_px, bid3_sz, ask1_px, ask1_sz, ask2_px, ask2_sz, ask3_px, ask3_sz FROM coinapi_orderbook_clean WHERE symbol_id = ?"
    params = [symbol]
    if date_filter:
        query += " AND date = ?" 
        params.append(date_filter)
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        df["ts"] = pd.to_datetime(df["ts"], errors='coerce')
        df["date"] = pd.to_datetime(df["date"], errors='coerce') 
        df = df.dropna(subset=['ts']) 
        logger.info(f"Snapshots cargados y 'ts' convertido para {symbol}: {len(df)}")
    except Exception as e:
        logger.error(f"Error cargando datos de orderbook para {symbol}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    return df

def calculate_vwap_and_depth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df_calc = df.copy()
    bid_sz_cols = ["bid1_sz", "bid2_sz", "bid3_sz"]; ask_sz_cols = ["ask1_sz", "ask2_sz", "ask3_sz"]
    bid_px_cols = ["bid1_px", "bid2_px", "bid3_px"]; ask_px_cols = ["ask1_px", "ask2_px", "ask3_px"]
    for col_group in [bid_sz_cols, ask_sz_cols, bid_px_cols, ask_px_cols]:
        for col in col_group:
            if col not in df_calc.columns: df_calc[col] = 0.0
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)
    total_bid_value = sum(df_calc[px] * df_calc[sz] for px, sz in zip(bid_px_cols, bid_sz_cols))
    total_ask_value = sum(df_calc[px] * df_calc[sz] for px, sz in zip(ask_px_cols, ask_sz_cols))
    total_bid_size = sum(df_calc[sz] for sz in bid_sz_cols)
    total_ask_size = sum(df_calc[sz] for sz in ask_sz_cols)
    df_calc["total_value"] = total_bid_value + total_ask_value
    df_calc["total_size"] = total_bid_size + total_ask_size
    df_calc["vwap"] = np.where(df_calc["total_size"] > 1e-9, df_calc["total_value"] / df_calc["total_size"], np.nan)
    df_calc["depth_sum_sz"] = df_calc["total_size"] 
    return df_calc[["ts", "vwap", "depth_sum_sz"]]

def aggregate_vwap(df: pd.DataFrame, symbol_id: str, freq: str) -> pd.DataFrame:
    if df.empty or 'vwap' not in df.columns: return pd.DataFrame()
    df_agg = df.set_index("ts")
    if not isinstance(df_agg.index, pd.DatetimeIndex): return pd.DataFrame()
    df_agg["ts_start"] = df_agg.index.floor(freq)
    df_agg["ts_end"]   = df_agg["ts_start"] + pd.Timedelta(freq)
    aggregated = (
        df_agg.groupby("ts_start")
          .agg(
              mark_price     = ("vwap", "median"), 
              depth_sum_sz   = ("depth_sum_sz", "median"),
              n_snapshots    = ("vwap", "count"), 
              ts_end         = ("ts_end", "first") 
          )
    ).reset_index()
    aggregated["symbol_id"] = symbol_id
    cols = ["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"]
    aggregated = aggregated[cols]
    logger.info(f"Agregados {len(aggregated)} buckets de mark price para {symbol_id}.")
    return aggregated

def cross_check_mark_price(df_mark: pd.DataFrame, db_path: str, symbol: str, threshold: float) -> pd.DataFrame:
    if df_mark.empty:
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
    conn = sqlite3.connect(db_path)
    min_ts_start_dt = pd.to_datetime(df_mark["ts_start"].min())
    max_ts_start_dt = pd.to_datetime(df_mark["ts_start"].max())
    min_ts_start_str = min_ts_start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(min_ts_start_dt) else None
    max_ts_start_str = max_ts_start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(max_ts_start_dt) else None

    if not min_ts_start_str or not max_ts_start_str:
        logger.warning(f"Rango de ts_start inválido para cross-check de {symbol}.")
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
        
    query_ohlcv = "SELECT time_period_start, price_close FROM coinapi_ohlcv_clean WHERE symbol = ? AND time_period_start BETWEEN ? AND ?"
    try:
        df_ohlcv = pd.read_sql_query(query_ohlcv, conn, params=[symbol, min_ts_start_str, max_ts_start_str])
        df_ohlcv["time_period_start"] = pd.to_datetime(df_ohlcv["time_period_start"])
    except Exception as e:
        logger.error(f"Error cargando OHLCV para cross-check de {symbol}: {e}")
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
    finally: conn.close()
    if df_ohlcv.empty:
        logger.warning(f"No se encontraron datos OHLCV para cross-check para {symbol}.")
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
    
    df_mark_copy = df_mark.copy()
    df_mark_copy["ts_start"] = pd.to_datetime(df_mark_copy["ts_start"])
    df_merged = pd.merge(df_mark_copy, df_ohlcv, left_on="ts_start", right_on="time_period_start", how="left")
    df_merged["relative_difference"] = np.where(
        df_merged["price_close"].notna() & (df_merged["price_close"].abs() > 1e-9), 
        (df_merged["mark_price"] - df_merged["price_close"]).abs() / df_merged["price_close"],
        np.nan
    )
    df_merged["flag_anomaly"] = df_merged["relative_difference"] > threshold
    logger.info(f"Cross-check para {symbol}: {df_merged['flag_anomaly'].sum()} anomalías encontradas (umbral {threshold*100:.1f}%).")
    return df_merged.rename(columns={"price_close": "ohlcv_price_close"})

def format_value_for_sqlite(value):
    """Prepara un valor individual para la inserción en SQLite."""
    if pd.isna(value): # Cubre pd.NaT, np.nan, None
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%f')
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value) # Como último recurso, convertir a string

def persist_results(df_to_persist: pd.DataFrame, table_name: str, db_path: str):
    if df_to_persist.empty:
        logger.info(f"No hay datos para persistir en la tabla {table_name}.")
        return

    # No es necesario hacer df_copy aquí si solo vamos a iterar para crear tuplas
    
    conn = sqlite3.connect(db_path)
    try:
        columns = df_to_persist.columns.tolist()
        placeholders = ','.join('?' * len(columns))
        
        sql_action = "INSERT OR REPLACE" if table_name == "mark_price_vwap" else "INSERT"
        sql = f"{sql_action} INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Convertir cada fila del DataFrame a una tupla de valores formateados para SQLite
        list_of_tuples = []
        for record_tuple_raw in df_to_persist.itertuples(index=False): 
            processed_record = [format_value_for_sqlite(value) for value in record_tuple_raw]
            list_of_tuples.append(tuple(processed_record))
        
        if not list_of_tuples: 
            logger.warning(f"No hay tuplas válidas para insertar en {table_name} después de la preparación."); return
        
        with conn: 
            conn.executemany(sql, list_of_tuples)
        logger.info(f"{len(df_to_persist)} filas procesadas para la tabla {table_name}.") # Loguear len(df_to_persist)
    except Exception as e:
        logger.error(f"Error al persistir datos en {table_name}: {e}", exc_info=True)
        if 'list_of_tuples' in locals() and list_of_tuples:
            logger.error(f"Primeros registros que se intentaron insertar en {table_name}:\n{list_of_tuples[:2]}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Calcula Mark Price (VWAP) desde datos de orderbook limpio y lo valida contra OHLCV.")
    parser.add_argument("--symbol", "-s", nargs="*", help="Símbolo(s) a procesar. Si no se especifica, procesa todos.")
    parser.add_argument("--db-file", "-d", default=DEFAULT_DB_FILE, help="Ruta al archivo de base de datos SQLite.")
    parser.add_argument("--freq", "-f", default=DEFAULT_FREQ, help="Frecuencia de agregación para el mark price (ej. '5min').")
    parser.add_argument("--date-filter", help="Fecha específica (YYYY-MM-DD) para filtrar los snapshots del orderbook.")
    parser.add_argument("--cross-check-threshold", type=float, default=DEFAULT_CROSS_CHECK_THRESHOLD, help="Umbral de diferencia relativa para marcar anomalías en validación cruzada.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Nivel de logging.")
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    db_path = args.db_file
    if not os.path.exists(db_path):
        logger.critical(f"Archivo de base de datos no encontrado: {db_path}"); return

    conn_setup = sqlite3.connect(db_path)
    setup_output_tables(conn_setup) 
    conn_setup.close()

    symbols_to_process = args.symbol
    if not symbols_to_process:
        conn_sym = sqlite3.connect(db_path)
        try:
            symbols_to_process = pd.read_sql_query("SELECT DISTINCT symbol_id FROM coinapi_orderbook_clean;", conn_sym)["symbol_id"].tolist()
            if not symbols_to_process:
                logger.warning("No se encontraron símbolos en 'coinapi_orderbook_clean'. No hay nada que procesar.")
                return
            logger.info(f"Procesando todos los símbolos encontrados: {symbols_to_process}")
        except Exception as e:
            logger.error(f"No se pudo obtener la lista de símbolos: {e}"); return
        finally:
            conn_sym.close()
            
    processed_symbols_count = 0
    for symbol in symbols_to_process:
        logger.info(f"--- Procesando Mark Price para Símbolo: {symbol} (Freq: {args.freq}) ---")
        
        df_ob = load_orderbook_clean(db_path, symbol, args.date_filter)
        if df_ob.empty:
            logger.warning(f"No hay snapshots de orderbook limpio para {symbol}" + (f" en {args.date_filter}" if args.date_filter else "") + ". Saltando.")
            continue

        df_vwap_snapshots = calculate_vwap_and_depth(df_ob)
        if df_vwap_snapshots.empty or 'vwap' not in df_vwap_snapshots.columns or df_vwap_snapshots['vwap'].isnull().all():
            logger.warning(f"No se pudo calcular VWAP para {symbol}. Saltando.")
            continue
            
        df_mark_price_agg = aggregate_vwap(df_vwap_snapshots, symbol, args.freq)
        if df_mark_price_agg.empty:
            logger.warning(f"No se generaron buckets de mark price agregado para {symbol}. Saltando.")
            continue

        df_checked = cross_check_mark_price(df_mark_price_agg, db_path, symbol, args.cross_check_threshold)
        
        anomalies = df_checked[df_checked["flag_anomaly"] == True].copy() 
        valid_mark_prices = df_checked[df_checked["flag_anomaly"] == False].copy()

        if not anomalies.empty:
            cols_anomalies = ["symbol_id", "ts_start", "mark_price", "ohlcv_price_close", "relative_difference"]
            anomalies_to_persist = anomalies[cols_anomalies] 
            persist_results(anomalies_to_persist, "mark_price_anomalies", db_path)
        
        if not valid_mark_prices.empty:
            cols_mark_price = ["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"] 
            valid_to_persist = valid_mark_prices[cols_mark_price] 
            persist_results(valid_to_persist, "mark_price_vwap", db_path)
        else:
            logger.info(f"No hay mark prices válidos para persistir para {symbol} después del cross-check.")
        
        processed_symbols_count +=1
        logger.info(f"--- Mark Price para Símbolo: {symbol} completado ---")

    logger.info(f"Proceso de cálculo de Mark Price completado. {processed_symbols_count} símbolos procesados.")

if __name__ == "__main__":
    main()
