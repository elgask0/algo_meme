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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
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

def load_orderbook_clean(db_path: str, symbol_id: str, date_filter: str = None) -> pd.DataFrame:
    logger.info(f"[{symbol_id}] Cargando datos de orderbook limpio" + (f" en fecha {date_filter}" if date_filter else ""))
    conn = sqlite3.connect(db_path)
    query = "SELECT ts, date, bid1_px, bid1_sz, bid2_px, bid2_sz, bid3_px, bid3_sz, ask1_px, ask1_sz, ask2_px, ask2_sz, ask3_px, ask3_sz FROM coinapi_orderbook_clean WHERE symbol_id = ?"
    params = [symbol_id]
    if date_filter:
        query += " AND date = ?" 
        params.append(date_filter)
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            logger.warning(f"[{symbol_id}] No hay datos de orderbook limpio.")
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["ts"], errors='coerce')
        # 'date' column might not be essential for VWAP calculation itself, handle NaT if it occurs
        if 'date' in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce') 
        
        df = df.dropna(subset=['ts']) # Critical for time series operations
        if df.empty:
            logger.warning(f"[{symbol_id}] No hay datos de orderbook limpio después de eliminar NaT en 'ts'.")
            return pd.DataFrame()

        # Convert price/size columns to numeric, coercing errors
        px_sz_cols = [col for col in df.columns if col.endswith(('_px', '_sz'))]
        for col in px_sz_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where essential L1 prices are NaN, as they are needed for VWAP
        df.dropna(subset=['bid1_px', 'ask1_px', 'bid1_sz', 'ask1_sz'], inplace=True)

        logger.info(f"[{symbol_id}] Snapshots de orderbook limpio cargados y 'ts' convertido: {len(df)} filas.")
    except Exception as e:
        logger.error(f"[{symbol_id}] Error cargando datos de orderbook limpio: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()
    return df

def calculate_vwap_and_depth(df: pd.DataFrame, symbol_id: str) -> pd.DataFrame:
    logger.debug(f"[{symbol_id}] Calculando VWAP y profundidad para {len(df)} snapshots.")
    if df.empty: return pd.DataFrame(columns=["ts", "vwap", "depth_sum_sz"]) # Devolver DF vacío con columnas esperadas
    
    df_calc = df.copy()
    # Define bid/ask price and size columns for up to 3 levels
    bid_px_cols = [f"bid{i}_px" for i in range(1, 4)]
    bid_sz_cols = [f"bid{i}_sz" for i in range(1, 4)]
    ask_px_cols = [f"ask{i}_px" for i in range(1, 4)]
    ask_sz_cols = [f"ask{i}_sz" for i in range(1, 4)]

    all_level_cols = bid_px_cols + bid_sz_cols + ask_px_cols + ask_sz_cols

    # Ensure all expected columns exist, fill with 0 if not (already done in load_orderbook_clean by coerce and fillna for px/sz)
    # but good to be defensive if this function is called with data from other sources
    for col in all_level_cols:
        if col not in df_calc.columns:
            df_calc[col] = 0.0
        # Ensure numeric and fill NaNs with 0 for calculation (VWAP should be NaN if all sizes are 0)
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)

    total_bid_value = df_calc[bid_px_cols[0]] * df_calc[bid_sz_cols[0]] + \
                      df_calc[bid_px_cols[1]] * df_calc[bid_sz_cols[1]] + \
                      df_calc[bid_px_cols[2]] * df_calc[bid_sz_cols[2]]
    
    total_ask_value = df_calc[ask_px_cols[0]] * df_calc[ask_sz_cols[0]] + \
                      df_calc[ask_px_cols[1]] * df_calc[ask_sz_cols[1]] + \
                      df_calc[ask_px_cols[2]] * df_calc[ask_sz_cols[2]]

    total_bid_size = df_calc[bid_sz_cols].sum(axis=1)
    total_ask_size = df_calc[ask_sz_cols].sum(axis=1)
    
    df_calc["total_value"] = total_bid_value + total_ask_value
    df_calc["total_size"] = total_bid_size + total_ask_size
    
    df_calc["vwap"] = np.where(df_calc["total_size"] > 1e-9, df_calc["total_value"] / df_calc["total_size"], np.nan)
    df_calc["depth_sum_sz"] = df_calc["total_size"] 
    
    logger.debug(f"[{symbol_id}] VWAP y profundidad calculados. {df_calc['vwap'].notna().sum()} VWAPs válidos.")
    return df_calc[["ts", "vwap", "depth_sum_sz"]]

def aggregate_vwap(df_vwap_snapshots: pd.DataFrame, symbol_id: str, freq: str) -> pd.DataFrame:
    logger.info(f"[{symbol_id}] Agregando VWAP a frecuencia {freq}...")
    if df_vwap_snapshots.empty or 'vwap' not in df_vwap_snapshots.columns:
        logger.warning(f"[{symbol_id}] DataFrame de snapshots VWAP vacío o sin columna 'vwap'. No se puede agregar.")
        return pd.DataFrame()

    df_with_ts_index = df_vwap_snapshots.set_index("ts")
    if not isinstance(df_with_ts_index.index, pd.DatetimeIndex):
        logger.error(f"[{symbol_id}] El índice no es DatetimeIndex después de set_index('ts'). No se puede agregar.")
        return pd.DataFrame()
    
    # Determinar el rango completo de los datos
    min_ts = df_with_ts_index.index.min()
    max_ts = df_with_ts_index.index.max()

    if pd.isna(min_ts) or pd.isna(max_ts):
        logger.warning(f"[{symbol_id}] No se pudo determinar el rango de tiempo (min_ts o max_ts es NaT). No se puede rellenar el índice.")
        # Proceder con la agregación sin rellenar el índice completo si el rango no es válido
        # Esto mantendrá el comportamiento original si el rango no se puede determinar.
        # O, alternativamente, devolver un DataFrame vacío:
        # return pd.DataFrame() 
    else:
        logger.info(f"[{symbol_id}] Rango de datos para agregación: {min_ts} a {max_ts}")
        # Crear un índice de tiempo completo para la frecuencia dada
        full_range_index = pd.date_range(start=min_ts.floor(freq), end=max_ts.ceil(freq), freq=freq, name="ts_start")
        
        # Reindexar antes de agrupar para asegurar que todos los intervalos estén presentes
        # Los valores de 'vwap' y 'depth_sum_sz' serán NaN para los nuevos intervalos
        df_with_ts_index = df_with_ts_index.reindex(df_with_ts_index.index.union(full_range_index.tz_localize(None) if full_range_index.tz is not None else full_range_index))
        # El reindex puede haber introducido NaNs en el índice si full_range_index tenía timestamps no presentes en df_with_ts_index.index
        # Y el floor/ceil puede haber cambiado el tipo de índice si no se maneja con cuidado.
        # Mejor reindexar sobre el DataFrame original agrupado por el ts_start deseado.
    
    # Agrupar por la frecuencia deseada
    # ts_start será el inicio de cada bucket de 'freq'
    # ts_end se calcula a partir de ts_start
    df_with_ts_index["ts_start_agg"] = df_with_ts_index.index.floor(freq)
    
    aggregated = (
        df_with_ts_index.groupby("ts_start_agg")
          .agg(
              mark_price_median = ("vwap", "median"), 
              depth_sum_sz_median = ("depth_sum_sz", "median"),
              n_snapshots = ("vwap", "count") # Contará solo los no-NaN
          )
    ).rename_axis("ts_start") # El índice del grupo es ahora ts_start

    # Crear un DataFrame con el índice completo de la frecuencia deseada
    if not pd.isna(min_ts) and not pd.isna(max_ts):
        expected_index = pd.date_range(start=min_ts.floor(freq), end=max_ts.ceil(freq), freq=freq, name="ts_start")
        aggregated = aggregated.reindex(expected_index)
        logger.info(f"[{symbol_id}] Reindexado a índice completo de {len(expected_index)} filas para frecuencia {freq}.")


    # Rellenar NaNs en mark_price usando forward fill
    # Esto propagará el último mark_price conocido a los intervalos vacíos.
    if 'mark_price_median' in aggregated.columns:
        nans_before_ffill = aggregated['mark_price_median'].isna().sum()
        aggregated['mark_price'] = aggregated['mark_price_median'].ffill()
        nans_after_ffill = aggregated['mark_price'].isna().sum()
        if nans_before_ffill > 0:
            logger.info(f"[{symbol_id}] 'mark_price' rellenado con ffill(): {nans_before_ffill - nans_after_ffill} NaNs rellenados.")
        # Si aún quedan NaNs al principio, se podrían rellenar con bfill o dejar como NaN
        # aggregated['mark_price'] = aggregated['mark_price'].bfill() # Opcional
    else:
        aggregated['mark_price'] = np.nan


    # Rellenar depth_sum_sz (podría ser con 0 o ffill)
    if 'depth_sum_sz_median' in aggregated.columns:
        aggregated['depth_sum_sz'] = aggregated['depth_sum_sz_median'].fillna(0) # Rellenar NaNs con 0
    else:
        aggregated['depth_sum_sz'] = 0
        
    # n_snapshots ya es 0 para los intervalos que eran originalmente vacíos.
    if 'n_snapshots' not in aggregated.columns:
        aggregated['n_snapshots'] = 0
    else:
        aggregated['n_snapshots'] = aggregated['n_snapshots'].fillna(0).astype(int)


    aggregated["ts_end"] = aggregated.index + pd.Timedelta(freq) - pd.Timedelta(seconds=1) # ts_end es el final del bucket
    aggregated["symbol_id"] = symbol_id
    aggregated = aggregated.reset_index() # Mover ts_start de índice a columna
    
    cols = ["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"]
    # Asegurar que todas las columnas existan, incluso si son todo NaN o 0
    for col in cols:
        if col not in aggregated.columns:
            if col == "mark_price" or col == "depth_sum_sz":
                aggregated[col] = np.nan
            elif col == "n_snapshots":
                 aggregated[col] = 0
            # ts_start, ts_end, symbol_id deberían existir
            
    aggregated = aggregated[cols]
    logger.info(f"[{symbol_id}] Agregados {len(aggregated)} buckets de mark price (frecuencia {freq}).")
    return aggregated


def get_ohlcv_symbol_for_mark_price(mark_price_symbol_id: str) -> str:
    """Determina el nombre del símbolo OHLCV correspondiente al symbol_id del mark_price."""
    if "FTS_PERP_" in mark_price_symbol_id:
        try:
            asset_pair = mark_price_symbol_id.split("FTS_PERP_")[1]
            return f"MEXC_SPOT_{asset_pair}"
        except IndexError:
            logger.error(f"No se pudo derivar el símbolo SPOT de OHLCV para el PERP ID '{mark_price_symbol_id}'. Usando el ID original.")
            return mark_price_symbol_id # Fallback, podría no encontrar datos
    elif "SPOT_" in mark_price_symbol_id: # Asume que el symbol_id del SPOT es el mismo que el nombre en OHLCV
        return mark_price_symbol_id
    else:
        logger.warning(f"Formato de symbol_id '{mark_price_symbol_id}' no reconocido para mapeo OHLCV. Usando tal cual.")
        return mark_price_symbol_id


def cross_check_mark_price(df_mark: pd.DataFrame, db_path: str, mark_price_symbol_id: str, threshold: float) -> pd.DataFrame:
    if df_mark.empty:
        logger.warning(f"[{mark_price_symbol_id}] DataFrame de mark_price vacío para cross-check.")
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)

    ohlcv_symbol_name = get_ohlcv_symbol_for_mark_price(mark_price_symbol_id)
    logger.info(f"[{mark_price_symbol_id}] Iniciando cross-check con OHLCV symbol '{ohlcv_symbol_name}'.")

    conn = sqlite3.connect(db_path)
    # Asegurar que ts_start sea datetime antes de min/max
    df_mark['ts_start'] = pd.to_datetime(df_mark['ts_start'])
    min_ts_start_dt = df_mark["ts_start"].min()
    max_ts_start_dt = df_mark["ts_start"].max()
    
    min_ts_start_str = min_ts_start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(min_ts_start_dt) else None
    max_ts_start_str = max_ts_start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f') if pd.notna(max_ts_start_dt) else None

    if not min_ts_start_str or not max_ts_start_str:
        logger.warning(f"[{mark_price_symbol_id}] Rango de ts_start inválido para cross-check.")
        conn.close()
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
        
    query_ohlcv = "SELECT time_period_start, price_close FROM coinapi_ohlcv_clean WHERE symbol = ? AND time_period_start BETWEEN ? AND ?"
    try:
        df_ohlcv = pd.read_sql_query(query_ohlcv, conn, params=[ohlcv_symbol_name, min_ts_start_str, max_ts_start_str])
        if df_ohlcv.empty:
            logger.warning(f"[{mark_price_symbol_id}] No se encontraron datos OHLCV para cross-check para el símbolo OHLCV '{ohlcv_symbol_name}'.")
            return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
        df_ohlcv["time_period_start"] = pd.to_datetime(df_ohlcv["time_period_start"])
    except Exception as e:
        logger.error(f"[{mark_price_symbol_id}] Error cargando OHLCV para cross-check (símbolo OHLCV '{ohlcv_symbol_name}'): {e}", exc_info=True)
        return df_mark.assign(ohlcv_price_close=np.nan, relative_difference=np.nan, flag_anomaly=False)
    finally: 
        conn.close()
    
    df_mark_copy = df_mark.copy() # ts_start ya es datetime
    df_merged = pd.merge(df_mark_copy, df_ohlcv, left_on="ts_start", right_on="time_period_start", how="left")
    
    # Calcular diferencia relativa solo donde mark_price y ohlcv_price_close son válidos
    valid_prices_mask = df_merged["mark_price"].notna() & df_merged["price_close"].notna() & (df_merged["price_close"].abs() > 1e-9)
    
    df_merged["relative_difference"] = np.nan # Inicializar
    df_merged.loc[valid_prices_mask, "relative_difference"] = \
        (df_merged.loc[valid_prices_mask, "mark_price"] - df_merged.loc[valid_prices_mask, "price_close"]).abs() / df_merged.loc[valid_prices_mask, "price_close"]
    
    df_merged["flag_anomaly"] = (df_merged["relative_difference"] > threshold) & df_merged["mark_price"].notna()
    
    num_anomalies = df_merged['flag_anomaly'].sum()
    logger.info(f"[{mark_price_symbol_id}] Cross-check completado: {num_anomalies} anomalías encontradas (umbral {threshold*100:.1f}%).")
    return df_merged.rename(columns={"price_close": "ohlcv_price_close"})


def format_value_for_sqlite(value):
    if pd.isna(value): 
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%f')
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)

def persist_results(df_to_persist: pd.DataFrame, table_name: str, db_path: str):
    if df_to_persist.empty:
        logger.info(f"No hay datos para persistir en la tabla {table_name}.")
        return
    
    conn = sqlite3.connect(db_path)
    try:
        columns = df_to_persist.columns.tolist()
        placeholders = ','.join('?' * len(columns))
        
        sql_action = "INSERT OR REPLACE" if table_name == "mark_price_vwap" else "INSERT"
        sql = f"{sql_action} INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        list_of_tuples = []
        for record_tuple_raw in df_to_persist.itertuples(index=False): 
            processed_record = [format_value_for_sqlite(value) for value in record_tuple_raw]
            list_of_tuples.append(tuple(processed_record))
        
        if not list_of_tuples: 
            logger.warning(f"No hay tuplas válidas para insertar en {table_name} después de la preparación."); return
        
        with conn: 
            conn.executemany(sql, list_of_tuples)
        logger.info(f"{len(df_to_persist)} filas procesadas para la tabla {table_name}.") 
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

    # Configurar logging
    try: 
        log_level_upper = args.log_level.upper()
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler) # Limpiar handlers existentes
        logging.basicConfig(level=log_level_upper, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
        logger.info(f"Nivel de logging configurado a: {log_level_upper}")
    except ValueError: 
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] [%(name)s] %(message)s")
        logger.error(f"Nivel de logging inválido: {args.log_level}. Usando INFO por defecto.")


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
            # Leer de coinapi_orderbook_clean ya que es la fuente para mark_price
            symbols_to_process = pd.read_sql_query("SELECT DISTINCT symbol_id FROM coinapi_orderbook_clean;", conn_sym)["symbol_id"].tolist()
            if not symbols_to_process:
                logger.warning("No se encontraron símbolos en 'coinapi_orderbook_clean'. No hay nada que procesar para mark_price.")
                return
            logger.info(f"Procesando todos los símbolos encontrados en 'coinapi_orderbook_clean': {symbols_to_process}")
        except Exception as e:
            logger.error(f"No se pudo obtener la lista de símbolos de 'coinapi_orderbook_clean': {e}", exc_info=True); return
        finally:
            conn_sym.close()
            
    processed_symbols_count = 0
    for symbol_id in symbols_to_process: # 'symbol' aquí es en realidad symbol_id
        logger.info(f"--- Procesando Mark Price para Símbolo: {symbol_id} (Freq: {args.freq}) ---")
        
        df_ob = load_orderbook_clean(db_path, symbol_id, args.date_filter)
        if df_ob.empty:
            logger.warning(f"No hay snapshots de orderbook limpio para {symbol_id}" + (f" en {args.date_filter}" if args.date_filter else "") + ". Saltando.")
            continue

        df_vwap_snapshots = calculate_vwap_and_depth(df_ob, symbol_id)
        if df_vwap_snapshots.empty or 'vwap' not in df_vwap_snapshots.columns or df_vwap_snapshots['vwap'].isnull().all():
            logger.warning(f"No se pudo calcular VWAP para {symbol_id}. Saltando.")
            continue
            
        df_mark_price_agg = aggregate_vwap(df_vwap_snapshots, symbol_id, args.freq)
        if df_mark_price_agg.empty:
            logger.warning(f"No se generaron buckets de mark price agregado para {symbol_id}. Saltando.")
            continue

        # El 'symbol' para cross_check_mark_price debe ser el nombre del símbolo OHLCV
        df_checked = cross_check_mark_price(df_mark_price_agg, db_path, symbol_id, args.cross_check_threshold)
        
        anomalies = df_checked[df_checked["flag_anomaly"] == True].copy() 
        # Persistir solo los mark_prices que NO son anomalías Y donde mark_price no es NaN
        # Los NaNs en mark_price aquí significan que el intervalo fue rellenado por el full_index y no hubo snapshots
        # o que el VWAP de los snapshots fue NaN (ej. total_size = 0).
        # El ffill en aggregate_vwap rellena estos, pero n_snapshots seguirá siendo 0.
        # Decisión: persistir todos los mark_price, incluso los ffilled, ya que representan la mejor estimación.
        # La columna n_snapshots indicará si el valor fue calculado o ffilled.
        # Sin embargo, para la tabla de anomalías, solo queremos filas donde hubo un mark_price calculado.
        
        valid_mark_prices_for_persistence = df_checked.copy() # Empezar con todo
        # No filtrar aquí basado en flag_anomaly para la tabla mark_price_vwap, ya que queremos persistir todos los buckets.
        # El ffill ya se hizo en aggregate_vwap.

        anomalies_to_persist = anomalies[anomalies['mark_price'].notna() & anomalies['ohlcv_price_close'].notna()]
        if not anomalies_to_persist.empty:
            cols_anomalies = ["symbol_id", "ts_start", "mark_price", "ohlcv_price_close", "relative_difference"]
            # Asegurar que las columnas existan en anomalies_to_persist
            cols_anomalies_existing = [col for col in cols_anomalies if col in anomalies_to_persist.columns]
            persist_results(anomalies_to_persist[cols_anomalies_existing], "mark_price_anomalies", db_path)
        
        if not valid_mark_prices_for_persistence.empty:
            cols_mark_price = ["symbol_id", "ts_start", "ts_end", "mark_price", "depth_sum_sz", "n_snapshots"] 
            cols_mark_price_existing = [col for col in cols_mark_price if col in valid_mark_prices_for_persistence.columns]
            persist_results(valid_mark_prices_for_persistence[cols_mark_price_existing], "mark_price_vwap", db_path)
        else:
            logger.info(f"No hay mark prices (válidos o no) para persistir para {symbol_id}.") # Esto no debería ocurrir si aggregate_vwap funcionó
        
        processed_symbols_count +=1
        logger.info(f"--- Mark Price para Símbolo: {symbol_id} completado ---")

    logger.info(f"Proceso de cálculo de Mark Price completado. {processed_symbols_count} símbolos procesados.")

if __name__ == "__main__":
    main()
