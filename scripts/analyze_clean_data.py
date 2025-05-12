#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import argparse
import logging
from datetime import timedelta
import io # Para capturar info()

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Constantes ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
CHARTS_OUTPUT_DIR = "anomalies_summary_charts"
DEFAULT_AGG_FREQ = "5min" 

def get_symbols_from_table(conn, table_name, symbol_column_name="symbol_id"):
    try:
        query = f"SELECT DISTINCT {symbol_column_name} FROM {table_name};"
        symbols_df = pd.read_sql_query(query, conn)
        if symbol_column_name not in symbols_df.columns:
            logger.error(f"Columna '{symbol_column_name}' no encontrada en la tabla {table_name}.")
            return []
        return symbols_df[symbol_column_name].tolist()
    except pd.io.sql.DatabaseError as e: 
        if "no such table" in str(e).lower():
            logger.error(f"Tabla '{table_name}' no encontrada. Asegúrate de que el script 'compute_mark_price_vwap.py' se haya ejecutado correctamente.")
            return []
        logger.error(f"Error de BD al obtener símbolos de '{table_name}': {e}")
        return []
    except Exception as e:
        logger.error(f"No se pudo obtener símbolos de la tabla {table_name}: {e}")
        return []

def get_aggregated_l1_metrics(conn, symbol_id: str, freq: str) -> pd.DataFrame:
    logger.info(f"Calculando métricas L1 agregadas para {symbol_id} a frecuencia {freq}...")
    query = "SELECT ts, bid1_px, bid1_sz, ask1_px, ask1_sz FROM coinapi_orderbook_clean WHERE symbol_id = ?"
    try:
        df_ob = pd.read_sql_query(query, conn, params=(symbol_id,))
        if df_ob.empty:
            logger.warning(f"No hay datos de orderbook para calcular métricas L1 para {symbol_id}.")
            return pd.DataFrame(columns=['ts', 'mid_price_l1_agg', 'vwap_l1_agg'])

        df_ob["ts"] = pd.to_datetime(df_ob["ts"])
        for col in ["bid1_px", "bid1_sz", "ask1_px", "ask1_sz"]:
            df_ob[col] = pd.to_numeric(df_ob[col], errors='coerce')
        df_ob.dropna(subset=["ts", "bid1_px", "ask1_px", "bid1_sz", "ask1_sz"], inplace=True)

        if df_ob.empty:
            logger.warning(f"No hay datos válidos en orderbook después de limpiar NaNs para métricas L1 de {symbol_id}.")
            return pd.DataFrame(columns=['ts', 'mid_price_l1_agg', 'vwap_l1_agg'])

        df_ob["mid_price_l1"] = (df_ob["bid1_px"] + df_ob["ask1_px"]) / 2
        val_bid = df_ob["bid1_px"] * df_ob["ask1_sz"] 
        val_ask = df_ob["ask1_px"] * df_ob["bid1_sz"] 
        total_sz_cross = df_ob["bid1_sz"] + df_ob["ask1_sz"]
        df_ob["vwap_l1"] = np.where(total_sz_cross > 1e-9, (val_bid + val_ask) / total_sz_cross, np.nan)
            
        df_agg_l1 = df_ob.set_index("ts").resample(freq).agg(
            mid_price_l1_agg = ("mid_price_l1", "median"),
            vwap_l1_agg = ("vwap_l1", "median")
        ).reset_index() 
        logger.info(f"Métricas L1 agregadas para {symbol_id}: {len(df_agg_l1)} filas.")
        return df_agg_l1
    except Exception as e:
        logger.error(f"Error calculando métricas L1 agregadas para {symbol_id}: {e}", exc_info=True)
        return pd.DataFrame(columns=['ts', 'mid_price_l1_agg', 'vwap_l1_agg'])


def analyze_symbol_anomalies(
    df_anomalies_symbol: pd.DataFrame, 
    df_mark_price_symbol: pd.DataFrame, 
    df_ohlcv_symbol: pd.DataFrame,
    df_l1_metrics_symbol: pd.DataFrame,
    symbol_id: str, 
    output_dir: str
):
    logger.info(f"--- Análisis Agregado y de Series Temporales para Símbolo: {symbol_id} ---")

    if not df_anomalies_symbol.empty:
        logger.info(f"Estadísticas de 'relative_difference' para {symbol_id} ({len(df_anomalies_symbol)} anomalías):")
        desc_stats = df_anomalies_symbol['relative_difference'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        logger.info(f"\n{desc_stats}")
        top_n = 5
        logger.info(f"Top {top_n} anomalías por 'relative_difference' para {symbol_id}:")
        logger.info(f"\n{df_anomalies_symbol.nlargest(top_n, 'relative_difference')}")
        if not df_mark_price_symbol.empty:
            total_mark_price_buckets = len(df_mark_price_symbol)
            num_anomalies = len(df_anomalies_symbol)
            if total_mark_price_buckets > 0:
                percentage_anomalies = (num_anomalies / total_mark_price_buckets) * 100
                logger.info(f"Porcentaje de buckets anómalos: {percentage_anomalies:.2f}% ({num_anomalies}/{total_mark_price_buckets})")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df_anomalies_symbol['relative_difference'], kde=True, bins=50)
        plt.title(f'Distribución de Diferencia Relativa (Anomalías) - {symbol_id}')
        plt.xlabel("Diferencia Relativa (Mark Price vs OHLCV Close)"); plt.ylabel("Frecuencia"); plt.grid(True, linestyle='--', alpha=0.7)
        filepath_hist = os.path.join(output_dir, f"{symbol_id}_anomalies_rel_diff_hist.png")
        plt.savefig(filepath_hist); plt.close()
        logger.info(f"Histograma de diferencia relativa guardado en: {filepath_hist}")

        plt.figure(figsize=(10, 8))
        plt.scatter(df_anomalies_symbol['ohlcv_price_close'], df_anomalies_symbol['mark_price'], alpha=0.5, s=10, label="Anomalías")
        min_val = min(df_anomalies_symbol['ohlcv_price_close'].min(), df_anomalies_symbol['mark_price'].min())
        max_val = max(df_anomalies_symbol['ohlcv_price_close'].max(), df_anomalies_symbol['mark_price'].max())
        if pd.notna(min_val) and pd.notna(max_val): plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identidad (y=x)')
        plt.title(f'Mark Price vs OHLCV Close (Anomalías) - {symbol_id}')
        plt.xlabel("OHLCV Price Close"); plt.ylabel("Mark Price (VWAP3 Agg)"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.axis('equal')
        filepath_scatter = os.path.join(output_dir, f"{symbol_id}_anomalies_scatter_mark_vs_ohlcv.png")
        plt.savefig(filepath_scatter); plt.close()
        logger.info(f"Gráfico de dispersión de anomalías guardado en: {filepath_scatter}")
    else:
        logger.info(f"No hay anomalías para analizar para el símbolo {symbol_id}.")

    # --- Gráfico de Series Temporales ---
    logger.info(f"Intentando generar gráfico de series temporales para {symbol_id}...")
    if not df_mark_price_symbol.empty and 'ts_start' in df_mark_price_symbol.columns:
        df_plot = df_mark_price_symbol.copy()
        df_plot['ts_start'] = pd.to_datetime(df_plot['ts_start'])
        df_plot = df_plot.set_index('ts_start').sort_index()
        logger.debug(f"df_plot (base de mark_price) para {symbol_id} tiene {len(df_plot)} filas. Columnas: {df_plot.columns.tolist()}")
        logger.debug(f"df_plot head:\n{df_plot.head()}")


        if not df_ohlcv_symbol.empty and 'time_period_start' in df_ohlcv_symbol.columns:
            df_ohlcv_plot = df_ohlcv_symbol.copy()
            df_ohlcv_plot['time_period_start'] = pd.to_datetime(df_ohlcv_plot['time_period_start'])
            df_ohlcv_plot = df_ohlcv_plot.set_index('time_period_start').sort_index()
            df_plot = df_plot.join(df_ohlcv_plot[['price_close']], how='left') 
            logger.debug(f"Después de join con OHLCV, df_plot para {symbol_id} tiene {len(df_plot)} filas. Columnas: {df_plot.columns.tolist()}")
            logger.debug(f"df_plot head post OHLCV join:\n{df_plot.head()}")
        else:
            logger.warning(f"No hay datos OHLCV para unir para {symbol_id} para series temporales.")
            df_plot['price_close'] = np.nan 

        if not df_l1_metrics_symbol.empty and 'ts' in df_l1_metrics_symbol.columns:
            df_l1_plot = df_l1_metrics_symbol.copy()
            df_l1_plot.rename(columns={'ts': 'ts_start'}, inplace=True) 
            df_l1_plot['ts_start'] = pd.to_datetime(df_l1_plot['ts_start'])
            df_l1_plot = df_l1_plot.set_index('ts_start').sort_index()
            df_plot = df_plot.join(df_l1_plot[['mid_price_l1_agg', 'vwap_l1_agg']], how='left')
            logger.debug(f"Después de join con L1 metrics, df_plot para {symbol_id} tiene {len(df_plot)} filas. Columnas: {df_plot.columns.tolist()}")
            logger.debug(f"df_plot head post L1 join:\n{df_plot.head()}")
        else:
            logger.warning(f"No hay datos de métricas L1 para unir para {symbol_id} para series temporales.")
            df_plot['mid_price_l1_agg'] = np.nan
            df_plot['vwap_l1_agg'] = np.nan
        
        buffer = io.StringIO()
        df_plot.info(buf=buffer)
        info_str = buffer.getvalue()
        # Loguear info de df_plot a nivel INFO si la gráfica se va a saltar, sino a DEBUG
        log_level_for_df_info = logging.DEBUG 

        plt.figure(figsize=(18, 9))
        plot_cols_plotted = [] 
        
        series_to_plot = {
            'OHLCV Close': 'price_close',
            'Mark Price (VWAP3 Agg)': 'mark_price',
            'Mid Price L1 (Agg)': 'mid_price_l1_agg',
            'VWAP L1 (Agg)': 'vwap_l1_agg'
        }
        line_styles = ['-', '--', ':', '-.']
        
        for i, (label, col_name) in enumerate(series_to_plot.items()):
            col_exists = col_name in df_plot.columns
            has_data = df_plot[col_name].notna().any() if col_exists else False
            logger.debug(f"Chequeando para plotear '{label}' ({col_name}): Existe={col_exists}, TieneDatosNoNaN={has_data}")
            if col_exists and has_data:
                plt.plot(df_plot.index, df_plot[col_name], label=label, alpha=0.8, linewidth=1, linestyle=line_styles[i % len(line_styles)])
                plot_cols_plotted.append(label)
                logger.info(f"Ploteando '{label}' ({col_name}) para {symbol_id}.")
            else: 
                logger.warning(f"No se ploteará '{label}' ({col_name}) para {symbol_id}. Existe={col_exists}, TieneDatosNoNaN={has_data}")
        
        if not plot_cols_plotted:
            logger.warning(f"No hay columnas de precios válidas para graficar series temporales para {symbol_id}.")
            log_level_for_df_info = logging.WARNING # Elevar a warning si no se plotea nada
            plt.close() 
        else:
            plt.title(f'Comparación de Series de Precios - {symbol_id}')
            plt.xlabel("Fecha"); plt.ylabel("Precio"); plt.legend(); plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=15))
            plt.gcf().autofmt_xdate() 
            filename_series = f"{symbol_id}_aggregated_price_series.png"
            filepath_series = os.path.join(output_dir, filename_series)
            plt.savefig(filepath_series); plt.close()
            logger.info(f"Gráfico de series temporales guardado en: {filepath_series}")
        
        # Loguear info de df_plot si se saltó el gráfico o si el nivel es DEBUG
        if not plot_cols_plotted or logger.isEnabledFor(logging.DEBUG):
             logger.log(log_level_for_df_info, f"Información de df_plot para {symbol_id} (contexto series temporales):\n{info_str}")
             logger.log(log_level_for_df_info, f"Primeras filas de df_plot para {symbol_id}:\n{df_plot.head()}")

    else:
        logger.warning(f"No hay datos de mark_price (df_mark_price_symbol vacío o sin ts_start) para graficar series temporales para {symbol_id}.")
    logger.info(f"--- Fin Análisis para Símbolo: {symbol_id} ---")

def main():
    parser = argparse.ArgumentParser(description="Analiza agregadamente las anomalías del Mark Price y series de precios.")
    parser.add_argument("--db-file", default=DEFAULT_DB_FILE, help="Ruta al archivo de base de datos SQLite.")
    parser.add_argument("--symbol", nargs="+", help="Lista de símbolos específicos a analizar. Si no se provee, analiza todos los que tengan anomalías o datos de mark_price.")
    parser.add_argument("--output-dir", default=CHARTS_OUTPUT_DIR, help="Directorio para guardar los gráficos.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Nivel de logging.")
    parser.add_argument("--agg-freq", default=DEFAULT_AGG_FREQ, help="Frecuencia para agregar métricas L1 (ej. '5min', '1min').")
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    if not os.path.exists(args.db_file):
        logger.critical(f"Archivo de base de datos no encontrado: {args.db_file}"); return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir); logger.info(f"Directorio de gráficos creado: {args.output_dir}")

    conn = sqlite3.connect(args.db_file)
    
    try:
        symbols_to_analyze = args.symbol
        if not symbols_to_analyze:
            logger.info("Buscando todos los símbolos con datos de mark_price o anomalías...")
            symbols_anomalies = get_symbols_from_table(conn, "mark_price_anomalies", "symbol_id")
            symbols_markprice = get_symbols_from_table(conn, "mark_price_vwap", "symbol_id")
            symbols_to_analyze = sorted(list(set(symbols_anomalies + symbols_markprice)))
            if not symbols_to_analyze:
                logger.info("No se encontraron símbolos con datos de mark_price o anomalías para analizar.")
                return
            logger.info(f"Símbolos a analizar: {symbols_to_analyze}")

        all_anomalies_dfs = []
        for symbol_id in symbols_to_analyze:
            logger.info(f"\n{'='*15} Análisis Agregado para Símbolo: {symbol_id} {'='*15}")
            
            df_anomalies_symbol = pd.read_sql_query("SELECT symbol_id, ts_start, mark_price, ohlcv_price_close, relative_difference FROM mark_price_anomalies WHERE symbol_id = ?", conn, params=(symbol_id,))
            df_mark_price_symbol = pd.read_sql_query("SELECT symbol_id, ts_start, ts_end, mark_price, depth_sum_sz, n_snapshots FROM mark_price_vwap WHERE symbol_id = ?", conn, params=(symbol_id,))
            df_ohlcv_symbol = pd.read_sql_query("SELECT time_period_start, price_close, price_open, price_high, price_low FROM coinapi_ohlcv_clean WHERE symbol = ?", conn, params=(symbol_id,))
            
            if not df_anomalies_symbol.empty: df_anomalies_symbol['ts_start'] = pd.to_datetime(df_anomalies_symbol['ts_start'])
            if not df_mark_price_symbol.empty: df_mark_price_symbol['ts_start'] = pd.to_datetime(df_mark_price_symbol['ts_start']) # ts_end también es datetime
            if not df_ohlcv_symbol.empty: df_ohlcv_symbol['time_period_start'] = pd.to_datetime(df_ohlcv_symbol['time_period_start'])

            df_l1_metrics_symbol = get_aggregated_l1_metrics(conn, symbol_id, args.agg_freq)
            
            analyze_symbol_anomalies(df_anomalies_symbol, df_mark_price_symbol, df_ohlcv_symbol, df_l1_metrics_symbol, symbol_id, args.output_dir)
            
            if not df_anomalies_symbol.empty:
                all_anomalies_dfs.append(df_anomalies_symbol)

        if all_anomalies_dfs:
            df_all_anomalies = pd.concat(all_anomalies_dfs)
            if not df_all_anomalies.empty and 'relative_difference' in df_all_anomalies.columns:
                plt.figure(figsize=(12, max(8, len(symbols_to_analyze) * 0.5))) 
                sns.boxplot(x='relative_difference', y='symbol_id', data=df_all_anomalies, orient='h', showfliers=False) 
                plt.title('Comparación de Diferencia Relativa de Anomalías por Símbolo (sin outliers extremos del boxplot)')
                plt.xlabel("Diferencia Relativa (Mark Price vs OHLCV Close)")
                plt.ylabel("Símbolo ID")
                try: plt.xscale('log') 
                except ValueError: logger.warning("No se pudo aplicar escala logarítmica al boxplot (posiblemente por valores no positivos).")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                filepath_boxplot = os.path.join(args.output_dir, "global_anomalies_rel_diff_boxplot.png")
                plt.savefig(filepath_boxplot); plt.close()
                logger.info(f"Gráfico de cajas comparativo global guardado en: {filepath_boxplot}")
            else:
                logger.info("No hay datos de anomalías para generar el boxplot global o falta la columna 'relative_difference'.")
    except Exception as e:
        logger.critical(f"Error fatal durante el script de análisis de anomalías: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Conexión a la base de datos cerrada.")
    logger.info("<<<<<<<<<< Script de Análisis Agregado de Anomalías Completado >>>>>>>>>>")

if __name__ == "__main__":
    main()
