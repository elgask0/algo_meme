#!/usr/bin/env python3
import sqlite3
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import numpy as np
import os
import argparse
from dotenv import load_dotenv

"""
Script: compute_mark_price_vwap.py
Ubicación: scripts/compute_mark_price_vwap.py

Este script:
 1. Carga snapshots limpios de coinapi_orderbook_clean
 2. Valida esquema con Pandera
 3. Calcula VWAP a 3 niveles y suma de profundidad
 4. Agrega en buckets de 5 minutos (mediana, recuento)
 5. Inserta (append) en mark_price_vwap
"""

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
DEFAULT_FREQ = "5min"

# Esquema de validación de snapshots limpios
ORDERBOOK_CLEAN_SCHEMA = DataFrameSchema({
    "symbol_id": Column(str, nullable=False),
    "ts": Column(pa.DateTime, nullable=False),
    "date": Column(pa.DateTime, nullable=False),
    "bid1_px": Column(float, Check.ge(0), nullable=False),
    "bid1_sz": Column(float, Check.gt(0), nullable=False),
    "bid2_px": Column(float, Check.ge(0), nullable=False),
    "bid2_sz": Column(float, Check.gt(0), nullable=False),
    "bid3_px": Column(float, Check.ge(0), nullable=False),
    "bid3_sz": Column(float, Check.gt(0), nullable=False),
    "ask1_px": Column(float, Check.ge(0), nullable=False),
    "ask1_sz": Column(float, Check.gt(0), nullable=False),
    "ask2_px": Column(float, Check.ge(0), nullable=False),
    "ask2_sz": Column(float, Check.gt(0), nullable=False),
    "ask3_px": Column(float, Check.ge(0), nullable=False),
    "ask3_sz": Column(float, Check.gt(0), nullable=False),
    "flag_ob_bad_structure": Column(int, Check.isin([0,1]), nullable=False),
    "flag_spread_mad": Column(int, Check.isin([0,1]), nullable=False),
    "flag_mid_mad": Column(int, Check.isin([0,1]), nullable=False),
})

def load_orderbook_clean(db_path: str, symbol: str, date: str = None) -> pd.DataFrame:
    """Carga snapshots limpios, opcionalmente filtrando por fecha."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM coinapi_orderbook_clean WHERE symbol_id = ?"
    params = [symbol]
    if date:
        query += " AND date = ?"
        params.append(date)
    df = pd.read_sql(query, conn, params=params, parse_dates=["ts", "date"])
    print(f"[DEBUG] load_orderbook_clean: params={params}, filas tras SQL={len(df)}")
    conn.close()
    return df

def validate_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el DataFrame con Pandera."""
    ORDERBOOK_CLEAN_SCHEMA.validate(df, lazy=False)
    return df

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula VWAP y profundidad para cada snapshot."""
    df = df.copy()
    df["vwap"] = (
        df["bid1_px"] * df["bid1_sz"] +
        df["bid2_px"] * df["bid2_sz"] +
        df["bid3_px"] * df["bid3_sz"] +
        df["ask1_px"] * df["ask1_sz"] +
        df["ask2_px"] * df["ask2_sz"] +
        df["ask3_px"] * df["ask3_sz"]
    ) / (
        df["bid1_sz"] + df["bid2_sz"] + df["bid3_sz"] +
        df["ask1_sz"] + df["ask2_sz"] + df["ask3_sz"]
    )
    df["depth_sum_sz"] = (
        df["bid1_sz"] + df["bid2_sz"] + df["bid3_sz"] +
        df["ask1_sz"] + df["ask2_sz"] + df["ask3_sz"]
    )
    return df

def aggregate_mark_price(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Agrega VWAP en buckets de frecuencia dada (p.ej. '5min')."""
    df = df.set_index("ts")
    print(f"[DEBUG] aggregate_mark_price: filas antes de agrupar = {df.shape}")
    df["ts_start"] = df.index.floor(freq)
    df["ts_end"]   = df["ts_start"] + pd.Timedelta(freq)
    agg = (
        df.groupby(["symbol_id", "ts_start"], as_index=False)
          .agg(
              mark_price   = ("vwap",         "median"),
              depth_sum_sz = ("depth_sum_sz", "median"),
              n_snapshots  = ("vwap",         "count"),
              ts_end       = ("ts_end",       "first")
          )
    )
    # Formatear a ISO UTC
    agg["ts_start"] = agg["ts_start"].dt.tz_localize("UTC").astype(str)
    agg["ts_end"]   = agg["ts_end"].dt.tz_localize("UTC").astype(str)
    return agg


def cross_check_with_ohlcv(conn, df_mark, symbol, threshold=0.005):
    """
    Cross-check mark price against OHLCV close:
    - Loads price_close from coinapi_ohlcv_clean over the same ts_start range.
    - Computes relative difference and flags anomalies.
    """
    # Load OHLCV close prices
    q = """
    SELECT time_period_start AS ts_start, price_close
    FROM coinapi_ohlcv_clean
    WHERE symbol = ?
      AND time_period_start BETWEEN ? AND ?
    """
    ohlcv = pd.read_sql(q, conn, params=[
        symbol,
        df_mark.ts_start.min(),
        df_mark.ts_start.max()
    ], parse_dates=["ts_start"])
    # Alinear tipos de ts_start: convertir ambos a datetime sin zona
    df_mark = df_mark.copy()
    df_mark["ts_start"] = pd.to_datetime(df_mark["ts_start"], utc=True).dt.tz_localize(None)
    ohlcv["ts_start"]   = ohlcv["ts_start"].dt.tz_localize(None)
    # Merge y cálculo
    df = df_mark.merge(ohlcv, on="ts_start", how="left")
    df["rel_diff"] = (df["mark_price"] - df["price_close"]).abs() / df["price_close"]
    df["flag_rel_diff"] = df["rel_diff"] > threshold
    return df

def persist_mark_price(df: pd.DataFrame, db_path: str):
    """Inserta o actualiza los resultados en SQLite."""
    # Keep only the table's columns
    df_to_write = df[["symbol_id","ts_start","ts_end","mark_price","depth_sum_sz","n_snapshots"]].copy()
    # Ensure datetime columns are strings for SQLite binding
    df_to_write["ts_start"] = df_to_write["ts_start"].astype(str)
    df_to_write["ts_end"]   = df_to_write["ts_end"].astype(str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    records = df_to_write.to_dict(orient='records')
    cursor.executemany(
        "INSERT OR IGNORE INTO mark_price_vwap (symbol_id, ts_start, ts_end, mark_price, depth_sum_sz, n_snapshots) VALUES (?, ?, ?, ?, ?, ?)",
        [(r['symbol_id'], r['ts_start'], r['ts_end'], r['mark_price'], r['depth_sum_sz'], r['n_snapshots']) for r in records]
    )
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Compute VWAP-based mark price (3 levels, 5min)"
    )
    parser.add_argument(
        "--symbol", "-s",
        nargs="*",
        help="Símbolos a procesar (p.ej. BTC_USDT); si no se indica, procesa todos los symbols en coinapi_orderbook_clean",
    )
    parser.add_argument(
        "--db-file", "-d",
        default=DEFAULT_DB_FILE,
        help="Ruta a la DB SQLite"
    )
    parser.add_argument(
        "--freq", "-f",
        default=DEFAULT_FREQ,
        help="Frecuencia de bucket (ej. '5min')"
    )
    parser.add_argument(
        "--date",
        help="Fecha YYYY-MM-DD para filtrar snapshots"
    )
    args = parser.parse_args()

    # Determinar símbolos a procesar; si no se pasa ninguno, tomar todos de la tabla clean
    if not args.symbol:
        conn_all = sqlite3.connect(args.db_file)
        cur_all = conn_all.cursor()
        cur_all.execute("SELECT DISTINCT symbol_id FROM coinapi_orderbook_clean;")
        symbols = [r[0] for r in cur_all.fetchall()]
        conn_all.close()
    else:
        symbols = args.symbol

    for symbol in symbols:
        print(f"Computando mark price VWAP para {symbol} (freq={args.freq})")
        df_ob = load_orderbook_clean(args.db_file, symbol, args.date)
        print(f"[DEBUG] snapshots cargados para {symbol}: {len(df_ob)}")
        if df_ob.empty:
            print(f"No hay snapshots para {symbol}. Continuando.")
            continue
        df_ob = validate_orderbook(df_ob)
        df_v  = compute_vwap(df_ob)
        df_agg = aggregate_mark_price(df_v, args.freq)
        if df_agg.empty:
            print(f"No hay buckets para insertar para {symbol}. Continuando.")
            continue
        # Cross-check against OHLCV close and record anomalies
        with sqlite3.connect(args.db_file) as conn_x:
            df_agg = cross_check_with_ohlcv(conn_x, df_agg, symbol)
            anomalies = df_agg[df_agg.flag_rel_diff]
            print(f"[DEBUG] buckets totales antes de filtrar anomalías para {symbol}: {len(df_agg)}")
            print(f"[DEBUG] anomalías encontradas para {symbol}: {len(anomalies)}")
            if not anomalies.empty:
                anomalies.to_sql("mark_price_anomalies", conn_x,
                                 if_exists="append", index=False)
            df_agg = df_agg[~df_agg.flag_rel_diff]
        # Persist cleaned buckets
        persist_mark_price(df_agg, args.db_file)
        print(f"Insertados {len(df_agg)} buckets en mark_price_vwap para {symbol} (sin anomalías).")

if __name__ == "__main__":
    main()