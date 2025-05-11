#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
from dotenv import load_dotenv

"""
Script: compute_perp_synthetic.py
Ubicación: scripts/compute_perp_synthetic.py

Este script:
 1. Determina símbolos perp a procesar (de funding rate history).
 2. Para cada perp symbol:
    a) Mapea al symbol spot subyacente.
    b) Carga mark_price_vwap del spot (5-min buckets).
    c) Carga funding rate history (timestamp y collect_cycle).
    d) Resample funding a rejilla de freq, calcula funding_per_bucket.
    e) Fusiona spot y funding, calcula carry_cum y perp_price.
    f) Persiste resultados en la tabla perp_synthetic.

Dependencias: pandas, numpy, python-dotenv
"""

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
DEFAULT_FREQ = "5min"


def map_perp_to_spot(perp_symbol: str) -> str:
    """
    Dado un symbol perp (p.ej. MEXCFTS_PERP_SPX_USDT), devuelve el symbol spot subyacente.
    Convención: 'FTS_PERP_' -> 'SPOT_'
    """
    if "FTS_PERP_" in perp_symbol:
        tail = perp_symbol.split("FTS_PERP_")[1]
        return f"MEXC_SPOT_{tail}"
    return perp_symbol


def load_spot(db_path: str, spot_symbol: str, date_from: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    q = "SELECT symbol_id, ts_start, ts_end, mark_price AS spot_price FROM mark_price_vwap WHERE symbol_id = ?"
    params = [spot_symbol]
    if date_from:
        q += " AND ts_start >= ?"
        params.append(date_from)
    df = pd.read_sql(q, conn, params=params, parse_dates=["ts_start", "ts_end"])
    conn.close()
    return df


def load_funding(db_path: str, perp_symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    # collect_cycle in seconds, convert to hours
    q = """
    SELECT ts AS ts_funding,
           funding_rate,
           collect_cycle/3600.0 AS hours
    FROM mexc_funding_rate_history
    WHERE symbol = ?
    """
    df = pd.read_sql(q, conn, params=[perp_symbol], parse_dates=["ts_funding"])
    conn.close()
    return df


def compute_perp(df_spot: pd.DataFrame, df_fund: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Build a synthetic perp using the median funding rate over all periods.
    funding_rate in df_fund applies over each collect_cycle (hours*3600 seconds).
    We distribute that cost evenly across each freq bucket.
    """
    # Calculate bucket duration seconds
    bucket_sec = pd.Timedelta(freq).total_seconds()

    # Compute median funding_rate and median collect_cycle
    med_rate = df_fund["funding_rate"].median()
    med_cycle_sec = df_fund["hours"].median() * 3600.0

    # Per-bucket funding cost based on median rate
    fund_per_bucket = med_rate * (bucket_sec / med_cycle_sec)

    # Build the synthetic perp DataFrame
    df = df_spot.copy()
    df["funding_per_bucket"] = fund_per_bucket
    df["funding_cum"] = df["funding_per_bucket"].cumsum()
    df["perp_price"] = df["spot_price"] * np.exp(df["funding_cum"])
    df["funding_cum"] = df["funding_cum"]  # rename for consistency

    return df


def persist_perp(df: pd.DataFrame, db_path: str):
    # Columnas destino
    df_to_write = df[["symbol_id","ts_start","ts_end","perp_price","funding_cum","spot_price"]].copy()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    sql = (
        "INSERT OR REPLACE INTO perp_synthetic "
        "(symbol_id, ts_start, ts_end, perp_price, funding_cum, spot_price) "
        "VALUES (?, ?, ?, ?, ?, ?);"
    )
    batch = []
    for row in df_to_write.itertuples(index=False, name=None):
        # funding_cum ya calculado
        batch.append((row[0], row[1].isoformat(), row[2].isoformat(), float(row[3]), float(row[4]), float(row[5])))
    cur.executemany(sql, batch)
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute synthetic perp prices using spot VWAP and funding rates"
    )
    parser.add_argument(
        "--symbol", "-s",
        nargs="*",
        help="Símbolos perp a procesar (p.ej. MEXCFTS_PERP_SPX_USDT); si no se indica, procesa todos en funding history",
    )
    parser.add_argument(
        "--db-file", "-d",
        default=DEFAULT_DB_FILE,
        help="Ruta a la DB SQLite"
    )
    parser.add_argument(
        "--freq", "-f",
        default=DEFAULT_FREQ,
        help="Frecuencia de buckets para funding (p.ej. '5min')"
    )
    parser.add_argument(
        "--date", help="Fecha YYYY-MM-DD para filtrar desde ts_start"
    )
    args = parser.parse_args()

    # Determinar símbolos perp a procesar
    conn = sqlite3.connect(args.db_file)
    if args.symbol:
        symbols = args.symbol
    else:
        # Use symbols that have spot data with "SPOT" in symbol_id
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT m.symbol_id
            FROM mark_price_vwap m
            WHERE m.symbol_id LIKE '%SPOT_%'
        """)
        symbols = [r[0] for r in cur.fetchall()]
    conn.close()

    for spot_sym in symbols:
        # Map spot symbol to perp symbol
        if "SPOT_" in spot_sym:
            tail = spot_sym.split("SPOT_")[1]
            sym = f"MEXCFTS_PERP_{tail}"
        else:
            sym = spot_sym  # fallback if no SPOT_ in symbol

        print(f"Procesando perp_synthetic para {sym} (spot: {spot_sym})")

        df_spot = load_spot(args.db_file, spot_sym, args.date)
        if df_spot.empty:
            print(f"  No hay datos spot para {spot_sym}, salto.")
            continue
        df_fund = load_funding(args.db_file, sym)
        if df_fund.empty:
            print(f"  No hay datos funding para {sym}, salto.")
            continue

        df_out = compute_perp(df_spot, df_fund, args.freq)
        if df_out.empty:
            print("  ⚠️ Resultado vacío, salto.")
            continue

        # Adaptar columna funding_cum a DB naming
        df_out = df_out.rename(columns={"carry_cum": "funding_cum"})
        df_out["symbol_id"] = sym

        persist_perp(df_out, args.db_file)
        print(f"  Insertados {len(df_out)} filas en perp_synthetic para {sym}\n")

if __name__ == "__main__":
    main()
