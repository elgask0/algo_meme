#!/usr/bin/env python
# analyse_funding.py
"""
Analiza la serie de funding de un perp y sugiere la mejor imputación
( funding = 0  vs  funding = constante ) para periodos previos al launch.

Salidas:
  • Resumen estadístico impreso en terminal
  • Gráfica time‑series + histograma guardada en funding_<symbol>.png
"""

import sqlite3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def load_funding(db_path: str, symbol: str, table: str) -> pd.DataFrame:
    """Lee funding rate de la tabla `coinapi_funding` (ajusta si tu tabla se llama distinto)."""
    query = f"""
    SELECT ts, funding_rate
    FROM {table}
    WHERE symbol = ?
    ORDER BY ts
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=[symbol], parse_dates=["ts"])
    if df.empty:
        raise RuntimeError(f"No hay datos de funding para {symbol} en {db_path}.")
    df = df.drop_duplicates(subset=["ts"]).set_index("ts").sort_index()
    return df

def describe_series(df: pd.DataFrame) -> None:
    fr = df["funding_rate"]
    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(f"- Fechas      : {df.index.min()}  →  {df.index.max()}")
    print(f"- Observaciones: {len(df):,}")
    print(fr.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    print("\n--- Proporción de signos ---")
    pos = (fr > 0).mean()*100
    neg = (fr < 0).mean()*100
    zero = (fr == 0).mean()*100
    print(f"+  {pos:5.2f}%   |   –  {neg:5.2f}%   |   0  {zero:5.2f}%")
    print("\n--- Test de estacionariedad (ADF) ---")
    adf_stat, pval, *_ = adfuller(fr.dropna())
    print(f"ADF statistic = {adf_stat:8.4f}   |   p‑value = {pval:6.4f}")
    print("  ↳ p‑value < 0.05 ⇒ se rechaza raíz unitaria (serie estacionaria).\n")

def plot_series(df: pd.DataFrame, symbol: str) -> None:
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(211)
    df["funding_rate"].plot(ax=ax1)
    ax1.set_title(f"Funding rate – {symbol}")
    ax1.set_ylabel("rate")
    ax1.grid(True, linestyle="--", alpha=.4)

    ax2 = fig.add_subplot(212)
    df["funding_rate"].hist(bins=100, ax=ax2, density=True)
    ax2.set_xlabel("funding_rate")
    ax2.set_title("Distribución")
    ax2.grid(True, linestyle="--", alpha=.4)

    fig.tight_layout()
    out_file = f"funding_{symbol}.png"
    fig.savefig(out_file, dpi=120)
    print(f"Gráfica guardada en {out_file}\n")

def suggest_method(df: pd.DataFrame) -> None:
    fr = df["funding_rate"].dropna()
    mean_abs_bps = fr.abs().mean()*1e4        # bps (suponiendo funding ya en proporción)
    med_bps       = fr.median()*1e4
    max_gap_hours = (df.index.to_series().diff().dropna().max().total_seconds()/3600)

    print("===== SUGGESTION =====")
    if mean_abs_bps < 0.5 and med_bps < 0.25:
        print("• Funding medio ≈ 0 ⇒ usa **funding = 0** para el tramo pre‑lanzamiento.")
    elif mean_abs_bps < 1 and abs(med_bps) < 0.75:
        print(f"• Funding estable (|μ|<{mean_abs_bps:.2f} bps) ⇒ "
              "imputa **funding = mediana histórica**.")
    else:
        print(f"• Funding con sesgo significativo (μ={mean_abs_bps:.2f} bps, "
              f"mediana={med_bps:.2f} bps).")
        print("  └─ Opción conservadora: funding=0")
        print("  └─ Opción direccional : funding = mediana o media histórica.")
    if max_gap_hours > 10:
        print(f"• Ojo: hay huecos de hasta {max_gap_hours:.1f} h en la serie; "
              "rellénalos con 0 antes de imputar.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", "-s", required=True, help="Símbolo funding PERP")
    parser.add_argument("--db-file", "-d", default="trading_data.db",
                        help="Ruta a la base SQLite (por defecto: trading_data.db)")
    parser.add_argument("--table", "-t",
                        default="coinapi_funding",
                        help="Nombre de la tabla de funding en la base de datos")
    args = parser.parse_args()

    df = load_funding(args.db_file, args.symbol, args.table)
    describe_series(df)
    plot_series(df, args.symbol)
    suggest_method(df)

if __name__ == "__main__":
    main()